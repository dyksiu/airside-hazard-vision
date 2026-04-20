import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class HailoYOLOSegModel:
    """
    Minimal wrapper for YOLOv8 segmentation HEF models running on Hailo.

    Inference is executed on the Hailo device, while postprocessing (NMS,
    mask decoding, ROI/mask resizing) is executed on the host CPU.
    """

    def __init__(
        self,
        hef_path,
        labels=None,
        default_labels=None,
        iou_threshold=0.7,
        pre_nms_topk_per_scale=256,
        pre_nms_topk_total=384,
        post_nms_max_det=64,
        mask_threshold=0.4,
    ):
        self.hef_path = os.fspath(hef_path)
        self.iou_threshold = float(iou_threshold)
        self._closed = False

        self._hp = None
        self._format_order = None
        self.target = None
        self.hef = None
        self.infer_model = None
        self.config_ctx = None
        self.configured_model = None
        self.last_infer_job = None
        self.last_profile_ms = {}

        self.input_height = 0
        self.input_width = 0
        self.input_channels = 3
        self.nms_postprocess_enabled = False
        self.arch = None
        self.reg_max = None
        self.mask_channels = 32
        self.num_classes = 0

        self.pre_nms_topk_per_scale = max(32, int(pre_nms_topk_per_scale))
        self.pre_nms_topk_total = max(self.pre_nms_topk_per_scale, int(pre_nms_topk_total))
        self.post_nms_max_det = max(1, int(post_nms_max_det))
        self.mask_threshold = float(mask_threshold)

        self._center_cache = {}
        self._reg_range = None

        self._load_runtime()
        self._configure_model()
        self._configure_labels(labels=labels, default_labels=default_labels)

    def _load_runtime(self):
        try:
            import hailo_platform as hp
            from hailo_platform.pyhailort.pyhailort import FormatOrder
        except ImportError as exc:
            raise ImportError(
                "Brakuje pakietu hailo_platform. Na Raspberry Pi z Hailo AI HAT "
                "uruchom aplikację w środowisku Hailo / z zainstalowanym PyHailoRT."
            ) from exc
        self._hp = hp
        self._format_order = FormatOrder

    def _configure_model(self):
        hp = self._hp
        params = hp.VDevice.create_params()
        params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"

        self.target = hp.VDevice(params)
        self.hef = hp.HEF(self.hef_path)
        self.infer_model = self.target.create_infer_model(self.hef_path)
        self.infer_model.set_batch_size(1)

        first_output = self.infer_model.outputs[0]
        self.nms_postprocess_enabled = (
            first_output.format.order == self._format_order.HAILO_NMS_WITH_BYTE_MASK
        )
        if self.nms_postprocess_enabled:
            raise NotImplementedError(
                "Ten backend aplikacji obsługuje obecnie surowe modele YOLOv8-seg HEF "
                "(postprocessing na CPU), a nie HEF z wbudowanym Hailo NMS+mask."
            )

        for output in self.infer_model.outputs:
            self.infer_model.output(output.name).set_format_type(hp.FormatType.FLOAT32)

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(0)

        input_info = self.hef.get_input_vstream_infos()[0]
        ishape = tuple(int(x) for x in input_info.shape)
        if len(ishape) == 3:
            self.input_height, self.input_width, self.input_channels = ishape
        elif len(ishape) == 2:
            self.input_height, self.input_width = ishape
            self.input_channels = 3
        else:
            raise ValueError(f"Nieobsługiwany kształt wejścia HEF: {ishape}")

        self._infer_architecture_from_outputs()

    def _infer_architecture_from_outputs(self):
        infos = self.hef.get_output_vstream_infos()
        outputs = []
        for info in infos:
            shape = tuple(int(x) for x in info.shape)
            if len(shape) != 3:
                continue
            h, w, c = shape
            outputs.append({"name": info.name, "h": h, "w": w, "c": c})

        spatial = sorted({o["h"] for o in outputs if o["h"] == o["w"]})
        if not spatial:
            raise ValueError("Nie udało się rozpoznać wyjść HEF.")

        group_sizes = [s for s in spatial if s != max(spatial)]
        if len(group_sizes) != 3:
            raise ValueError(
                "Backend HAILO w tej aplikacji oczekuje HEF typu YOLOv8-seg "
                f"z 3 poziomami skali. Wykryte skale: {spatial}"
            )

        reg_counts = []
        cls_counts = []
        for size in group_sizes:
            group = [o for o in outputs if o["h"] == size and o["w"] == size]
            if len(group) < 3:
                raise ValueError(
                    "Nie udało się jednoznacznie rozpoznać wyjść regresji/klas/masek "
                    f"dla skali {size}x{size}."
                )
            group = sorted(group, key=lambda x: x["c"])
            coeff = None
            for item in group:
                if item["c"] == 32:
                    coeff = item
                    break
            if coeff is None:
                coeff = min(group, key=lambda x: abs(x["c"] - 32))
            reg = max(group, key=lambda x: x["c"])
            cls = [item for item in group if item["name"] not in {coeff["name"], reg["name"]}]
            if not cls:
                raise ValueError("Nie udało się ustalić liczby klas modelu HEF.")
            cls = cls[0]
            reg_counts.append(reg["c"])
            cls_counts.append(cls["c"])

        self.arch = "yolov8_seg"
        self.reg_max = int(max(reg_counts) // 4 - 1)
        self.num_classes = int(max(cls_counts))
        self.mask_channels = 32
        self._reg_range = np.arange(self.reg_max + 1, dtype=np.float32)

    def _configure_labels(self, labels=None, default_labels=None):
        labels_list = []
        if isinstance(labels, (list, tuple)):
            labels_list = [str(x).strip() for x in labels if str(x).strip()]
        elif isinstance(labels, str) and Path(labels).exists():
            labels_list = self._read_labels_file(labels)
        elif isinstance(default_labels, (list, tuple)):
            labels_list = [str(x).strip() for x in default_labels if str(x).strip()]

        if len(labels_list) == self.num_classes + 1 and labels_list[0].lower() in {"background", "tlo"}:
            labels_list = labels_list[1:]

        if len(labels_list) != self.num_classes:
            labels_list = [f"class_{idx}" for idx in range(self.num_classes)]

        self.labels = labels_list
        self.names = {idx: name for idx, name in enumerate(self.labels)}
        self.name = os.path.basename(self.hef_path)

    @staticmethod
    def _read_labels_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_preprocessed_rgb(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        scale = min(self.input_width / iw, self.input_height / ih)
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_w = (self.input_width - new_w) // 2
        pad_h = (self.input_height - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        meta = {
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "orig_h": ih,
            "orig_w": iw,
        }
        return padded, meta

    def _create_binding(self, frame_rgb):
        output_buffers = {
            output.name: np.empty(tuple(int(x) for x in output.shape), dtype=np.float32)
            for output in self.infer_model.outputs
        }
        binding = self.configured_model.create_bindings(output_buffers=output_buffers)
        input_buf = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        binding.input().set_buffer(input_buf)
        binding._input_ref = input_buf
        binding._output_refs = output_buffers
        return binding

    def _run_raw_inference(self, frame_rgb):
        holder = {}
        event = threading.Event()
        binding = self._create_binding(frame_rgb)
        output_names = [output.name for output in self.infer_model.outputs]

        def callback(*cb_args, **cb_kwargs):
            try:
                completion_info = cb_kwargs.get("completion_info")
                bindings_list = cb_kwargs.get("bindings_list")

                if completion_info is None and len(cb_args) >= 1:
                    completion_info = cb_args[0]
                if bindings_list is None and len(cb_args) >= 2:
                    bindings_list = cb_args[1]

                if completion_info is not None and getattr(completion_info, "exception", None):
                    holder["error"] = completion_info.exception
                    return

                if not bindings_list:
                    bindings_list = [binding]

                b = bindings_list[0]
                result = {}
                for name in output_names:
                    arr = b.output(name).get_buffer()
                    arr = np.asarray(arr)
                    if arr.ndim == 3:
                        arr = np.expand_dims(arr, axis=0)
                    result[name] = arr
                holder["result"] = result
            except Exception as exc:
                holder["error"] = exc
            finally:
                event.set()

        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        try:
            job = self.configured_model.run_async([binding], callback=callback)
        except TypeError:
            job = self.configured_model.run_async([binding], callback)
        self.last_infer_job = job
        job.wait(10000)
        event.wait(10.0)

        if "error" in holder:
            raise RuntimeError(holder["error"])
        if "result" not in holder:
            raise RuntimeError("Brak wyniku inferencji z Hailo.")
        return holder["result"]

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _xywh_to_xyxy(boxes):
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    def _get_centers(self, size, stride):
        key = (int(size), int(stride))
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        grid_x = np.arange(size, dtype=np.float32) + 0.5
        grid_y = np.arange(size, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.reshape(-1) * float(stride)
        ct_col = grid_x.reshape(-1) * float(stride)
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1).astype(np.float32)
        self._center_cache[key] = center
        return center

    def _decode_yolov8_boxes(self, raw_boxes_per_scale):
        decoded = []
        reg_range = self._reg_range
        for raw, stride in raw_boxes_per_scale:
            _, h, w, _ = raw.shape
            center = self._get_centers(h, stride)
            dist = raw.reshape((1, h * w, 4, self.reg_max + 1)).astype(np.float32, copy=False)
            dist = self._softmax(dist, axis=-1)
            dist = np.sum(dist * reg_range.reshape((1, 1, 1, -1)), axis=-1)
            dist = dist * float(stride)
            dist = np.concatenate([dist[:, :, :2] * (-1), dist[:, :, 2:]], axis=-1)
            decoded_xyxy = np.expand_dims(center, axis=0) + dist

            xmin = decoded_xyxy[:, :, 0]
            ymin = decoded_xyxy[:, :, 1]
            xmax = decoded_xyxy[:, :, 2]
            ymax = decoded_xyxy[:, :, 3]
            xywh = np.transpose([
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                xmax - xmin,
                ymax - ymin,
            ], [1, 2, 0])
            decoded.append(xywh)
        return np.concatenate(decoded, axis=1)

    def _decode_selected_boxes(self, reg, size, stride, selected_idx):
        if selected_idx.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        reg_flat = reg.reshape((-1, 4, self.reg_max + 1)).astype(np.float32, copy=False)
        dist = reg_flat[selected_idx]
        dist = self._softmax(dist, axis=-1)
        dist = np.sum(dist * self._reg_range.reshape((1, 1, -1)), axis=-1)
        dist = dist * float(stride)
        dist[:, :2] *= -1.0
        center = self._get_centers(size, stride)[selected_idx]
        return center + dist

    def _select_candidates_for_scale(self, cls, coeff, reg, size, conf_thres):
        cls_flat = cls.reshape((-1, cls.shape[-1])).astype(np.float32, copy=False)
        coeff_flat = coeff.reshape((-1, coeff.shape[-1])).astype(np.float32, copy=False)

        class_ids = np.argmax(cls_flat, axis=1).astype(np.int32)
        class_scores = cls_flat[np.arange(cls_flat.shape[0]), class_ids]
        selected_idx = np.flatnonzero(class_scores >= conf_thres)
        if selected_idx.size == 0:
            return None

        selected_scores = class_scores[selected_idx]
        if selected_idx.size > self.pre_nms_topk_per_scale:
            rel_idx = np.argpartition(selected_scores, -self.pre_nms_topk_per_scale)[-self.pre_nms_topk_per_scale:]
            selected_idx = selected_idx[rel_idx]
            selected_scores = class_scores[selected_idx]

        order = np.argsort(selected_scores)[::-1]
        selected_idx = selected_idx[order]
        selected_scores = selected_scores[order].astype(np.float32, copy=False)
        selected_class_ids = class_ids[selected_idx]
        selected_coeffs = coeff_flat[selected_idx]
        stride = self.input_height // int(size)
        selected_boxes = self._decode_selected_boxes(reg, size, stride, selected_idx)
        return selected_boxes, selected_coeffs, selected_class_ids, selected_scores

    @staticmethod
    def _iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
        area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _nms(self, boxes, scores, classes, iou_thresh):
        keep = []
        order = np.argsort(scores)[::-1]
        while order.size > 0 and len(keep) < self.post_nms_max_det:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            same_class = classes[rest] == classes[i]
            suppress = np.zeros(rest.shape[0], dtype=bool)
            if np.any(same_class):
                ious = self._iou(boxes[i], boxes[rest[same_class]])
                suppress[same_class] = ious > iou_thresh
            order = rest[~suppress]
        return np.array(keep, dtype=int)

    @staticmethod
    def _crop_mask_to_boxes(masks, boxes):
        out = np.zeros_like(masks)
        h, w = masks.shape[1:]
        boxes = np.round(boxes).astype(int)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
        return out

    def _decode_mask_probs(self, protos, masks_in):
        if masks_in.size == 0:
            mh, mw, _ = protos.shape
            return np.zeros((0, mh, mw), dtype=np.float32)
        mh, mw, c = protos.shape
        protos_flat = protos.reshape(-1, c).T.astype(np.float32, copy=False)
        masks = masks_in @ protos_flat
        return self._sigmoid(masks).reshape(-1, mh, mw).astype(np.float32, copy=False)

    def _project_masks_to_original(self, mask_probs, boxes_xyxy, boxes_orig, meta):
        orig_h = int(meta["orig_h"])
        orig_w = int(meta["orig_w"])
        if mask_probs.size == 0:
            return []

        mh = int(mask_probs.shape[1])
        mw = int(mask_probs.shape[2])
        sx = float(mw) / float(self.input_width)
        sy = float(mh) / float(self.input_height)
        projected = []

        for idx in range(mask_probs.shape[0]):
            x1i, y1i, x2i, y2i = boxes_xyxy[idx]
            x1i = int(np.floor(np.clip(x1i, 0, self.input_width - 1)))
            y1i = int(np.floor(np.clip(y1i, 0, self.input_height - 1)))
            x2i = int(np.ceil(np.clip(x2i, x1i + 1, self.input_width)))
            y2i = int(np.ceil(np.clip(y2i, y1i + 1, self.input_height)))
            if x2i <= x1i or y2i <= y1i:
                projected.append(None)
                continue

            x1p = int(np.floor(x1i * sx))
            y1p = int(np.floor(y1i * sy))
            x2p = int(np.ceil(x2i * sx))
            y2p = int(np.ceil(y2i * sy))
            x1p = int(np.clip(x1p, 0, mw - 1))
            y1p = int(np.clip(y1p, 0, mh - 1))
            x2p = int(np.clip(x2p, x1p + 1, mw))
            y2p = int(np.clip(y2p, y1p + 1, mh))

            crop = mask_probs[idx, y1p:y2p, x1p:x2p]
            if crop.size == 0:
                projected.append(None)
                continue

            x1o, y1o, x2o, y2o = boxes_orig[idx]
            x1o = int(np.floor(np.clip(x1o, 0, orig_w - 1)))
            y1o = int(np.floor(np.clip(y1o, 0, orig_h - 1)))
            x2o = int(np.ceil(np.clip(x2o, x1o + 1, orig_w)))
            y2o = int(np.ceil(np.clip(y2o, y1o + 1, orig_h)))
            if x2o <= x1o or y2o <= y1o:
                projected.append(None)
                continue

            target_w = x2o - x1o
            target_h = y2o - y1o
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            binary = resized > self.mask_threshold
            if not np.any(binary):
                projected.append(None)
                continue

            full_mask = np.zeros((orig_h, orig_w), dtype=bool)
            full_mask[y1o:y2o, x1o:x2o] = binary
            projected.append(full_mask)

        return projected

    def _decode_original_boxes(self, boxes_xyxy, meta):
        boxes = boxes_xyxy.copy().astype(np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_w"]) / max(meta["scale"], 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_h"]) / max(meta["scale"], 1e-6)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta["orig_w"] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta["orig_h"] - 1)
        return boxes

    def _decode_original_mask(self, mask, meta):
        y1 = int(meta["pad_h"])
        y2 = int(self.input_height - meta["pad_h"])
        x1 = int(meta["pad_w"])
        x2 = int(self.input_width - meta["pad_w"])
        cropped = mask[y1:y2, x1:x2]
        if cropped.shape[:2] != (meta["orig_h"], meta["orig_w"]):
            cropped = cv2.resize(
                cropped.astype(np.uint8),
                (meta["orig_w"], meta["orig_h"]),
                interpolation=cv2.INTER_NEAREST,
            )
        return cropped.astype(bool)

    @property
    def input_h(self):
        return self.input_height

    @property
    def input_w(self):
        return self.input_width

    def infer(self, frame_bgr, conf_thres=0.25, iou_thres=None):
        if self._closed:
            raise RuntimeError("Model Hailo został już zamknięty.")

        iou_thres = self.iou_threshold if iou_thres is None else float(iou_thres)
        profile = {}

        t0 = time.perf_counter()
        frame_rgb, meta = self._build_preprocessed_rgb(frame_bgr)
        profile["hailo_preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw = self._run_raw_inference(frame_rgb)
        profile["hailo_infer_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        detections, post_profile = self._postprocess_yolov8_seg(
            raw,
            meta,
            conf_thres=float(conf_thres),
            iou_thres=iou_thres,
        )
        profile["hailo_postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        profile.update(post_profile)
        self.last_profile_ms = profile
        return detections

    def _postprocess_yolov8_seg(self, raw_outputs, meta, conf_thres=0.25, iou_thres=0.7):
        profile = {}

        t0 = time.perf_counter()
        groups = {}
        proto = None
        for arr in raw_outputs.values():
            shape = tuple(int(x) for x in arr.shape)
            if len(shape) != 4:
                continue
            _, h, w, _ = shape
            if h == w and h in {20, 40, 80}:
                groups.setdefault(h, []).append(arr.astype(np.float32, copy=False))
            elif h == w and h == 160:
                proto = arr.astype(np.float32, copy=False)
        profile["hailo_post_groups_ms"] = (time.perf_counter() - t0) * 1000.0

        if proto is None or len(groups) != 3:
            raise ValueError("Nie udało się dopasować wyjść HEF do YOLOv8-seg.")

        t0 = time.perf_counter()
        candidate_boxes = []
        candidate_coeffs = []
        candidate_class_ids = []
        candidate_scores = []

        for size in sorted(groups.keys()):
            arrs = groups[size]
            arrs_sorted = sorted(arrs, key=lambda a: a.shape[-1])
            coeff = None
            for item in arrs_sorted:
                if item.shape[-1] == self.mask_channels:
                    coeff = item
                    break
            if coeff is None:
                coeff = min(arrs_sorted, key=lambda a: abs(a.shape[-1] - self.mask_channels))
            reg = max(arrs_sorted, key=lambda a: a.shape[-1])
            cls_candidates = [item for item in arrs_sorted if id(item) not in {id(coeff), id(reg)}]
            if not cls_candidates:
                raise ValueError("Nie udało się rozpoznać tensora klas dla YOLOv8-seg.")
            cls = cls_candidates[0]

            selected = self._select_candidates_for_scale(cls[0], coeff[0], reg[0], size, conf_thres)
            if selected is None:
                continue
            boxes_xyxy, coeffs, class_ids, class_scores = selected
            candidate_boxes.append(boxes_xyxy)
            candidate_coeffs.append(coeffs)
            candidate_class_ids.append(class_ids)
            candidate_scores.append(class_scores)

        profile["hailo_post_decode_ms"] = (time.perf_counter() - t0) * 1000.0

        if not candidate_boxes:
            profile["hailo_post_filter_ms"] = 0.0
            profile["hailo_post_nms_ms"] = 0.0
            profile["hailo_mask_decode_ms"] = 0.0
            profile["hailo_post_scale_boxes_ms"] = 0.0
            profile["hailo_post_project_masks_ms"] = 0.0
            profile["hailo_post_candidates"] = 0
            return [], profile

        boxes_xyxy = np.concatenate(candidate_boxes, axis=0).astype(np.float32, copy=False)
        coeffs = np.concatenate(candidate_coeffs, axis=0).astype(np.float32, copy=False)
        class_ids = np.concatenate(candidate_class_ids, axis=0).astype(np.int32, copy=False)
        class_scores = np.concatenate(candidate_scores, axis=0).astype(np.float32, copy=False)

        t0 = time.perf_counter()
        if class_scores.shape[0] > self.pre_nms_topk_total:
            top_idx = np.argpartition(class_scores, -self.pre_nms_topk_total)[-self.pre_nms_topk_total:]
            top_order = np.argsort(class_scores[top_idx])[::-1]
            top_idx = top_idx[top_order]
            boxes_xyxy = boxes_xyxy[top_idx]
            coeffs = coeffs[top_idx]
            class_ids = class_ids[top_idx]
            class_scores = class_scores[top_idx]
        profile["hailo_post_filter_ms"] = (time.perf_counter() - t0) * 1000.0
        profile["hailo_post_candidates"] = int(class_scores.shape[0])

        t0 = time.perf_counter()
        nms_keep = self._nms(boxes_xyxy, class_scores, class_ids, iou_thres)
        boxes_xyxy = boxes_xyxy[nms_keep]
        coeffs = coeffs[nms_keep]
        class_ids = class_ids[nms_keep]
        class_scores = class_scores[nms_keep]
        profile["hailo_post_nms_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        mask_probs = self._decode_mask_probs(proto[0], coeffs)
        profile["hailo_mask_decode_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        boxes_orig = self._decode_original_boxes(boxes_xyxy, meta)
        profile["hailo_post_scale_boxes_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        projected_masks = self._project_masks_to_original(mask_probs, boxes_xyxy, boxes_orig, meta)
        detections = []
        for idx, mask_orig in enumerate(projected_masks):
            if mask_orig is None:
                continue
            x1, y1, x2, y2 = boxes_orig[idx]
            detections.append({
                "class_id": int(class_ids[idx]),
                "class_name": self.names.get(int(class_ids[idx]), f"class_{int(class_ids[idx])}"),
                "score": float(class_scores[idx]),
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "mask": mask_orig,
            })
        profile["hailo_post_project_masks_ms"] = (time.perf_counter() - t0) * 1000.0
        return detections, profile

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.last_infer_job is not None:
                self.last_infer_job.wait(10000)
        except Exception:
            pass
