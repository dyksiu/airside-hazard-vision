import importlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _bbox_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int64)


def _letterbox(image: np.ndarray, new_shape: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)):
    orig_h, orig_w = image.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    ratio = min(new_w / orig_w, new_h / orig_h)
    resized_w = int(round(orig_w * ratio))
    resized_h = int(round(orig_h * ratio))
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left = int(round(pad_w / 2 - 0.1))
    pad_right = int(round(pad_w / 2 + 0.1))
    pad_top = int(round(pad_h / 2 - 0.1))
    pad_bottom = int(round(pad_h / 2 + 0.1))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR) if (orig_w, orig_h) != (resized_w, resized_h) else image
    out = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return out, ratio, (pad_left, pad_top)


def _scale_boxes_from_letterbox(boxes_xyxy: np.ndarray, orig_shape: Tuple[int, int], ratio: float, pad: Tuple[int, int]) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32, copy=False)
    pad_x, pad_y = pad
    out = boxes_xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] -= float(pad_x)
    out[:, [1, 3]] -= float(pad_y)
    out /= max(float(ratio), 1e-6)
    h, w = orig_shape
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, max(0, w - 1))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, max(0, h - 1))
    return out


def _scale_mask_from_letterbox(mask: np.ndarray, orig_shape: Tuple[int, int], ratio: float, pad: Tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = orig_shape
    pad_x, pad_y = pad
    in_h, in_w = mask.shape[:2]
    x1 = int(np.clip(pad_x, 0, in_w))
    y1 = int(np.clip(pad_y, 0, in_h))
    x2 = int(np.clip(in_w - pad_x, x1 + 1, in_w))
    y2 = int(np.clip(in_h - pad_y, y1 + 1, in_h))
    cropped = mask[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((orig_h, orig_w), dtype=np.float32)
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _process_mask(protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mh, mw, c = protos.shape
    ih, iw = shape
    if masks_in.size == 0:
        return np.zeros((0, ih, iw), dtype=np.float32)
    protos_flat = protos.reshape(-1, c).T
    masks = _sigmoid(masks_in @ protos_flat).reshape(-1, mh, mw)
    resized = np.empty((masks.shape[0], ih, iw), dtype=np.float32)
    for i in range(masks.shape[0]):
        resized[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
    cropped = np.zeros_like(resized)
    boxes = np.round(bboxes).astype(int)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, iw - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ih - 1)
    for i in range(resized.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        if x2 <= x1 or y2 <= y1:
            continue
        cropped[i, y1:y2, x1:x2] = resized[i, y1:y2, x1:x2]
    return cropped


class HailoInstanceSegmentationModel:
    def __init__(self, hef_path: str, class_names: Dict[int, str], score_threshold: float = 0.01, nms_iou_thresh: float = 0.7, regression_length: int = 15, mask_channels: int = 32):
        try:
            hp = importlib.import_module("hailo_platform")
        except Exception as exc:
            raise ImportError(
                "Brakuje biblioteki hailo_platform. Zainstaluj HailoRT / pyHailoRT na Raspberry Pi."
            ) from exc
        self._hp = hp
        self.HEF = hp.HEF
        self.VDevice = hp.VDevice
        self.FormatType = getattr(hp, "FormatType", None)
        self.HailoSchedulingAlgorithm = getattr(hp, "HailoSchedulingAlgorithm", None)
        self.hef_path = hef_path
        self.names = dict(class_names)
        self.score_threshold = float(score_threshold)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.regression_length = int(regression_length)
        self.mask_channels = int(mask_channels)
        self._lock = threading.Lock()
        self.hef = self.HEF(hef_path)
        params = self.VDevice.create_params()
        if self.HailoSchedulingAlgorithm is not None:
            try:
                params.scheduling_algorithm = self.HailoSchedulingAlgorithm.ROUND_ROBIN
            except Exception:
                pass
        self.target = self.VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        self.input_infos = list(self.hef.get_input_vstream_infos())
        self.output_infos = list(self.hef.get_output_vstream_infos())
        self.input_name = self.input_infos[0].name
        self.output_names = [info.name for info in self.output_infos]
        if self.FormatType is not None:
            try:
                self.infer_model.input(self.input_name).set_format_type(self.FormatType.UINT8)
            except Exception:
                try:
                    self.infer_model.input().set_format_type(self.FormatType.UINT8)
                except Exception:
                    pass
            for output_name in self.output_names:
                try:
                    self.infer_model.output(output_name).set_format_type(self.FormatType.FLOAT32)
                except Exception:
                    pass
        self._configured_infer_model = self.infer_model.configure()
        self._input_hw = self._resolve_input_hw()
        self._debug_path = Path(hef_path).with_suffix('.hailo_debug.txt')
        self._predict_debug_count = 0
        self._write_debug_header()

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_hw

    def _resolve_input_hw(self) -> Tuple[int, int]:
        info = self.input_infos[0]
        shape = getattr(info, 'shape', None)
        for cand in (shape, info):
            if cand is None:
                continue
            if hasattr(cand, 'height') and hasattr(cand, 'width'):
                return int(cand.height), int(cand.width)
            if hasattr(cand, 'h') and hasattr(cand, 'w'):
                return int(cand.h), int(cand.w)
            if isinstance(cand, (tuple, list)) and len(cand) >= 2:
                return int(cand[0]), int(cand[1])
        return 640, 640

    def _write_debug_header(self):
        lines = [f'HEF: {self.hef_path}', f'Input: {self.input_name} size={self.input_size}', 'Outputs:']
        for info in self.output_infos:
            shape = getattr(info, 'shape', None)
            fmt = getattr(getattr(info, 'format', None), 'type', None)
            order = getattr(getattr(info, 'format', None), 'order', None)
            lines.append(f'  - {getattr(info, "name", "?")} shape={shape} format={fmt} order={order}')
        self._debug_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    def _append_debug(self, text: str):
        try:
            with self._debug_path.open('a', encoding='utf-8') as f:
                f.write(text + "\n")
        except Exception:
            pass

    def close(self):
        for name in ('_configured_infer_model', 'infer_model', 'target', 'hef'):
            try:
                delattr(self, name)
            except Exception:
                pass

    def _prepare_input(self, frame_bgr: np.ndarray):
        in_h, in_w = self.input_size
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        letterboxed, ratio, pad = _letterbox(rgb, (in_h, in_w))
        tensor = np.ascontiguousarray(letterboxed[np.newaxis, ...], dtype=np.uint8)
        return tensor, ratio, pad

    def _output_shape(self, output_obj):
        shape = getattr(output_obj, 'shape', None)
        if shape is None:
            infer_shape = getattr(output_obj, 'get_shape', None)
            if callable(infer_shape):
                shape = infer_shape()
        if shape is None:
            raise RuntimeError('Nie udało się odczytać kształtu wyjścia Hailo.')
        return tuple(int(x) for x in shape)

    def _make_output_buffers(self):
        bufs = {}
        for name in self.output_names:
            shape = self._output_shape(self.infer_model.output(name))
            bufs[name] = np.empty(shape, dtype=np.float32)
        return bufs

    def _run_inference(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        with self._lock:
            output_buffers = self._make_output_buffers()
            bindings = self._configured_infer_model.create_bindings(output_buffers=output_buffers)
            try:
                bindings.input(self.input_name).set_buffer(input_tensor)
            except Exception:
                bindings.input().set_buffer(input_tensor)
            try:
                self._configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            except Exception:
                pass
            job = self._configured_infer_model.run_async([bindings])
            job.wait(10000)
            result = {}
            for name in self.output_names:
                try:
                    result[name] = bindings.output(name).get_buffer()
                except Exception:
                    result[name] = output_buffers[name]
            return result

    def _normalize_tensor_layout(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            return arr.astype(np.float32, copy=False)
        possible_channels = {4 * (self.regression_length + 1), len(self.names), self.mask_channels}
        if arr.shape[-1] in possible_channels:
            return arr.astype(np.float32, copy=False)
        if arr.shape[1] in possible_channels:
            return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _classify_and_order_outputs(self, outputs: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        boxes, scores, coeffs, proto = {}, {}, {}, None
        num_classes = len(self.names)
        for name, raw in outputs.items():
            arr = self._normalize_tensor_layout(raw)
            if arr.ndim != 4:
                continue
            _, h, w, c = arr.shape
            if c == 4 * (self.regression_length + 1) and h in (20, 40, 80):
                boxes[h] = arr
            elif c == num_classes and h in (20, 40, 80):
                scores[h] = arr
            elif c == self.mask_channels and h in (20, 40, 80):
                coeffs[h] = arr
            elif c == self.mask_channels and max(h, w) >= 160:
                proto = arr
        if self._predict_debug_count < 10:
            for name, raw in outputs.items():
                arr = np.asarray(raw)
                self._append_debug(f'RAW {name}: shape={arr.shape} min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f}')
        if not boxes or not scores or not coeffs or proto is None:
            return None
        ordered = []
        for h in (20, 40, 80):
            if h not in boxes or h not in scores or h not in coeffs:
                return None
            ordered.extend([boxes[h], scores[h], coeffs[h]])
        ordered.append(proto)
        return ordered

    def _decode(self, ordered_outputs: List[np.ndarray], orig_shape: Tuple[int, int], ratio: float, pad: Tuple[int, int]) -> List[Dict[str, object]]:
        classes = len(self.names)
        box_tensors = ordered_outputs[:7:3]
        score_tensors = ordered_outputs[1:8:3]
        coeff_tensors = ordered_outputs[2:9:3]
        proto = ordered_outputs[9][0]
        strides = [32, 16, 8]
        reg_max = self.regression_length
        reg_bins = np.arange(reg_max + 1, dtype=np.float32)

        all_boxes = []
        all_scores = []
        all_coeffs = []
        for box_tensor, score_tensor, coeff_tensor, stride in zip(box_tensors, score_tensors, coeff_tensors, strides):
            box_tensor = box_tensor[0]
            score_tensor = score_tensor[0]
            coeff_tensor = coeff_tensor[0]
            h, w, _ = box_tensor.shape

            dist = box_tensor.reshape(h * w, 4, reg_max + 1)
            dist = _softmax(dist, axis=-1)
            dist = (dist * reg_bins).sum(axis=-1) * float(stride)

            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
            centers = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1) * float(stride)

            x1 = centers[:, 0] - dist[:, 0]
            y1 = centers[:, 1] - dist[:, 1]
            x2 = centers[:, 0] + dist[:, 2]
            y2 = centers[:, 1] + dist[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            scores = _sigmoid(score_tensor.reshape(h * w, classes))
            coeffs = coeff_tensor.reshape(h * w, self.mask_channels)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_coeffs.append(coeffs)

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        coeffs = np.concatenate(all_coeffs, axis=0).astype(np.float32)

        if self._predict_debug_count < 10:
            self._append_debug(f'DECODE: total_candidates={len(boxes)} score_max={float(scores.max()):.4f} score_mean={float(scores.mean()):.4f}')

        detections = []
        for cls_id in range(classes):
            cls_scores = scores[:, cls_id]
            keep = cls_scores >= self.score_threshold
            if not np.any(keep):
                continue
            cls_boxes = boxes[keep]
            cls_scores_kept = cls_scores[keep]
            cls_coeffs = coeffs[keep]
            keep_idx = _nms_per_class(cls_boxes, cls_scores_kept, self.nms_iou_thresh, 100)
            cls_boxes = cls_boxes[keep_idx]
            cls_scores_kept = cls_scores_kept[keep_idx]
            cls_coeffs = cls_coeffs[keep_idx]
            masks = _process_mask(proto, cls_coeffs, cls_boxes, self.input_size)
            cls_boxes = _scale_boxes_from_letterbox(cls_boxes, orig_shape, ratio, pad)
            for i in range(len(cls_scores_kept)):
                x1, y1, x2, y2 = cls_boxes[i]
                mask = _scale_mask_from_letterbox(masks[i], orig_shape, ratio, pad)
                mask = (mask > 0.45).astype(np.uint8) * 255
                detections.append({
                    'cls_id': int(cls_id),
                    'class_name': self.names.get(int(cls_id), str(cls_id)),
                    'conf': float(cls_scores_kept[i]),
                    'bbox': (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    'mask': mask,
                })
        detections.sort(key=lambda d: d['conf'], reverse=True)
        if self._predict_debug_count < 10:
            self._append_debug(f'DECODE: detections_after_nms={len(detections)} top_conf={(detections[0]["conf"] if detections else 0.0):.4f}')
        self._predict_debug_count += 1
        return detections

    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, object]]:
        orig_h, orig_w = frame_bgr.shape[:2]
        input_tensor, ratio, pad = self._prepare_input(frame_bgr)
        outputs = self._run_inference(input_tensor)
        ordered_outputs = self._classify_and_order_outputs(outputs)
        if ordered_outputs is None:
            self._append_debug('ERROR: could not classify output tensors for YOLOv8-seg')
            return []
        return self._decode(ordered_outputs, (orig_h, orig_w), ratio, pad)
