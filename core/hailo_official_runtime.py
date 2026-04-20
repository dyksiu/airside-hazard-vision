import hashlib
import inspect
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np


class HailoOfficialPipelineError(RuntimeError):
    pass


@dataclass
class HailoModelAssets:
    source_path: str
    hef_path: str
    assets_dir: str
    har_path: Optional[str] = None
    metadata_json: Optional[str] = None
    original_model_meta_json: Optional[str] = None
    modifications_meta_json: Optional[str] = None
    postprocess_onnx: Optional[str] = None

    def iter_config_json_candidates(self) -> Iterable[str]:
        seen = set()
        for item in (
            self.metadata_json,
            self.original_model_meta_json,
            self.modifications_meta_json,
            str(Path(self.hef_path).with_suffix('.json')),
            str(Path(self.hef_path).with_name(f"{Path(self.hef_path).stem}.config.json")),
            str(Path(self.hef_path).parent / 'config.json'),
        ):
            if not item:
                continue
            try:
                p = str(Path(item))
            except Exception:
                continue
            if p in seen:
                continue
            seen.add(p)
            if Path(p).exists() and Path(p).is_file():
                yield p


def _stable_cache_dir(source_path: Path) -> Path:
    key = hashlib.sha1(str(source_path.resolve()).encode('utf-8')).hexdigest()[:12]
    root = Path(tempfile.gettempdir()) / 'hailo_har_cache'
    root.mkdir(parents=True, exist_ok=True)
    out = root / f'{source_path.stem}_{key}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_hailo_model_assets(model_path: str) -> HailoModelAssets:
    src = Path(model_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f'Nie znaleziono pliku modelu HAILO: {src}')

    if src.suffix.lower() == '.har':
        out_dir = _stable_cache_dir(src)
        sentinel = out_dir / '.extracted.ok'
        if not sentinel.exists():
            with tarfile.open(src, 'r') as tar:
                tar.extractall(out_dir)
            sentinel.write_text('ok', encoding='utf-8')

        hef_files = sorted(out_dir.glob('*.hef'))
        if not hef_files:
            raise FileNotFoundError(
                f'Archiwum HAR nie zawiera pliku HEF: {src.name}'
            )
        hef_path = hef_files[0]

        metadata_json = next(iter(sorted(out_dir.glob('*.metadata.json'))), None)
        original_meta = next(iter(sorted(out_dir.glob('*.original_model_meta.json'))), None)
        modifications_meta = next(iter(sorted(out_dir.glob('*.modifications_meta_data.json'))), None)
        postprocess_onnx = next(iter(sorted(out_dir.glob('*.postprocess.onnx'))), None)

        return HailoModelAssets(
            source_path=str(src),
            hef_path=str(hef_path),
            assets_dir=str(out_dir),
            har_path=str(src),
            metadata_json=str(metadata_json) if metadata_json else None,
            original_model_meta_json=str(original_meta) if original_meta else None,
            modifications_meta_json=str(modifications_meta) if modifications_meta else None,
            postprocess_onnx=str(postprocess_onnx) if postprocess_onnx else None,
        )

    if src.suffix.lower() != '.hef':
        raise ValueError('Model HAILO musi być plikiem .hef albo .har.')

    return HailoModelAssets(
        source_path=str(src),
        hef_path=str(src),
        assets_dir=str(src.parent),
    )


class HailoOfficialInstanceSegRunner:
    """
    Adapter dla oficjalnego pipeline'u Hailo.

    Ta wersja umie przyjąć jako wejście zarówno .hef, jak i .har.
    Dla .har automatycznie wypakowuje archiwum do cache, używa zawartego HEF-a
    i próbuje podstawić JSON z metadanymi jako config dla pipeline'u.

    Uwaga praktyczna:
    dla custom instance-seg pipeline OFFICIAL zwykle nadal potrzebuje zgodnego
    postprocess .so. Jeśli go nie znajdzie, zgłasza jasny błąd zamiast milcząco
    wracać do CUSTOM.
    """

    def __init__(self, app, role, model, video_path):
        self.app = app
        self.role = str(role)
        self.model = model
        self.video_path = os.fspath(video_path)
        self.assets = self._resolve_assets_from_model(model)
        self.hef_path = self.assets.hef_path

        self._stop_requested = False
        self._started_at = None
        self._infer_count = 0
        self._latest_avg_fps = 0.0
        self._pipeline = None
        self._bus = None
        self._Gst = None
        self._GLib = None
        self._hailo = None
        self._get_caps_from_pad = None
        self._get_numpy_from_buffer = None
        self._helper = None

        self.config_json = self._resolve_config_json()
        self.postprocess_so = self._resolve_postprocess_so()
        self.post_function_name = os.environ.get('HAILO_SEG_POSTPROCESS_FUNCTION', 'filter')

    def _resolve_assets_from_model(self, model) -> HailoModelAssets:
        assets = getattr(model, 'hailo_assets', None)
        if isinstance(assets, HailoModelAssets):
            return assets
        source_path = getattr(model, 'source_path', None) or getattr(model, 'original_path', None)
        if source_path:
            return resolve_hailo_model_assets(source_path)
        hef_path = getattr(model, 'hef_path', None)
        if hef_path:
            return resolve_hailo_model_assets(hef_path)
        raise HailoOfficialPipelineError('Model HAILO nie zawiera ścieżki do pliku HEF/HAR.')

    def _lazy_imports(self):
        if self._Gst is not None:
            return

        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib

        import hailo
        from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
        from hailo_apps.python.core.gstreamer import gstreamer_helper_pipelines as helper

        Gst.init(None)
        self._Gst = Gst
        self._GLib = GLib
        self._hailo = hailo
        self._get_caps_from_pad = get_caps_from_pad
        self._get_numpy_from_buffer = get_numpy_from_buffer
        self._helper = helper

    def _resolve_config_json(self) -> str:
        env_path = os.environ.get('HAILO_SEG_CONFIG_JSON')
        if env_path and Path(env_path).exists() and Path(env_path).is_file():
            return str(Path(env_path))

        explicit_candidates = list(getattr(self.model, 'config_json_candidates', []) or [])
        if not explicit_candidates:
            explicit_candidates = list(self.assets.iter_config_json_candidates())

        for candidate in explicit_candidates:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)

        extra = ''
        if self.assets.har_path:
            extra = (
                ' Wybrano plik HAR, ale nie znaleziono w nim pliku JSON, który dałoby się '
                "podstawić jako config dla pipeline'u OFFICIAL."
            )
        raise HailoOfficialPipelineError(
            'Tryb OFFICIAL wymaga config JSON dla custom instance-seg. '
            'Nie znaleziono pliku *.json obok HEF, w archiwum HAR ani w HAILO_SEG_CONFIG_JSON.'
            + extra
        )

    def _resolve_postprocess_so(self) -> str:
        env_path = os.environ.get('HAILO_SEG_POSTPROCESS_SO')
        if env_path and Path(env_path).exists() and Path(env_path).is_file():
            return str(Path(env_path))

        hef = Path(self.hef_path)
        candidates: List[Path] = [
            hef.with_suffix('.so'),
            hef.with_name(f'{hef.stem}.so'),
            hef.parent / 'postprocess.so',
            hef.parent / 'libyolo_hailortpp_post.so',
            hef.parent / 'libyolo_post.so',
        ]

        common_dirs = [
            Path('/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes'),
            Path('/usr/local/hailo/tappas/post_processes'),
            Path('/opt/hailo/tappas/post_processes'),
            Path('/usr/lib/hailo/tappas/post_processes'),
        ]
        common_names = [
            'libyolo_hailortpp_post.so',
            'libyolo_post.so',
            'libyolov8seg_post.so',
        ]
        for directory in common_dirs:
            for name in common_names:
                candidates.append(directory / name)

        seen = set()
        for candidate in candidates:
            c = str(candidate)
            if c in seen:
                continue
            seen.add(c)
            if candidate.exists() and candidate.is_file():
                return c

        hint = ''
        if self.assets.postprocess_onnx:
            hint = (
                f' W HAR znaleziono {Path(self.assets.postprocess_onnx).name}, ale helper OFFICIAL '
                'w tej wersji aplikacji oczekuje biblioteki .so.'
            )
        raise HailoOfficialPipelineError(
            'Tryb OFFICIAL wymaga postprocess .so dla custom instance-seg. '
            'Nie znaleziono pliku *.so obok HEF ani w standardowych katalogach. '
            'Ustaw HAILO_SEG_POSTPROCESS_SO na właściwą bibliotekę.' + hint
        )

    def _call_helper(self, func, /, **kwargs):
        try:
            params = inspect.signature(func).parameters
        except Exception:
            params = None
        if params is None:
            return func(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return func(**filtered)

    def build_pipeline_string(self):
        helper = self._helper
        source = self._call_helper(
            helper.SOURCE_PIPELINE,
            video_source=self.video_path,
            video_width=self.app.TARGET_WIDTH,
            video_height=self.app.TARGET_HEIGHT,
            frame_rate=30,
            sync=False,
            video_format='RGB',
            name=f'{self.role}_source',
        )
        infer = self._call_helper(
            helper.INFERENCE_PIPELINE,
            hef_path=self.hef_path,
            post_process_so=self.postprocess_so,
            postprocess_so=self.postprocess_so,
            post_function_name=self.post_function_name,
            batch_size=1,
            config_json=self.config_json,
            config_path=self.config_json,
            name=f'{self.role}_inference',
        )
        wrapped = self._call_helper(
            helper.INFERENCE_PIPELINE_WRAPPER,
            pipeline=infer,
            infer_pipeline=infer,
            name=f'{self.role}_wrapper',
        )
        tracker = self._call_helper(
            helper.TRACKER_PIPELINE,
            class_id=-1,
            keep_lost_frames=self.app.max_track_gap,
            name=f'{self.role}_tracker',
        )
        callback = self._call_helper(helper.USER_CALLBACK_PIPELINE, name=f'{self.role}_callback')
        ui = self._call_helper(helper.UI_APPSINK_PIPELINE, name=f'{self.role}_ui', sync='false')
        return f'{source} ! {wrapped} ! {tracker} ! {callback} ! {ui}'

    def stop(self):
        self._stop_requested = True

    def _queue_frame_for_ui(self, frame_bgr):
        avg_fps = self._latest_avg_fps
        frame_bgr = self.app.draw_fps_overlay(frame_bgr, avg_fps)
        try:
            while True:
                _ = self.app.frame_queue.get_nowait()
        except Exception:
            pass
        self.app.frame_queue.put(frame_bgr)

    def _full_mask_from_detection(self, detection, width, height):
        hailo = self._hailo
        masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
        if not masks:
            return None

        mask = masks[0]
        mask_h = int(mask.get_height())
        mask_w = int(mask.get_width())
        data = np.array(mask.get_data(), dtype=np.float32).reshape((mask_h, mask_w))

        bbox = detection.get_bbox()
        x1 = int(max(0, min(width - 1, round(float(bbox.xmin()) * width))))
        y1 = int(max(0, min(height - 1, round(float(bbox.ymin()) * height))))
        x2 = int(max(x1 + 1, min(width, round(float(bbox.xmax()) * width))))
        y2 = int(max(y1 + 1, min(height, round(float(bbox.ymax()) * height))))
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        resized = cv2.resize(data, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0.5
        full = np.zeros((height, width), dtype=bool)
        full[y1:y2, x1:x2] = resized[: y2 - y1, : x2 - x1]
        return full

    def _handle_detections_from_buffer(self, buffer, width, height):
        hailo = self._hailo
        frame_idx = self._infer_count

        self.app.begin_frame_event_scan()
        analysis_roi, _ = self.app.get_analysis_roi_mask(height, width)
        analysis_roi_bool = analysis_roi.astype(bool) if analysis_roi is not None else None

        if self.role == 'surface':
            surface_roi = np.zeros((height, width), dtype=np.uint8)
        else:
            surface_roi = None

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for det in detections:
            class_name = str(det.get_label())
            conf = float(det.get_confidence())
            bbox = det.get_bbox()

            x1 = int(max(0, min(width - 1, round(float(bbox.xmin()) * width))))
            y1 = int(max(0, min(height - 1, round(float(bbox.ymin()) * height))))
            x2 = int(max(x1 + 1, min(width, round(float(bbox.xmax()) * width))))
            y2 = int(max(y1 + 1, min(height, round(float(bbox.ymax()) * height))))
            bbox_px = (x1, y1, x2, y2)

            mask = self._full_mask_from_detection(det, width, height)
            if mask is not None and analysis_roi_bool is not None:
                mask = mask & analysis_roi_bool
            if mask is not None and not mask.any():
                continue

            if self.role == 'surface':
                area = int(mask.sum()) if mask is not None else max(0, (x2 - x1) * (y2 - y1))
                self.app.observe_surface_candidate(class_name, area)
                if mask is not None:
                    surface_roi[mask] = 1
            else:
                self.app.observe_line_candidate(class_name)

            self.app.register_confirmed_candidate(frame_idx, self.role, class_name, conf, bbox_px)

        if self.role == 'surface':
            self.app.last_surface_roi = surface_roi if surface_roi.any() else None

        self.app.finalize_frame_event_scan()

    def _on_probe(self, pad, info):
        Gst = self._Gst
        if self._stop_requested:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        self._infer_count += 1
        if self._started_at is None:
            self._started_at = time.perf_counter()
        elapsed = max(time.perf_counter() - self._started_at, 1e-6)
        self._latest_avg_fps = self._infer_count / elapsed

        _fmt, width, height = self._get_caps_from_pad(pad)
        width = int(width) if width is not None else int(self.app.TARGET_WIDTH)
        height = int(height) if height is not None else int(self.app.TARGET_HEIGHT)
        self._handle_detections_from_buffer(buffer, width, height)

        with self.app.stats_lock:
            self.app.frame_count = self._infer_count
            self.app.current_fps = self._latest_avg_fps
            self.app.avg_fps = self._latest_avg_fps

        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink):
        Gst = self._Gst
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        fmt = structure.get_value('format')
        width = int(structure.get_value('width'))
        height = int(structure.get_value('height'))

        frame = self._get_numpy_from_buffer(buf, fmt, width, height)
        if frame is None:
            return Gst.FlowReturn.OK

        if fmt == 'RGB':
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = np.asarray(frame).copy()
            if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        self._queue_frame_for_ui(frame_bgr)
        return Gst.FlowReturn.OK

    def run(self):
        self._lazy_imports()
        Gst = self._Gst

        pipeline_string = self.build_pipeline_string()
        self._pipeline = Gst.parse_launch(pipeline_string)

        identity = self._pipeline.get_by_name(f'{self.role}_callback')
        if identity is None:
            raise HailoOfficialPipelineError('Nie znaleziono elementu callback w pipeline Hailo.')
        pad = identity.get_static_pad('src')
        if pad is None:
            raise HailoOfficialPipelineError('Nie udało się pobrać pada src dla callbacku Hailo.')
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_probe)

        appsink = self._pipeline.get_by_name(f'{self.role}_ui')
        if appsink is None:
            raise HailoOfficialPipelineError('Nie znaleziono appsink UI w pipeline Hailo.')
        appsink.connect('new-sample', self._on_new_sample)

        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)

        try:
            while not self._stop_requested and not self.app.stop_event.is_set():
                msg = self._bus.timed_pop_filtered(
                    100 * Gst.MSECOND,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )
                if msg is None:
                    continue
                if msg.type == Gst.MessageType.EOS:
                    break
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    raise HailoOfficialPipelineError(f'Błąd GStreamer/Hailo: {err}; {debug}')
        finally:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
            self._bus = None
