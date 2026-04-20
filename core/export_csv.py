import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


RUNTIME_METRIC_LABELS = {
    "frame_count": "Liczba klatek",
    "input_fps": "FPS wejścia",
    "inference_fps": "FPS inferencji",
    "detection_fps": "FPS detekcji",
    "display_fps": "FPS podglądu",
    "avg_infer_ms": "Śr. czas inferencji [ms]",
    "current_fps": "Aktualne FPS",
    "avg_fps": "Średnie FPS",
    "gui_render_count": "Liczba renderów GUI",
    "cpu_usage_percent": "CPU użycie [%]",
    "cpu_usage_avg_percent": "CPU średnie użycie [%]",
    "accelerator_name": "Akcelerator",
    "accelerator_mode": "Tryb akceleratora",
    "accelerator_usage_percent": "Akcelerator użycie [%]",
    "accelerator_usage_avg_percent": "Akcelerator średnie użycie [%]",
}

STAGE_PROFILE_LABELS = {
    "hailo_preprocess_ms": "Hailo pre [ms]",
    "hailo_infer_ms": "Hailo infer [ms]",
    "hailo_postprocess_ms": "Hailo post [ms]",
    "hailo_post_groups_ms": "Post groups [ms]",
    "hailo_post_decode_ms": "Post decode [ms]",
    "hailo_post_filter_ms": "Post filter [ms]",
    "hailo_post_nms_ms": "Post NMS [ms]",
    "hailo_mask_decode_ms": "Mask decode [ms]",
    "hailo_post_scale_boxes_ms": "Scale boxes [ms]",
    "hailo_post_project_masks_ms": "Project masks [ms]",
    "hailo_contours_ms": "Contours [ms]",
    "hailo_contour_draw_ms": "Contour draw [ms]",
    "hailo_label_draw_ms": "Label draw [ms]",
    "gui_render_ms": "GUI render [ms]",
    "hailo_post_candidates": "Liczba kandydatów postprocessingu",
}

PREFERRED_STAGE_ORDER = [
    "hailo_preprocess_ms",
    "hailo_infer_ms",
    "hailo_postprocess_ms",
    "hailo_post_groups_ms",
    "hailo_post_decode_ms",
    "hailo_post_filter_ms",
    "hailo_post_nms_ms",
    "hailo_mask_decode_ms",
    "hailo_post_scale_boxes_ms",
    "hailo_post_project_masks_ms",
    "hailo_contours_ms",
    "hailo_contour_draw_ms",
    "hailo_label_draw_ms",
    "gui_render_ms",
    "hailo_post_candidates",
]

SUMMARY_INFO_ORDER = [
    "timestamp",
    "benchmark_mode",
    "benchmark_display_stride",
    "surface_backend",
    "line_backend",
    "surface_model_name",
    "line_model_name",
    "process_stride",
    "confidence_threshold",
    "video_source",
    "runtime_overlay_enabled",
    "usage_overlay_enabled",
]


def _round_if_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def _ordered_keys(keys: Iterable[str], preferred_order: Iterable[str]) -> list[str]:
    keys = list(dict.fromkeys(keys))
    preferred = [key for key in preferred_order if key in keys]
    rest = sorted(key for key in keys if key not in set(preferred))
    return preferred + rest


def _metric_csv_columns(
    run_info: Mapping[str, Any],
    runtime_stats: Mapping[str, Any],
    stage_profile_avg: Mapping[str, Any],
) -> list[str]:
    info_columns = [key for key in SUMMARY_INFO_ORDER if key in run_info]
    extra_info = sorted(key for key in run_info.keys() if key not in info_columns)

    runtime_columns = [
        key for key in [
            "frame_count",
            "input_fps",
            "inference_fps",
            "detection_fps",
            "display_fps",
            "avg_infer_ms",
            "current_fps",
            "avg_fps",
            "gui_render_count",
            "cpu_usage_percent",
            "cpu_usage_avg_percent",
            "accelerator_name",
            "accelerator_mode",
            "accelerator_usage_percent",
            "accelerator_usage_avg_percent",
        ] if key in runtime_stats
    ]
    extra_runtime = sorted(key for key in runtime_stats.keys() if key not in runtime_columns)

    stage_columns = _ordered_keys(stage_profile_avg.keys(), PREFERRED_STAGE_ORDER)
    stage_columns = [f"avg::{key}" for key in stage_columns]

    return info_columns + extra_info + runtime_columns + extra_runtime + stage_columns


def _metric_csv_values(
    columns: Iterable[str],
    run_info: Mapping[str, Any],
    runtime_stats: Mapping[str, Any],
    stage_profile_avg: Mapping[str, Any],
) -> list[Any]:
    out = []
    for column in columns:
        if column.startswith("avg::"):
            key = column.split("::", 1)[1]
            out.append(_round_if_number(stage_profile_avg.get(key, "")))
        elif column in run_info:
            out.append(_round_if_number(run_info.get(column, "")))
        else:
            out.append(_round_if_number(runtime_stats.get(column, "")))
    return out


def _write_metric_summary_csv(
    save_path: str,
    run_info: Mapping[str, Any],
    runtime_stats: Mapping[str, Any],
    stage_profile_avg: Mapping[str, Any],
) -> str:
    base = Path(save_path)
    metrics_path = str(base.with_name(f"{base.stem}_metrics.csv"))

    columns = _metric_csv_columns(run_info, runtime_stats, stage_profile_avg)
    values = _metric_csv_values(columns, run_info, runtime_stats, stage_profile_avg)

    with open(metrics_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)
        writer.writerow(values)

    return metrics_path


def _write_key_value_section(writer: csv.writer, title: str, data: Mapping[str, Any], label_map: Optional[Mapping[str, str]] = None, preferred_order: Optional[Iterable[str]] = None):
    if not data:
        return
    writer.writerow([])
    writer.writerow([title])
    writer.writerow(["Pole", "Wartość"])

    ordered = _ordered_keys(data.keys(), preferred_order or [])
    for key in ordered:
        label = (label_map or {}).get(key, key)
        writer.writerow([label, _round_if_number(data.get(key, ""))])


def save_results_csv(
    save_path,
    surface_backend,
    line_backend,
    surface_model,
    line_model,
    unet_num_classes,
    unet_names,
    line_unet_num_classes,
    line_unet_names,
    confidences_copy,
    detections_copy,
    current_fps=0.0,
    avg_fps=0.0,
    event_log_copy=None,
    runtime_stats=None,
    stage_profile_avg=None,
    stage_profile_last=None,
    run_info=None,
):
    runtime_stats = dict(runtime_stats or {})
    stage_profile_avg = dict(stage_profile_avg or {})
    stage_profile_last = dict(stage_profile_last or {})
    run_info = dict(run_info or {})

    run_info.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    runtime_stats.setdefault("current_fps", float(current_fps))
    runtime_stats.setdefault("avg_fps", float(avg_fps))

    metrics_csv_path = _write_metric_summary_csv(
        save_path=save_path,
        run_info=run_info,
        runtime_stats=runtime_stats,
        stage_profile_avg=stage_profile_avg,
    )

    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(["# LEGENDA KLAS (format: typ - ID = Nazwa)"])

        if surface_backend in {"YOLO", "HAILO"}:
            if surface_model is not None and hasattr(surface_model, "names"):
                for cls_id, name in surface_model.names.items():
                    writer.writerow([f"# surface: {cls_id} = {name}"])
        else:
            writer.writerow(["# surface: 0 = tlo"])
            for cls_id in range(1, unet_num_classes):
                writer.writerow([f"# surface: {cls_id} = {unet_names.get(cls_id, str(cls_id))}"])

        if line_backend in {"YOLO", "HAILO"}:
            if line_model is not None and hasattr(line_model, "names"):
                for cls_id, name in line_model.names.items():
                    writer.writerow([f"# line: {cls_id} = {name}"])
        else:
            writer.writerow(["# line: 0 = tlo"])
            for cls_id in range(1, line_unet_num_classes):
                writer.writerow([f"# line: {cls_id} = {line_unet_names.get(cls_id, str(cls_id))}"])

        writer.writerow([])
        writer.writerow(["PODSUMOWANIE_POMIARU_1_WIERSZ"])
        summary_columns = _metric_csv_columns(run_info, runtime_stats, stage_profile_avg)
        writer.writerow(summary_columns)
        writer.writerow(_metric_csv_values(summary_columns, run_info, runtime_stats, stage_profile_avg))

        _write_key_value_section(writer, "INFORMACJE O POMIARZE", run_info, preferred_order=SUMMARY_INFO_ORDER)
        _write_key_value_section(writer, "STATYSTYKI RUNTIME", runtime_stats, label_map=RUNTIME_METRIC_LABELS)
        _write_key_value_section(writer, "ŚREDNIE CZASY ETAPÓW", stage_profile_avg, label_map=STAGE_PROFILE_LABELS, preferred_order=PREFERRED_STAGE_ORDER)
        _write_key_value_section(writer, "OSTATNIE CZASY ETAPÓW", stage_profile_last, label_map=STAGE_PROFILE_LABELS, preferred_order=PREFERRED_STAGE_ORDER)

        writer.writerow([])
        writer.writerow(["PODSUMOWANIE DETEKCJI"])
        writer.writerow(["Typ modelu", "Klasa", "Średni confidence", "Liczba wykryć"])

        for key, confs in confidences_copy.items():
            model_type, class_name = key.split(":", 1)
            avg_conf = sum(confs) / len(confs)
            writer.writerow([model_type, class_name, round(avg_conf, 4), len(confs)])

        writer.writerow([])
        writer.writerow(["WSZYSTKIE WYKRYCIA"])
        writer.writerow(["Numer klatki", "Typ modelu", "Klasa", "Confidence"])
        for detection in detections_copy:
            writer.writerow(detection)

        if event_log_copy:
            writer.writerow([])
            writer.writerow(["LOG ZDARZEŃ"])
            writer.writerow(["Czas", "Typ zdarzenia", "Opis"])
            for entry in event_log_copy:
                writer.writerow(list(entry))

        writer.writerow([])
        writer.writerow(["PLIK_ZBIORCZY_METRYK"])
        writer.writerow([metrics_csv_path])

    return {
        "details_csv_path": str(save_path),
        "metrics_csv_path": metrics_csv_path,
    }
