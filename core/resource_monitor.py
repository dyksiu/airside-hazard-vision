import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


class ResourceMonitor:
    def __init__(self, sample_interval_s: float = 0.5):
        self.sample_interval_s = max(0.1, float(sample_interval_s))
        self._last_sample_ts = 0.0
        self._cpu_prev = self._read_proc_stat()
        self._cpu_current = 0.0
        self._cpu_sum = 0.0
        self._cpu_count = 0
        self._accel_current = None
        self._accel_sum = 0.0
        self._accel_count = 0
        self._accel_name = ""
        self._accel_mode = ""
        self._last_hailo_file = None
        self._last_hailo_mtime = 0.0

    def reset(self):
        self._last_sample_ts = 0.0
        self._cpu_prev = self._read_proc_stat()
        self._cpu_current = 0.0
        self._cpu_sum = 0.0
        self._cpu_count = 0
        self._accel_current = None
        self._accel_sum = 0.0
        self._accel_count = 0
        self._accel_name = ""
        self._accel_mode = ""
        self._last_hailo_file = None
        self._last_hailo_mtime = 0.0

    def snapshot(self) -> dict[str, Any]:
        cpu_avg = (self._cpu_sum / self._cpu_count) if self._cpu_count else 0.0
        accel_avg = (self._accel_sum / self._accel_count) if self._accel_count else None
        return {
            "cpu_usage_percent": float(self._cpu_current or 0.0),
            "cpu_usage_avg_percent": float(cpu_avg or 0.0),
            "accelerator_name": self._accel_name,
            "accelerator_mode": self._accel_mode,
            "accelerator_usage_percent": accel_avg if self._accel_current is None else float(self._accel_current),
            "accelerator_usage_avg_percent": None if accel_avg is None else float(accel_avg),
            "accelerator_available": bool(self._accel_name),
        }

    def update(self, prefer_hailo: bool = False) -> dict[str, Any]:
        now = time.time()
        if (now - self._last_sample_ts) < self.sample_interval_s:
            return self.snapshot()

        self._last_sample_ts = now
        cpu = self._read_cpu_percent()
        if cpu is not None:
            self._cpu_current = float(max(0.0, min(100.0, cpu)))
            self._cpu_sum += self._cpu_current
            self._cpu_count += 1

        accel_name = ""
        accel_mode = ""
        accel_value = None

        if prefer_hailo:
            os.environ.setdefault("HAILO_MONITOR", "1")
            accel_value = self._read_hailo_usage_percent()
            if accel_value is not None:
                accel_name = "Hailo"
                accel_mode = "hailo"
            else:
                accel_name = "Hailo"
                accel_mode = "hailo"
        else:
            accel_value = self._read_nvidia_gpu_percent()
            if accel_value is not None:
                accel_name = "GPU"
                accel_mode = "gpu"
            else:
                accel_value = self._read_hailo_usage_percent()
                if accel_value is not None:
                    accel_name = "Hailo"
                    accel_mode = "hailo"

        self._accel_name = accel_name
        self._accel_mode = accel_mode
        self._accel_current = None if accel_value is None else float(max(0.0, min(100.0, accel_value)))
        if self._accel_current is not None:
            self._accel_sum += self._accel_current
            self._accel_count += 1

        return self.snapshot()

    def _read_cpu_percent(self):
        current = self._read_proc_stat()
        prev = self._cpu_prev
        self._cpu_prev = current
        if prev is None or current is None:
            return None

        prev_idle, prev_total = prev
        idle, total = current
        total_delta = total - prev_total
        idle_delta = idle - prev_idle
        if total_delta <= 0:
            return None
        busy = 1.0 - (idle_delta / total_delta)
        return busy * 100.0

    @staticmethod
    def _read_proc_stat():
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip()
            if not first.startswith("cpu "):
                return None
            parts = [int(x) for x in first.split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            total = sum(parts)
            return idle, total
        except Exception:
            return None

    @staticmethod
    def _read_nvidia_gpu_percent():
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        try:
            out = subprocess.check_output(
                [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                text=True,
            ).strip()
        except Exception:
            return None
        if not out:
            return None
        try:
            first = out.splitlines()[0].strip()
            return float(first)
        except Exception:
            return None

    def _read_hailo_usage_percent(self):
        candidates = []
        root = Path("/tmp/hmon_files")
        if root.exists():
            try:
                candidates.extend([p for p in root.rglob("*") if p.is_file()])
            except Exception:
                pass

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        for path in candidates[:8]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            value = self._extract_usage_from_text(text)
            if value is not None:
                try:
                    self._last_hailo_file = str(path)
                    self._last_hailo_mtime = path.stat().st_mtime
                except Exception:
                    pass
                return value
        return None

    def _extract_usage_from_text(self, text: str):
        if not text:
            return None

        parsed_json = None
        try:
            parsed_json = json.loads(text)
        except Exception:
            parsed_json = None
        if parsed_json is not None:
            value = self._find_usage_in_obj(parsed_json)
            if value is not None:
                return value

        for line in text.splitlines():
            lower = line.lower()
            if not any(key in lower for key in ("util", "usage", "percent", "%", "hailo", "npu")):
                continue

            keyword_patterns = [
                r"(?:device\s+)?util(?:ization)?[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
                r"(?:hailo|npu)[^\n:]{0,30}(?:util(?:ization)?|usage)[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
                r"(?:util(?:ization)?|usage)[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
            ]
            for pattern in keyword_patterns:
                match = re.search(pattern, line, flags=re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except Exception:
                        pass

            generic_match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if generic_match and any(key in lower for key in ("util", "usage", "hailo", "npu")):
                try:
                    return float(generic_match.group(1))
                except Exception:
                    pass

        return None

    def _find_usage_in_obj(self, obj: Any):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if any(token in key_l for token in ("util", "usage", "percent")):
                    numeric = self._coerce_percent(value)
                    if numeric is not None:
                        return numeric
                nested = self._find_usage_in_obj(value)
                if nested is not None:
                    return nested
            return None

        if isinstance(obj, list):
            for item in obj:
                nested = self._find_usage_in_obj(item)
                if nested is not None:
                    return nested
            return None

        return None

    @staticmethod
    def _coerce_percent(value: Any):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            num = float(value)
            if 0.0 <= num <= 100.0:
                return num
            return None
        if isinstance(value, str):
            match = re.search(r"(\d+(?:\.\d+)?)\s*%?", value)
            if match:
                try:
                    num = float(match.group(1))
                except Exception:
                    return None
                if 0.0 <= num <= 100.0:
                    return num
        return None
