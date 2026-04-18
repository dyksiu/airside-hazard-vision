import cv2
import numpy as np


class RoiMixin:
    def get_analysis_roi_mask(self, height, width):
        pct = int(self.roi_slider.get()) if hasattr(self, "roi_slider") else 0
        if pct <= 0:
            return None, None

        roi_h = max(1, int(round(height * (pct / 100.0))))
        y_start = max(0, height - roi_h)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:, :] = 1
        return mask, y_start

    def apply_analysis_roi_to_frame(self, frame, roi_mask):
        if roi_mask is None:
            return frame
        out = frame.copy()
        out[roi_mask == 0] = 0
        return out

    def draw_roi_boundary(self, frame, y_line):
        if y_line is None:
            return frame

        y_line = int(max(0, min(frame.shape[0] - 1, y_line)))
        color = (220, 220, 220)
        dash_len = 10
        gap_len = 8

        for x in range(0, frame.shape[1], dash_len + gap_len):
            x2 = min(x + dash_len, frame.shape[1] - 1)
            cv2.line(frame, (x, y_line), (x2, y_line), color, 1, cv2.LINE_AA)

        return frame