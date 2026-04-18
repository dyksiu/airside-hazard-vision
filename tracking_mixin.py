class TrackingMixin:
    def copy_stats(self):
        with self.stats_lock:
            stats = {
                k: {"count": v["count"], "conf_sum": v["conf_sum"]}
                for k, v in self.class_stats.items()
            }
            fc = self.frame_count
        return stats, fc

    def bump_stat(self, key, conf):
        s = self.class_stats.setdefault(key, {"count": 0, "conf_sum": 0.0})
        s["count"] += 1
        s["conf_sum"] += float(conf)

    def reset_tracking(self):
        self.active_tracks = {}
        self.next_track_id = 1

    def bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def cleanup_tracks(self, frame_idx):
        to_delete = []
        for tid, tr in self.active_tracks.items():
            if frame_idx - tr["last_frame"] > self.max_track_gap:
                to_delete.append(tid)
        for tid in to_delete:
            del self.active_tracks[tid]

    def register_confirmed_candidate(self, frame_idx, model_type, class_name, conf, bbox):
        key = f"{model_type}:{class_name}"
        self.cleanup_tracks(frame_idx)

        cx, cy = self.bbox_center(bbox)
        best_tid = None
        best_score = -1e9

        for tid, tr in self.active_tracks.items():
            if tr["key"] != key:
                continue

            prev_cx, prev_cy = self.bbox_center(tr["bbox"])
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            iou = self.bbox_iou(bbox, tr["bbox"])

            if dist <= self.match_distance_px or iou >= 0.2:
                score = iou - (dist / max(1.0, self.match_distance_px))
                if score > best_score:
                    best_score = score
                    best_tid = tid

        if best_tid is None:
            self.active_tracks[self.next_track_id] = {
                "key": key,
                "bbox": bbox,
                "hits": 1,
                "last_frame": frame_idx,
                "counted": False,
                "conf_sum": float(conf),
                "obs": 1,
            }
            self.next_track_id += 1
            return False

        tr = self.active_tracks[best_tid]
        if frame_idx != tr["last_frame"]:
            tr["hits"] += 1
            tr["obs"] += 1
            tr["conf_sum"] += float(conf)

        tr["bbox"] = bbox
        tr["last_frame"] = frame_idx

        if (not tr["counted"]) and tr["hits"] >= self.min_confirm_frames:
            tr["counted"] = True
            avg_conf = tr["conf_sum"] / max(1, tr["obs"])
            with self.stats_lock:
                self.class_confidences.setdefault(key, []).append(avg_conf)
                self.detections.append((frame_idx, model_type, class_name, round(avg_conf, 4)))
                self.bump_stat(key, avg_conf)
            return True

        return False