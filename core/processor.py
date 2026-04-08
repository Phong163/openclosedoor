import cv2
import numpy as np


def is_point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly.astype(np.int32), point, False) >= 0


class PersonCounter:
    def __init__(self, zone_poly1, zone_poly2, max_person, smooth_window=10):
        self.zone_poly1 = zone_poly1
        self.zone_poly2 = zone_poly2
        self.max_person = max_person
        
        self.count_history = []
        self.smooth_window = smooth_window

    def compute(self, detections):
        count = 0

        for d in detections:
            if d["class_name"] != "person":
                continue

            x1, y1, x2, y2 = d["bbox"]

            # center bbox
            cx = int((x2 + x1) / 2)

            in_zone1 = is_point_in_poly((cx, y2), self.zone_poly1)
            in_zone2 = is_point_in_poly((cx, y2), self.zone_poly2)

            # ✅ Logic mới
            if in_zone1 and not in_zone2:
                count += 1

        # smoothing
        self.count_history.append(count)
        if len(self.count_history) > self.smooth_window:
            self.count_history.pop(0)

        smooth_count = int(sum(self.count_history) / len(self.count_history))

        return smooth_count