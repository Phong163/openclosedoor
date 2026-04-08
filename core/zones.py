import cv2
import numpy as np
from tool.utils import get_zone_coords


class Zone:
    def __init__(self, zone_points, frame_shape):
        """
        zone_points: [(x, y)] normalized (0-1)
        frame_shape: (H, W, C)
        """
        self.zone_points = zone_points
        self.height, self.width = frame_shape[:2]

        # dùng utils
        coords = get_zone_coords(
            np.zeros((self.height, self.width, 3), dtype=np.uint8),
            zone_points
        )

        # get_zone_coords trả về dạng [[[x,y],...]]
        self.poly = np.array(coords[0], dtype=np.int32)

        self.area = cv2.contourArea(self.poly)

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.poly], 255)

    def draw(self, frame, color=(0, 0, 255), thickness=3):
        coords = get_zone_coords(frame, self.zone_points)
        cv2.polylines(frame, coords, True, color, thickness)

    def intersection_area(self, bbox):
        x1, y1, x2, y2 = bbox

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        inter = cv2.bitwise_and(mask, self.mask)
        return cv2.countNonZero(inter)