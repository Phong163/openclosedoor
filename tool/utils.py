import cv2
import numpy as np
import torch
import base64
import logging
import yaml
import os

logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn muốn xem chi tiết
    format="%(asctime)s - %(levelname)s - %(message)s"
)


CONFIG_PATH = os.getenv("CONFIG_PATH", "./config/config.yaml")
_config_cache = None
def get_config():
    global _config_cache

    if _config_cache is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)

    return _config_cache

def check_base64_image(base64_string):

    img_bytes = base64.b64decode(base64_string)

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    cv2.imwrite("./output/images/output_images.jpg", img)
    
def get_zone_coords(frame, zone_relative):
    height, width = frame.shape[:2]
    points_abs = [(int(x * width), int(y * height)) for x, y in zone_relative]
    return np.array([points_abs], dtype=np.int32)

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def is_box_in_zone(box, zone_coords, score):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    # Kiểm tra xem tâm của box có nằm trong vùng không
    if cv2.pointPolygonTest(zone_coords[0], (center_x, center_y), False) >= 0:
        return True
    # Kiểm tra IoU như một phương pháp dự phòng
    box_area = [(x1, y1, x2, y2)]
    zone_points = zone_coords[0]
    zone_x = [point[0] for point in zone_points]
    zone_y = [point[1] for point in zone_points]
    zone_x1, zone_y1 = min(zone_x), min(zone_y)
    zone_x2, zone_y2 = max(zone_x), max(zone_y)
    zone_area = [(zone_x1, zone_y1, zone_x2, zone_y2)]
    box_xyxy = np.array(box_area)
    zone_xyxy = np.array(zone_area)
    iou = box_iou(torch.tensor(box_xyxy), torch.tensor(zone_xyxy)).item()
    return iou > score

def rescale(frame, img_size, x1, y1, x2, y2):

    h, w = frame.shape[:2]

    scale_x = w / img_size
    scale_y = h / img_size

    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    return x1, y1, x2, y2

