
from tool.utils import rescale
import numpy as np
from collections import deque
CLASS_NAMES = ["person", "goods"]

class YoloDetector:
    def __init__(self, model, img_size=480, conf_thres=0.25):
        self.model = model
        self.img_size = img_size
        self.conf_thres = conf_thres

    def detect(self, frame):
        preds = self.model.infer(frame, self.img_size)
        results = []

        for d in preds[0]:
            x_min, y_min, x_max, y_max, conf, cls = d[:6]

            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = rescale(frame, self.img_size, x_min, y_min, x_max, y_max)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "class_id": int(cls),
                "class_name": CLASS_NAMES[int(cls)]
            })

        return results

class YoloClassify:
    def __init__(self, model, img_size=224, conf_thres=0.25):
        self.model = model
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.buffer = deque(maxlen=200)  # lưu 10 frame

        # 👇 map class cố định
        self.class_names = ["closed", "open"]

    def predict(self, frame):
        outputs = self.model.infer(frame, self.img_size)
        if outputs is None:
            return None

        out = outputs[0]
        probs = np.array(out).squeeze()

        if probs.ndim == 2:
            probs = probs[0]

        # 👉 Lưu buffer (tối đa 20)
        self.buffer.append(probs)

        # 👉 Luôn tính trung bình (dù có 1 hay 20 frame)
        avg_probs = np.mean(self.buffer, axis=0)
        # print("avg_probs",avg_probs)
        class_id = int(np.argmax(avg_probs))
        conf = float(avg_probs[class_id])
        
        if conf < self.conf_thres:
            return None

        return {
            "class_id": class_id,
            "class_name": ["closed", "open"][class_id],
            "confidence": conf
        }