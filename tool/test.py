from datetime import datetime
import json
import cv2
import pytz
import base64
import time
import os
from confluent_kafka import Producer
import logging

# ================= LOG =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= KAFKA =================
_producer = None

_producer_config = {
    'bootstrap.servers': 'gateway-1.hn.eidox.ai:9093',
    'client.id': 'python-producer',
    'security.protocol': 'PLAINTEXT',
    # 'security.protocol': 'SSL',
    # 'ssl.ca.location': './cert/ca-root.pem',
    # 'ssl.certificate.location': './cert/ca-cert.pem',
    # 'ssl.key.location': './cert/ca-key.pem',
    'retries': 5,
    'retry.backoff.ms': 1000
}


def get_producer():
    global _producer
    if _producer is None:
        _producer = Producer(_producer_config)
        logging.info("Kafka Producer initialized")
    return _producer


def send_data_to_kafka(box_id, camera_id, percent, image_base64, topic="gateway-box-pchm"):
    timestamp = int(datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).timestamp())

    payload = {
            "box_id": "31f52bb2-686f-48ae-a472-2c2aa1329d80",
            "cam_id": "bc4e9e35-cf47-4f1e-a83e-b4b2e5e2639d",
            "metric":"opencloseddoor",
            "door_state": "open",
            "conf": 50,
            "emb": image_base64,
            "timestamp": timestamp
            }

    def delivery_report(err, msg):
        if err:
            logging.error(f"Kafka send failed: {err}")
        else:
            logging.info(
                f"Kafka sent | box_id={box_id} cam_id={camera_id} percent={percent:.2f}"
            )

    try:
        producer = get_producer()
        message = json.dumps(payload).encode("utf-8")

        producer.produce(
            topic=topic,
            value=message,
            callback=delivery_report
        )

        producer.poll(0)

    except Exception as e:
        logging.error(f"Kafka error: {e}")


def close_producer():
    global _producer
    if _producer:
        _producer.flush()
        logging.info("Kafka Producer closed")
        _producer = None


# ================= IMAGE =================
def encode_frame_base64(frame):
    frame = cv2.resize(frame, (720, 640))
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode("utf-8")


# ================= MAIN CLASS =================
class FireDetector:

    def __init__(self, box_id, camera_id):
        self.box_id = box_id
        self.camera_id = camera_id
        self.last_sent_time = 0  # để limit 5 phút

    def send_fire_alert(self, conf, frame):
        current_time = time.time()

        # ⛔ chỉ gửi mỗi 5 phút
        if current_time - self.last_sent_time < 60:
            return

        self.last_sent_time = current_time

        try:
            conf = float(conf)
            percent = conf * 100  # convert sang %

            img_base64 = encode_frame_base64(frame)

            send_data_to_kafka(
                self.box_id,
                self.camera_id,
                percent,
                img_base64
            )

            logging.warning(
                f"🔥 FIRE ALERT camera {self.camera_id} | conf={percent:.2f}%"
            )

        except Exception as e:
            logging.error(f"Error send_fire_alert: {e}")


# ================= TEST LOCAL =================
if __name__ == "__main__":
    try:
        detector = FireDetector(
            box_id="31f52bb2-686f-48ae-a472-2c2aa1329d80",
            camera_id="bc4e9e35-cf47-4f1e-a83e-b4b2e5e2639d"
        )

        # đọc ảnh test
        img_path = "./output/images/output_images.jpg"

        if not os.path.exists(img_path):
            raise Exception("Image not found!")

        frame = cv2.imread(img_path)

        print("Start test sending every 5 minutes...")

        while True:
            detector.send_fire_alert(conf=0.5, frame=frame)
            time.sleep(1)  # loop nhanh nhưng gửi bị limit 5 phút

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        close_producer()