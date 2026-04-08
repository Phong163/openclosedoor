import cv2
import numpy as np
import time
import os
import base64
import logging

from core.detector import YoloDetector
from core.processor import PersonCounter
from core.zones import Zone
from tool.rtsp_stream import RTSPStream
from tool.utils import check_base64_image
from services.kafka_service import KafkaService

class OverloadDetector:
    def __init__(
        self,
        camera_id,
        output_path,
        config_all,
        model,
        size=480,
        show_video=False,
        send_api=False,
        video_path=None
    ):
        camera_configs = config_all["cameras"]

        if camera_id not in camera_configs:
            raise ValueError(f"No config for camera {camera_id}")

        config = camera_configs[camera_id]

        # ===== CONFIG =====
        self.camera_id = camera_id
        self.cam_id = config["cam_id"]
        self.rtsp_url = config["rtsp_url"]

        self.zone_1 = config["zone_1"]
        self.zone_id = config["zone_id"]
        self.zone_2 = config["zone_2"]
        self.zone_2 = config["zone_2"]
        self.max_person = config["max_person"]

        self.box_id = config["box_id"]
        self.kafka_interval = config["kafka_interval"] * 60

        self.show_video = show_video
        self.send_api = send_api
        self.video_path = video_path
        self.output_path = output_path
        self.size = size

        # ===== STATE =====
        self.is_messy = False
        self.last_kafka_time = None

        # ===== STREAM =====
        self.rtsp_stream = None

        if self.video_path and os.path.exists(self.video_path):
            logging.info(f"[Cam {self.cam_id}] Using video file")
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_video_file = True
        else:
            logging.info(f"[Cam {self.cam_id}] Using RTSP")
            self.rtsp_stream = RTSPStream(self.rtsp_url, self.cam_id)
            self.rtsp_stream.start()
            self.cap = self.rtsp_stream.cap
            self.is_video_file = False

        # ===== FRAME INFO =====
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ===== ZONE =====
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.zone_1 = Zone(self.zone_1, dummy_frame.shape)
        self.zone_2 = Zone(self.zone_2, dummy_frame.shape)
        # ===== MODEL PIPELINE =====
        self.detector = YoloDetector(model, img_size=self.size)

        self.processor = PersonCounter(
            zone_poly1=self.zone_1.poly,
            zone_poly2=self.zone_2.poly,
            max_person=self.max_person
        )
        # ===== KAFKA =====
        self.kafka_service = KafkaService(config_all)

        # ===== VIDEO WRITER =====
        if self.show_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_file = output_path.replace(".mp4", f"_cam{camera_id}.mp4")
            self.out = cv2.VideoWriter(out_file, fourcc, self.fps, (self.width, self.height))

    # =========================
    # UTIL
    # =========================
    def encode_frame(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    # =========================
    # KAFKA
    # =========================
    def handle_kafka(self, count, frame):
        now = time.time()

        if (
            self.last_kafka_time is not None
            and now - self.last_kafka_time < self.kafka_interval
        ):
            return

        if count > self.max_person:
            self._send_kafka(count, frame)
            logging.info(f"[Kafka] Overload cam {self.camera_id}")

            self.last_kafka_time = now
    def _send_kafka(self, count, frame):
        frame = cv2.resize(frame, (720, 640))
        img_base64 = self.encode_frame(frame)
        check_base64_image(img_base64)

        self.kafka_service.send(
            self.box_id,
            self.camera_id,
            self.zone_id,
            count,   # gửi count thay vì percent
            img_base64
        )

    # =========================
    # MAIN PROCESS
    # =========================
    def process_frame(self, frame):
        annotated = frame.copy()

        # ===== DETECT =====
        detections = self.detector.detect(frame)

        # ===== PROCESS =====
        count = self.processor.compute(detections)

        # ===== DRAW BOX =====
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['class_name']} {d['conf']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)

            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ===== STATUS =====
        status = "QUA TAI" if count > self.max_person else "BINH THUONG"
        color = (0, 0, 255) if count > self.max_person else (0, 255, 0)

        cv2.putText(
            annotated,
            f"Count: {count}/{self.max_person} {status}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            3
        )

        # ===== DRAW ZONE =====
        self.zone_1.draw(annotated)
        self.zone_2.draw(annotated, color=(0,255,0))
        # print('percent:',percent)
        # ===== KAFKA =====
        if self.send_api:
            self.handle_kafka(count, annotated)

        return annotated

    # =========================
    # RUN
    # =========================
    def run(self):

        while True:
            if self.is_video_file:
                ret, frame = self.cap.read()
            else:
                ret, frame = self.rtsp_stream.get_frame()

            if not ret or frame is None:
                time.sleep(0.1)
                continue

            output = self.process_frame(frame)

            if self.show_video:
                self.out.write(output)

                show = cv2.resize(output, (720, 640))
                cv2.imshow(f"Camera {self.camera_id}", show)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if self.rtsp_stream:
            self.rtsp_stream.stop()

        if self.show_video:
            self.out.release()
            cv2.destroyAllWindows()