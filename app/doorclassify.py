import cv2
import numpy as np
import os
import base64
import logging
from core.zones import Zone
import time
from core.detector import YoloClassify
from tool.rtsp_stream import RTSPStream
from tool.utils import check_base64_image
from services.kafka_service import KafkaService

class Classify:
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

        self.box_id = config["box_id"]

        self.show_video = show_video
        self.send_api = send_api
        self.video_path = video_path
        self.output_path = output_path
        self.size = size

        # ===== STATE =====
        self.prev_state = None
        self.waiting_to_100 = False
        self.waiting_state = None
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
        # ===== MODEL PIPELINE =====
        self.detector = YoloClassify(model, img_size=self.size)
         # ===== ZONE =====
        dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.zone = Zone(self.zone_1, dummy_frame.shape)
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
    def handle_kafka(self, state, conf, frame):

        self._send_kafka(state, conf, frame)
        logging.info(f"[Kafka] Overload cam {self.camera_id}")

    def _send_kafka(self, state, conf, frame):
        frame = cv2.resize(frame, (720, 640))
        img_base64 = self.encode_frame(frame)
        check_base64_image(img_base64)

        self.kafka_service.send(
            self.box_id,
            self.cam_id,
            state,
            conf,   
            img_base64
        )
    # =========================
    # MAIN PROCESS
    def process_frame(self, frame):
        annotated = frame.copy()
        height, width, _ = frame.shape

        # ===== CROP ZONE =====
        pts = [(int(x * width), int(y * height)) for x, y in self.zone_1]

        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        x_max = max(p[0] for p in pts)
        y_max = max(p[1] for p in pts)

        crop = frame[y_min:y_max, x_min:x_max]

        if crop is None or crop.size == 0:
            return annotated

        # ===== CLASSIFY =====
        result = self.detector.predict(crop)

        current_state = None
        conf = 0
        self.zone.draw(annotated)
        if result:
            current_state = result["class_name"]   # "open" hoặc "close"
            conf = result["confidence"]

            # label = f"{current_state}_{(conf*100):.0f}%"
            # color = (0, 255, 0) if current_state == "open" else (0, 0, 255)

            # cv2.putText(
            #     annotated,
            #     label,
            #     (1650, 120),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     5,
            #     color,
            #     8
            # )
            #===== STATE to waiting_to_100100 conf LOGIC =====
            if self.waiting_to_100:
                if conf > 0.96 and self.waiting_state is not None and current_state == self.waiting_state:
                    logging.info(f"Waiting to 100 state={self.waiting_state}, conf={conf:.2f}")
                    if self.send_api:
                        logging.info(f"[SEND KAFKA] state={self.waiting_state}, conf={conf:.2f}")
                        self.handle_kafka(self.waiting_state, conf, annotated)
                    self.waiting_to_100 = False
                    self.waiting_state = None
               
            # ===== STATE CHANGE LOGIC =====
            if self.prev_state is not None and current_state != self.prev_state:
                logging.info(
                    f"[STATE CHANGE] Cam {self.camera_id}: {self.prev_state} -> {current_state}"
                )
                if current_state == "open":
                    self.waiting_to_100 = True
                    self.waiting_state = "open"
                elif current_state == "closed":
                    self.waiting_to_100 = True
                    self.waiting_state = "closed"
                # if self.send_api:
                #     # ===== GỬI =====
                #     if self.waiting_state:
                #         logging.info(f"[SEND KAFKA] state={self.waiting_state}, conf={conf:.2f}")
                #         self.handle_kafka(self.waiting_state, conf, annotated)

            # update state
            self.prev_state = current_state
    
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