import logging
import json
from datetime import datetime
import pytz
from confluent_kafka import Producer


class KafkaService:
    def __init__(self, config):
        kafka_config = config["kafka"]

        self.topic = kafka_config["topic"]

        producer_config = {
            "bootstrap.servers": kafka_config["broker"],
            "client.id": kafka_config["client_id"],
            "security.protocol": kafka_config["security"]["protocol"],
            "retries": kafka_config["retries"],
            "retry.backoff.ms": kafka_config["retry_backoff_ms"],
        }

        try:
            self.producer = Producer(producer_config)
            logging.info("Kafka Producer initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Kafka Producer: {e}")
            raise

    def _delivery_report(self, err, msg):
        if err is not None:
            logging.error(f"Kafka send failed: {err}")
        else:
            logging.info("Kafka message sent successfully")

    def send(self, box_id, camera_id, state, conf,  image_base64):
        timestamp = int(datetime.now(
            pytz.timezone("Asia/Ho_Chi_Minh")
        ).timestamp())

        data = {
            "box_id": box_id,
            "cam_id": camera_id,
            "metric":"opencloseddoor",
            "door_state": state,
            "conf": int(conf * 100),
            "emb": image_base64,
            "timestamp": timestamp
            }
        # debug_data = data.copy()
        # debug_data["emb"] = "...base64_image_hidden..."

        # print("data:", debug_data)
        try:
            message = json.dumps(data).encode("utf-8")

            self.producer.produce(
                topic=self.topic,
                value=message,
                callback=self._delivery_report
            )

            self.producer.poll(0)

        except Exception as e:
            logging.error(f"Error sending message to Kafka: {e}")

    def close(self):
        try:
            self.producer.flush()
            logging.info("Kafka Producer closed successfully")
        except Exception as e:
            logging.error(f"Error closing Kafka Producer: {e}")