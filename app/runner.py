
from app.doorclassify import Classify   
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# =========================
# CAMERA SELECT
# =========================
def get_camera_ids(config_all, args):
    camera_configs = config_all["cameras"]

    if args.video_path:
        return [0]

    if args.camera_id is not None:
        return args.camera_id

    return [int(k) for k in camera_configs.keys()]


# =========================
# RUN 1 CAMERA
# =========================
def start_camera(camera_id, args, config_all, model):
    try:
        tracker = Classify(
            camera_id=camera_id,
            output_path=args.output,
            config_all=config_all,
            model=model,
            size=args.imgsz,
            show_video=args.show_video,
            send_api=args.send_api,
            video_path=args.video_path
        )
        tracker.run()

    except Exception as e:
        logging.error(f"[Camera {camera_id}] Error: {e}")


# =========================
# MULTI CAMERA
# =========================
def run_multi_camera(camera_ids, args, config_all, model):

    # chạy video file → không cần thread
    if args.video_path:
        start_camera(camera_ids[0], args, config_all, model)
        return

    threads = []

    for cam_id in camera_ids:
        t = threading.Thread(
            target=start_camera,
            args=(cam_id, args, config_all, model)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
