import argparse
import logging
from tool.utils import get_config
from app.runner import get_camera_ids, run_multi_camera
from models.yolo_onnx_classify import YoloONNXCLASSIFY


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# ARGS
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Camera AI Pipeline")

    parser.add_argument(
        "--camera_id",
        type=int,
        nargs="*",
        default=None,
        help="Camera IDs (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output/output.mp4"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=384
    )

    parser.add_argument(
        "--show_video",
        action="store_true"
    )

    parser.add_argument(
        "--send_api",
        action="store_true"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Run with video file instead of RTSP"
    )

    return parser.parse_args()

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    # load config
    config_all = get_config()

    # load model (CHỈ LOAD 1 LẦN)
    model = YoloONNXCLASSIFY(config_all["models"]["yoloONNX_weight"])

    # chọn camera
    camera_ids = get_camera_ids(config_all, args)

    logging.info(f"Starting cameras: {camera_ids}")

    # chạy
    run_multi_camera(camera_ids, args, config_all, model)


if __name__ == "__main__":
    main()