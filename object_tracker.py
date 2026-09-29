import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


WINDOW_TITLE = "Webcam object recognition (Q/Esc to quit)"
BOX_COLOR = (0, 220, 0)


def find_model(model_name: str) -> str:
    """Use a local model file when one is available."""
    model_path = Path(model_name).expanduser()
    local_path = Path(__file__).parent / model_path

    if model_path.exists():
        return str(model_path.resolve())
    if local_path.exists():
        return str(local_path.resolve())
    return model_name


def draw_detection(frame: Any, box: Any, name: str) -> None:
    confidence = float(box.conf.item())
    track_id = None if box.id is None else int(box.id.item())
    label = f"{name} {confidence:.0%}"
    if track_id is not None:
        label += f" | ID {track_id}"

    x1, y1, x2, y2 = box.xyxy.int().tolist()[0]
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BOX_COLOR,
        2,
        cv2.LINE_AA,
    )


def annotate_frame(result: Any):
    frame = result.orig_img.copy()
    counts = Counter()

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            object_name = result.names[class_id]
            draw_detection(frame, box, object_name)
            counts[object_name] += 1

    summary = ", ".join(
        f"{name} ({count})" for name, count in counts.items()
    ) or "none"
    cv2.rectangle(
        frame, (0, 0), (min(frame.shape[1], 700), 38), (20, 20, 20), -1
    )
    cv2.putText(
        frame,
        f"Objects: {summary}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Webcam object recognition and tracking."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam index (default: 0)."
    )
    parser.add_argument(
        "--model", default="yolov8s.pt", help="YOLO model weights."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Minimum confidence (default: 0.25)."
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Inference image size (default: 640)."
    )
    return parser


def run(camera_index: int, model_name: str, confidence: float, image_size: int) -> None:
    if not 0 <= confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("--img-size must be positive.")

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam {camera_index}. Check camera permissions.")

    model = YOLO(model_name)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )[0]
            cv2.imshow(WINDOW_TITLE, annotate_frame(result))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args.camera, find_model(args.model), args.confidence, args.img_size)
    except (RuntimeError, ValueError) as error:
        build_parser().error(str(error))


if __name__ == "__main__":
    main()
