# Real-time object detection and tracking

This example reads frames from a webcam with OpenCV, detects objects using a
pre-trained YOLO model, and tracks them with ByteTrack. The live output
includes each object's class, confidence, bounding box, and persistent
tracking ID.

Each bounding box displays the recognized object class, confidence score, and
ByteTrack ID. A counter at the top of the window shows how many objects of
each recognized class are currently visible.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the default webcam:

```bash
python object_tracker.py
```

If the default camera is unavailable, select another camera index:

```bash
python object_tracker.py --camera 1
```

Press `q` or `Esc` to stop the webcam.

The YOLO weights download automatically on the first run.
