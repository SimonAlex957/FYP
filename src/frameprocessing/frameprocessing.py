import argparse
import cv2
import csv
import os
import numpy as np
import onnxruntime as ort
import torch
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from ultralytics.engine.results import Results
from ultralytics.utils.nms import non_max_suppression
from frameprocessing.hold_detection import RoboflowSpeedWallHoldDetector, SpeedWallHoldDetector

parser = argparse.ArgumentParser(description="Run pose estimation on a video.")
parser.add_argument(
    "--note",
    default="",
    help="Note describing this run or a change being tested.",
)
parser.add_argument(
    "--timing-csv",
    default="pose_runs.csv",
    help="CSV file where total video processing times are appended.",
)
parser.add_argument(
    "--hold-reference",
    default="Screenshot 2026-09-29 162720.png",
    help="Reference image of the empty speed wall layout.",
)
parser.add_argument(
    "--hold-lane",
    choices=["auto", "left", "right", "all"],
    default="auto",
    help="Speed-wall lane to render; auto selects the lane nearest the climber.",
)
args = parser.parse_args()

# --- CONFIGURATION ---
VIDEO_PATH = "speed_climb.mp4"
OUTPUT_VIDEO_PATH = "dino_miss_pose.mp4"
MODEL_PATH = "yolo11m-pose.onnx"
IMAGE_SIZE = 960
CONFIDENCE = 0.2
REDETECT_INTERVAL = 10
CROP_PADDING = 0.3

# Hold detection is disabled until the Roboflow model is ready.
# from inference_sdk import InferenceHTTPClient
# import supervision as sv
# HOLDS_MODEL_ID = "YOUR_HOLDS_MODEL_ID/1"
# ROBOFLOW_API_KEY = "your-roboflow-api-key"
# rf_client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key=ROBOFLOW_API_KEY
# )
available_providers = ort.get_available_providers()
inference_provider = (
    "DmlExecutionProvider"
    if "DmlExecutionProvider" in available_providers
    else "CPUExecutionProvider"
)
pose_session = ort.InferenceSession(
    MODEL_PATH,
    providers=[inference_provider, "CPUExecutionProvider"],
)
pose_input_name = pose_session.get_inputs()[0].name
print(f"Using inference provider: {pose_session.get_providers()[0]}")
if os.environ.get("ROBOFLOW_API_KEY"):
    hold_detector = RoboflowSpeedWallHoldDetector(args.hold_lane)
    print("Using Roboflow workflow hold detection: 20 large + 11 small")
else:
    hold_detector = SpeedWallHoldDetector(args.hold_reference, args.hold_lane)
    print(f"Using template hold detection: {len(hold_detector.reference_holds)} candidates")

tracking_box = None
tracking_velocity = np.zeros(4, dtype=np.float32)
last_full_detection_frame = -REDETECT_INTERVAL

def run_pose(image):
    # Hold detection is intentionally skipped until its model is available.
    # holds_result = rf_client.infer(frame, model_id=HOLDS_MODEL_ID)
    # holds_detections = sv.Detections.from_inference(holds_result)

    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_left = (IMAGE_SIZE - resized_width) // 2
    pad_top = (IMAGE_SIZE - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        IMAGE_SIZE - resized_height - pad_top,
        pad_left,
        IMAGE_SIZE - resized_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    model_input = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    raw_predictions = pose_session.run(None, {pose_input_name: model_input})[0]
    detections = non_max_suppression(
        torch.from_numpy(raw_predictions),
        conf_thres=CONFIDENCE,
        iou_thres=0.7,
        nc=1,
    )[0]

    if len(detections) == 0:
        return detections

    detections[:, [0, 2]] = (detections[:, [0, 2]] - pad_left) / scale
    detections[:, [1, 3]] = (detections[:, [1, 3]] - pad_top) / scale
    keypoints = detections[:, 6:].reshape(-1, 17, 3)
    keypoints[:, :, :2] = (
        keypoints[:, :, :2] - torch.tensor([pad_left, pad_top])
    ) / scale
    return detections

def render_pose(frame, detections):
    if len(detections) == 0:
        return frame.copy()

    keypoints = detections[:, 6:].reshape(-1, 17, 3)
    result = Results(
        orig_img=frame,
        path="",
        names={0: "person"},
        boxes=detections[:, :6],
        keypoints=keypoints,
    )
    return result.plot(img=frame.copy())

def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2

def choose_person(detections):
    if len(detections) == 0:
        return None
    best_index = int(torch.argmax(detections[:, 4]).item())
    return detections[best_index:best_index + 1].clone()

def process_frame(frame, frame_index):
    global tracking_box, tracking_velocity, last_full_detection_frame

    height, width = frame.shape[:2]
    needs_full_detection = (
        tracking_box is None
        or frame_index - last_full_detection_frame >= REDETECT_INTERVAL
    )

    if needs_full_detection:
        detections = choose_person(run_pose(frame))
        last_full_detection_frame = frame_index
        if detections is None:
            tracking_box = None
            return hold_detector.annotate(frame)
        current_box = detections[0, :4].cpu().numpy()
    else:
        predicted_box = tracking_box + tracking_velocity
        box_width = predicted_box[2] - predicted_box[0]
        box_height = predicted_box[3] - predicted_box[1]
        crop_box = (
            predicted_box[0] - box_width * CROP_PADDING,
            predicted_box[1] - box_height * CROP_PADDING,
            predicted_box[2] + box_width * CROP_PADDING,
            predicted_box[3] + box_height * CROP_PADDING,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = clamp_box(crop_box, width, height)
        crop_detections = choose_person(run_pose(frame[crop_y1:crop_y2, crop_x1:crop_x2]))

        if crop_detections is None:
            tracking_box = None
            return process_frame(frame, frame_index)

        crop_detections[:, [0, 2]] += crop_x1
        crop_detections[:, [1, 3]] += crop_y1
        keypoints = crop_detections[:, 6:].reshape(-1, 17, 3)
        keypoints[:, :, 0] += crop_x1
        keypoints[:, :, 1] += crop_y1
        detections = crop_detections
        current_box = detections[0, :4].cpu().numpy()

    if tracking_box is not None:
        tracking_velocity = current_box - tracking_box
    tracking_box = current_box
    climber_center_x = float((current_box[0] + current_box[2]) / 2)
    pose_keypoints = detections[0, 6:].reshape(17, 3).cpu().numpy()
    holds_annotated = hold_detector.annotate(
        frame,
        climber_center_x,
        pose_keypoints,
    )
    return render_pose(holds_annotated, detections)

# --- VIDEO LOOP ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
run_id = datetime.now(timezone.utc).isoformat()
timing_path = Path(args.timing_csv)
timing_file_exists = timing_path.exists() and timing_path.stat().st_size > 0
timing_file = timing_path.open("a", newline="", encoding="utf-8")
timing_writer = csv.DictWriter(
    timing_file,
    fieldnames=[
        "run_id",
        "total_processing_time_seconds",
        "total_processing_time_minutes",
        "frames_processed",
        "inference_provider",
        "note",
        "input_video",
        "output_video",
    ],
)
if not timing_file_exists:
    timing_writer.writeheader()

writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    cap.release()
    timing_file.close()
    raise RuntimeError(f"Could not open output video: {OUTPUT_VIDEO_PATH}")

run_start_time = perf_counter()
try:
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame, frame_index)
        writer.write(annotated)
        cv2.imshow("Pose Estimation", annotated)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

total_processing_time_seconds = perf_counter() - run_start_time
timing_writer.writerow(
    {
        "run_id": run_id,
        "total_processing_time_seconds": round(total_processing_time_seconds, 3),
        "total_processing_time_minutes": round(total_processing_time_seconds / 60, 3),
        "frames_processed": frame_index,
        "inference_provider": pose_session.get_providers()[0],
        "note": args.note,
        "input_video": VIDEO_PATH,
        "output_video": OUTPUT_VIDEO_PATH,
    }
)
timing_file.flush()
timing_file.close()
print(f"Saved pose video to {OUTPUT_VIDEO_PATH}")
print(f"Total processing time: {total_processing_time_seconds:.3f} seconds")