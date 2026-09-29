import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.engine.results import Results
from ultralytics.utils.nms import non_max_suppression

# --- CONFIGURATION ---
VIDEO_PATH = "speed_climb.mp4"
OUTPUT_VIDEO_PATH = "dino_miss_pose.mp4"
MODEL_PATH = "yolo11m-pose.onnx"
IMAGE_SIZE = 960
CONFIDENCE = 0.2

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

def process_frame(frame):
    # Hold detection is intentionally skipped until its model is available.
    # holds_result = rf_client.infer(frame, model_id=HOLDS_MODEL_ID)
    # holds_detections = sv.Detections.from_inference(holds_result)

    height, width = frame.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_left = (IMAGE_SIZE - resized_width) // 2
    pad_top = (IMAGE_SIZE - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height))
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
        return frame.copy()

    detections[:, [0, 2]] = (detections[:, [0, 2]] - pad_left) / scale
    detections[:, [1, 3]] = (detections[:, [1, 3]] - pad_top) / scale
    keypoints = detections[:, 6:].reshape(-1, 17, 3)
    keypoints[:, :, :2] = (
        keypoints[:, :, :2] - torch.tensor([pad_left, pad_top])
    ) / scale
    result = Results(
        orig_img=frame,
        path="",
        names={0: "person"},
        boxes=detections[:, :6],
        keypoints=keypoints,
    )
    return result.plot(img=frame.copy())

# --- VIDEO LOOP ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not open output video: {OUTPUT_VIDEO_PATH}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame)
        writer.write(annotated)
        cv2.imshow("Pose Estimation", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print(f"Saved pose video to {OUTPUT_VIDEO_PATH}")