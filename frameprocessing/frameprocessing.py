import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "dino_miss.mp4"
OUTPUT_VIDEO_PATH = "dino_miss_pose.mp4"

# Hold detection is disabled until the Roboflow model is ready.
# from inference_sdk import InferenceHTTPClient
# import supervision as sv
# HOLDS_MODEL_ID = "YOUR_HOLDS_MODEL_ID/1"
# ROBOFLOW_API_KEY = "your-roboflow-api-key"
# rf_client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key=ROBOFLOW_API_KEY
# )
pose_model = YOLO("yolo11n-pose.pt") # Downloads automatically

def process_frame(frame):
    # Hold detection is intentionally skipped until its model is available.
    # holds_result = rf_client.infer(frame, model_id=HOLDS_MODEL_ID)
    # holds_detections = sv.Detections.from_inference(holds_result)

    # Estimate pose locally with YOLO.
    pose_results = pose_model(frame, verbose=False)
    return pose_results[0].plot(img=frame.copy())

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