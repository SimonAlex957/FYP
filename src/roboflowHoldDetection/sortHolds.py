import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
RESULT_PATH = PROJECT_ROOT / "roboflowHoldDetection" / "roboflow_hold_result.json"
IMAGE_PATH = PROJECT_ROOT / "inputs" / "speed_climb_bad_angle.jpg"
OUTPUT_PATH = PROJECT_ROOT / "roboflowHoldDetection" / "speed_wall_hand_holds.json"
RED_RATIO_THRESHOLD = 0.15
LARGE_AREA_MULTIPLIER = 1.5


def load_predictions(path: Path) -> list[dict]:
	result = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(result, list):
		result = result[0]
	prediction_block = result.get("predictions", {})
	predictions = prediction_block.get("predictions", [])
	return [
		prediction
		for prediction in predictions
		if str(prediction.get("class_id", prediction.get("class"))) == "0"
	]


def red_ratio(image: np.ndarray, prediction: dict) -> float:
	height, width = image.shape[:2]
	center_x = float(prediction["x"])
	center_y = float(prediction["y"])
	box_width = float(prediction["width"])
	box_height = float(prediction["height"])
	left = max(0, int(round(center_x - box_width / 2)))
	top = max(0, int(round(center_y - box_height / 2)))
	right = min(width, int(round(center_x + box_width / 2)))
	bottom = min(height, int(round(center_y + box_height / 2)))
	if left >= right or top >= bottom:
		return 0.0

	hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
	red_mask = (
		cv2.inRange(hsv, np.array([0, 70, 40]), np.array([12, 255, 255]))
		| cv2.inRange(hsv, np.array([168, 70, 40]), np.array([179, 255, 255]))
	)
	return float(np.count_nonzero(red_mask)) / red_mask.size


def sort_holds(
	result_path: Path,
	image_path: Path,
	output_path: Path,
) -> list[dict]:
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Could not open image: {image_path}")

	predictions = load_predictions(result_path)
	if not predictions:
		output_path.write_text("[]\n", encoding="utf-8")
		return []

	areas = np.array(
		[float(item.get("width", 0)) * float(item.get("height", 0)) for item in predictions]
	)
	large_area_threshold = float(np.median(areas) * LARGE_AREA_MULTIPLIER)
	selected = []
	for prediction, area in zip(predictions, areas):
		current_red_ratio = red_ratio(image, prediction)
		if current_red_ratio < RED_RATIO_THRESHOLD or area < large_area_threshold:
			continue
		output_prediction = dict(prediction)
		output_prediction["area"] = float(area)
		output_prediction["red_ratio"] = round(current_red_ratio, 4)
		selected.append(output_prediction)

	selected.sort(key=lambda item: item["area"], reverse=True)
	output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
	return selected


if __name__ == "__main__":
	if len(sys.argv) > 1:
		RESULT_PATH = Path(sys.argv[1])
	if len(sys.argv) > 2:
		IMAGE_PATH = Path(sys.argv[2])
	if len(sys.argv) > 3:
		OUTPUT_PATH = Path(sys.argv[3])

	holds = sort_holds(RESULT_PATH, IMAGE_PATH, OUTPUT_PATH)
	print(f"wrote {OUTPUT_PATH}: {len(holds)} red large holds")
