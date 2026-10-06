import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.frameprocessing.hold_detection import SpeedWallHoldDetector


IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("speed_climb_bad_angle.jpg")
LAYOUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("speed_wall_holds.csv")
RESULT_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("roboflow_hold_result.json")
OUTPUT_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("speed_wall_coordinate_overlay.png")
REPORT_PATH = OUTPUT_PATH.with_suffix(".txt")


def load_layout():
    rows = list(csv.DictReader(LAYOUT_PATH.open(encoding="utf-8")))
    points = np.float32([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    hand_indices = [index for index, row in enumerate(rows) if row["type"].lower() == "hand"]
    return rows, points, points[hand_indices], hand_indices


def load_candidates():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    predictions = result[0]["predictions"]["predictions"]
    predictions = [
        prediction
        for prediction in predictions
        if str(prediction.get("class_id", prediction.get("class"))) == "0"
        and float(prediction.get("confidence", 0)) >= 0.6
    ]
    predictions.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
    candidates = []
    for prediction in predictions:
        center = np.array([float(prediction["x"]), float(prediction["y"])], dtype=np.float32)
        if all(np.linalg.norm(center - existing[0]) > 8 for existing in candidates):
            candidates.append((center, prediction))
    return candidates


def project(points, homography):
    return cv2.transform(points.reshape(1, -1, 2), homography).reshape(-1, 2)


def score_transform(source, candidate_points, homography):
    projected = project(source, homography)
    distances = np.linalg.norm(projected[:, None, :] - candidate_points[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    nearest_indices = distances.argmin(axis=1)
    order = np.argsort(nearest)
    used = set()
    matched = []
    for source_index in order:
        candidate_index = int(nearest_indices[source_index])
        if nearest[source_index] <= 18 and candidate_index not in used:
            used.add(candidate_index)
            matched.append(int(source_index))
    return len(matched), matched, projected, nearest


def is_upright_wall(source, projected):
    """Reject transforms that flip the wall or rotate it into a sideways fit."""
    if np.std(projected[:, 0]) < 1e-6 or np.std(projected[:, 1]) < 1e-6:
        return False
    x_correlation = np.corrcoef(source[:, 0], projected[:, 0])[0, 1]
    y_correlation = np.corrcoef(source[:, 1], projected[:, 1])[0, 1]
    bottom_y = np.median(projected[source[:, 1] <= np.percentile(source[:, 1], 20), 1])
    top_y = np.median(projected[source[:, 1] >= np.percentile(source[:, 1], 80), 1])
    image_width = projected[:, 0].max() - projected[:, 0].min()
    image_height = projected[:, 1].max() - projected[:, 1].min()
    return (
        x_correlation > 0.35
        and y_correlation < -0.75
        and top_y < bottom_y
        and image_height > image_width
    )


def fit_layout(source, candidates):
    candidate_points = np.float32([candidate[0] for candidate in candidates])
    rng = random.Random(31)
    best = None
    source_count = len(source)
    candidate_count = len(candidate_points)
    for _ in range(60000):
        source_indices = rng.sample(range(source_count), 3)
        candidate_indices = rng.sample(range(candidate_count), 3)
        source_sample = source[source_indices]
        candidate_sample = candidate_points[candidate_indices]
        first_vector = source_sample[1] - source_sample[0]
        second_vector = source_sample[2] - source_sample[0]
        triangle_area = first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
        if abs(triangle_area) < 1e-5:
            continue
        transform = cv2.getAffineTransform(source_sample, candidate_sample)
        if transform is None:
            continue
        projected = project(source, transform)
        if not np.isfinite(projected).all():
            continue
        if not is_upright_wall(source, projected):
            continue
        scale = np.linalg.norm(projected.max(axis=0) - projected.min(axis=0))
        if not 300 < scale < 3000:
            continue
        score, matched, projected, nearest = score_transform(source, candidate_points, transform)
        if best is None or score > best[0]:
            best = score, matched, projected, nearest, transform
    return best


def render(rows, all_source, hand_indices, candidates, fit):
    image = cv2.imread(str(IMAGE_PATH))
    score, matched, projected, nearest, _ = fit
    transform = fit[-1]
    projected = project(all_source, transform)
    matched_set = {hand_indices[index] for index in matched}
    visual_candidates = SpeedWallHoldDetector(IMAGE_PATH).reference_holds
    for hold in visual_candidates:
        point = tuple(np.round(hold.center).astype(int))
        cv2.circle(image, point, 5, (255, 255, 0), 1)
    for center, prediction in candidates:
        point = tuple(np.round(center).astype(int))
        cv2.circle(image, point, 4, (255, 0, 0), 1)
    for index, (row, point) in enumerate(zip(rows, projected)):
        center = tuple(np.round(point).astype(int))
        is_matched = index in matched_set
        color = (0, 220, 0) if is_matched else (0, 0, 255)
        cv2.circle(image, center, 10, color, 2)
        cv2.putText(
            image,
            f"{index + 1} {row['panel']}-{row['hold']}",
            (center[0] + 8, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"hand matches: {score}/{len(hand_indices)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUTPUT_PATH), image)


rows, all_source, hand_source, hand_indices = load_layout()
candidates = load_candidates()
fit = fit_layout(hand_source, candidates)
if fit is None:
    raise RuntimeError("Could not find a plausible wall-coordinate transform")
render(rows, all_source, hand_indices, candidates, fit)
REPORT_PATH.write_text(
    f"candidates={len(candidates)}\nhand_matches={fit[0]}/{len(hand_indices)}\n"
    f"matched_rows={','.join(str(index + 1) for index in sorted(fit[1]))}\n",
    encoding="utf-8",
)
print(f"candidates={len(candidates)} hand_matches={fit[0]}/{len(hand_indices)} output={OUTPUT_PATH}")
print("matched_rows=" + ",".join(str(index + 1) for index in sorted(fit[1])))
