import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
DETECTIONS_PATH = PROJECT_ROOT / "roboflowHoldDetection" / "speed_wall_hand_holds.json"
KNOWN_HOLDS_PATH = PROJECT_ROOT / "KnownHoldLocations" / "speed_wall_holds.csv"
IMAGE_PATH = PROJECT_ROOT / "inputs" / "speed_climb_bad_angle.jpg"
OUTPUT_DIR = PROJECT_ROOT / "src" / "knownHandHoldOverlay"
OVERLAY_PATH = OUTPUT_DIR / "known_hand_holds_overlay.png"
REPORT_PATH = OUTPUT_DIR / "known_hand_holds_report.json"
MATCH_DISTANCE = 35.0


def load_known_hand_holds(path: Path) -> tuple[list[dict], np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    hand_rows = [row for row in rows if row["type"].lower() == "hand"]
    points = np.float32([[float(row["x_m"]), float(row["y_m"])] for row in hand_rows])
    return hand_rows, points


def load_detections(path: Path) -> list[dict]:
    detections = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(detections, list):
        raise ValueError("The detection JSON must contain a list of predictions")
    return detections


def project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def is_upright(source: np.ndarray, projected: np.ndarray) -> bool:
    if not np.isfinite(projected).all():
        return False
    if np.std(projected[:, 0]) < 1e-6 or np.std(projected[:, 1]) < 1e-6:
        return False
    x_correlation = np.corrcoef(source[:, 0], projected[:, 0])[0, 1]
    y_correlation = np.corrcoef(source[:, 1], projected[:, 1])[0, 1]
    return abs(x_correlation) > 0.25 and y_correlation < -0.5


def assign_matches(projected: np.ndarray, detections: list[dict]) -> tuple[list[int], list[int], list[float]]:
    detected_points = np.float32([[item["x"], item["y"]] for item in detections])
    distances = np.linalg.norm(projected[:, None, :] - detected_points[None, :, :], axis=2)
    possible_pairs = [
        (float(distance), known_index, detection_index)
        for known_index, row in enumerate(distances)
        for detection_index, distance in enumerate(row)
        if distance <= MATCH_DISTANCE
    ]
    possible_pairs.sort()
    used_known = set()
    used_detections = set()
    matched_known = []
    matched_detections = []
    matched_distances = []
    for distance, known_index, detection_index in possible_pairs:
        if known_index in used_known or detection_index in used_detections:
            continue
        used_known.add(known_index)
        used_detections.add(detection_index)
        matched_known.append(known_index)
        matched_detections.append(detection_index)
        matched_distances.append(distance)
    return matched_known, matched_detections, matched_distances


def fit_homography(known_points: np.ndarray, detections: list[dict]):
    if len(known_points) != 20 or len(detections) < 4:
        raise ValueError("Expected 20 known hand holds and at least 4 detections")

    known_order = np.argsort(known_points[:, 1])[::-1]
    detection_points = np.float32([[item["x"], item["y"]] for item in detections])
    detection_order = np.argsort(detection_points[:, 1])
    ordered_known = known_points[known_order]
    ordered_detections = detection_points[detection_order]

    best = None
    for selected_indices in itertools.combinations(range(len(ordered_detections)), 20):
        selected_detections = ordered_detections[list(selected_indices)]
        homography, _ = cv2.findHomography(ordered_known, selected_detections, 0)
        if homography is None:
            continue
        projected = project(known_points, homography)
        if not is_upright(known_points, projected):
            continue
        matched_known, matched_detections, distances = assign_matches(projected, detections)
        score = (
            len(matched_known),
            -float(np.median(distances)) if distances else -float("inf"),
            -float(np.mean(distances)) if distances else -float("inf"),
        )
        if best is None or score > best[0]:
            best = score, homography, projected, matched_known, matched_detections, distances
            if len(matched_known) == len(known_points):
                break
        if best is not None and best[0][0] == len(known_points):
            break

    if best is None:
        raise RuntimeError("No upright homography was found")

    _, homography, projected, matched_known, matched_detections, distances = best
    if len(matched_known) >= 4:
        refined, _ = cv2.findHomography(
            known_points[matched_known],
            detection_points[matched_detections],
            0,
        )
        if refined is not None and is_upright(known_points, project(known_points, refined)):
            refined_projected = project(known_points, refined)
            refined_known, refined_detections, refined_distances = assign_matches(
                refined_projected, detections
            )
            refined_score = (
                len(refined_known),
                -float(np.median(refined_distances)) if refined_distances else -float("inf"),
                -float(np.mean(refined_distances)) if refined_distances else -float("inf"),
            )
            if refined_score >= best[0]:
                homography = refined
                projected = refined_projected
                matched_known = refined_known
                matched_detections = refined_detections
                distances = refined_distances

    return homography, projected, matched_known, matched_detections, distances


def render_overlay(
    image: np.ndarray,
    known_rows: list[dict],
    projected: np.ndarray,
    detections: list[dict],
    matched_known: list[int],
    matched_detections: list[int],
) -> None:
    matched_known_set = set(matched_known)
    matched_detection_set = set(matched_detections)
    for detection_index, detection in enumerate(detections):
        center = (int(round(detection["x"])), int(round(detection["y"])))
        color = (255, 0, 0) if detection_index in matched_detection_set else (255, 0, 255)
        cv2.circle(image, center, 5, color, 2)

    for known_index, (row, point) in enumerate(zip(known_rows, projected)):
        center = (int(round(point[0])), int(round(point[1])))
        color = (0, 220, 0) if known_index in matched_known_set else (0, 0, 255)
        cv2.circle(image, center, 10, color, 2)
        cv2.putText(
            image,
            f"{known_index + 1} {row['panel']}-{row['hold']}",
            (center[0] + 8, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}")

    known_rows, known_points = load_known_hand_holds(KNOWN_HOLDS_PATH)
    detections = load_detections(DETECTIONS_PATH)
    homography, projected, matched_known, matched_detections, distances = fit_homography(
        known_points, detections
    )

    overlay = image.copy()
    render_overlay(overlay, known_rows, projected, detections, matched_known, matched_detections)
    if not cv2.imwrite(str(OVERLAY_PATH), overlay):
        raise OSError(f"Could not write overlay: {OVERLAY_PATH}")

    matched_known_set = set(matched_known)
    matched_detection_set = set(matched_detections)
    report = {
        "known_hold_count": len(known_points),
        "detection_count": len(detections),
        "matched_count": len(matched_known),
        "missed_known_holds": [
            {"index": index + 1, **known_rows[index], "projected_x": float(projected[index, 0]), "projected_y": float(projected[index, 1])}
            for index in range(len(known_rows))
            if index not in matched_known_set
        ],
        "false_positive_detection_indices": [
            index + 1 for index in range(len(detections)) if index not in matched_detection_set
        ],
        "mean_match_distance": float(np.mean(distances)) if distances else None,
        "homography": homography.tolist(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OVERLAY_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(f"matched {len(matched_known)}/{len(known_points)} known hand holds")


if __name__ == "__main__":
    main()
