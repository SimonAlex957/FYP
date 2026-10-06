import csv
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RESULT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("roboflow_hold_results.json")
if not RESULT_PATH.exists():
    RESULT_PATH = Path("roboflow_hold_result.json")
IMAGE_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("speed_climb_bad_angle.jpg")
LAYOUT_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("speed_wall_holds.csv")
OUTPUT_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("roboflow_hand_hold_overlay.png")
REPORT_PATH = OUTPUT_PATH.with_suffix(".txt")
MAX_RANKED_CANDIDATES = 40


def load_layout():
    rows = list(csv.DictReader(LAYOUT_PATH.open(encoding="utf-8")))
    all_points = np.float32([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    hand_indices = [index for index, row in enumerate(rows) if row["type"].lower() == "hand"]
    return rows, all_points, all_points[hand_indices], hand_indices


def load_candidates():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    predictions = result[0]["predictions"]["predictions"]
    predictions = [
        prediction
        for prediction in predictions
        if str(prediction.get("class_id", prediction.get("class"))) == "0"
        and float(prediction.get("confidence", 0)) >= 0.6
    ]
    predictions.sort(
        key=lambda item: (
            float(item.get("confidence", 0)),
            float(item.get("width", 0)) * float(item.get("height", 0)),
        ),
        reverse=True,
    )
    candidates = []
    for prediction in predictions:
        center = np.float32([prediction["x"], prediction["y"]])
        if all(np.linalg.norm(center - previous[0]) > 8 for previous in candidates):
            candidates.append((center, prediction))
    return candidates


MATCH_DISTANCE = 25.0


def project(points, transform):
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)


def upright_transform(source, projected):
    if np.std(projected[:, 0]) < 1e-6 or np.std(projected[:, 1]) < 1e-6:
        return False
    x_correlation = np.corrcoef(source[:, 0], projected[:, 0])[0, 1]
    y_correlation = np.corrcoef(source[:, 1], projected[:, 1])[0, 1]
    return abs(x_correlation) > 0.35 and y_correlation < -0.75


def score_transform(source, candidates, transform):
    projected = project(source, transform)
    candidate_points = np.float32([candidate[0] for candidate in candidates])
    distances = np.linalg.norm(projected[:, None] - candidate_points[None, :], axis=2)
    pairs = [
        (float(distance), source_index, candidate_index)
        for source_index, row in enumerate(distances)
        for candidate_index, distance in enumerate(row)
        if distance <= MATCH_DISTANCE
    ]
    pairs.sort()
    used_sources = set()
    used_candidates = set()
    matched_sources = []
    matched_candidates = []
    for _, source_index, candidate_index in pairs:
        if source_index in used_sources or candidate_index in used_candidates:
            continue
        used_sources.add(source_index)
        used_candidates.add(candidate_index)
        matched_sources.append(source_index)
        matched_candidates.append(candidate_index)
    return len(matched_sources), matched_sources, matched_candidates


def fit_hand_layout(hand_points, candidates):
    best = None
    candidate_points = np.float32([candidate[0] for candidate in candidates])
    if len(hand_points) < 4 or len(candidate_points) < 4:
        return None

    source_groups = itertools.combinations(range(len(hand_points)), 4)
    ranked_candidate_count = min(len(candidate_points), MAX_RANKED_CANDIDATES)
    candidate_groups = [
        tuple(range(start, start + 4))
        for start in range(ranked_candidate_count - 3)
    ]
    permutations = list(itertools.permutations(range(4)))
    for source_indices in source_groups:
        for candidate_indices in candidate_groups:
            source_sample = hand_points[list(source_indices)]
            candidate_sample = candidate_points[list(candidate_indices)]

            for permutation in permutations:
                transform, _ = cv2.findHomography(
                    source_sample,
                    candidate_sample[list(permutation)],
                    0,
                )
                if transform is None:
                    continue
                projected = project(hand_points, transform)
                if not np.isfinite(projected).all() or not upright_transform(hand_points, projected):
                    continue
                score = score_transform(hand_points, candidates, transform)
                if best is None or score[0] > best[0]:
                    best = score[0], score[1], score[2], transform

    if best is None:
        return None

    _, matched_sources, matched_candidates, transform = best
    if len(matched_sources) >= 4:
        refined, _ = cv2.findHomography(
            hand_points[matched_sources],
            candidate_points[matched_candidates],
            0,
        )
        if refined is not None:
            refined_score = score_transform(hand_points, candidates, refined)
            if refined_score[0] >= best[0]:
                best = refined_score[0], refined_score[1], refined_score[2], refined
    return best


def render(rows, all_points, hand_indices, candidates, fit):
    image = cv2.imread(str(IMAGE_PATH))
    hand_match_indices = set(fit[1])
    matched_candidate_indices = set(fit[2])
    transform = fit[3]
    projected = project(all_points, transform)
    candidate_points = np.float32([candidate[0] for candidate in candidates])

    for candidate_index, point in enumerate(candidate_points):
        color = (255, 0, 0) if candidate_index in matched_candidate_indices else (255, 0, 255)
        cv2.circle(image, tuple(np.round(point).astype(int)), 4, color, 1)

    for index, (row, point) in enumerate(zip(rows, projected)):
        center = tuple(np.round(point).astype(int))
        is_hand = index in hand_indices
        hand_index = hand_indices.index(index) if is_hand else None
        is_matched = is_hand and hand_index in hand_match_indices
        color = (0, 220, 0) if is_matched else (0, 0, 255) if is_hand else (0, 200, 255)
        radius = 10 if is_hand else 7
        cv2.circle(image, center, radius, color, 2)
        cv2.putText(
            image,
            f"{row['panel']}-{row['hold']} {'H' if is_hand else 'F'}",
            (center[0] + 8, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        f"hand matches: {fit[0]}/{len(hand_indices)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OUTPUT_PATH), image)


rows, all_points, hand_points, hand_indices = load_layout()
candidates = load_candidates()
fit = fit_hand_layout(hand_points, candidates)
if fit is None:
    raise RuntimeError("Could not find an upright homography fit for the hand holds")
render(rows, all_points, hand_indices, candidates, fit)
REPORT_PATH.write_text(
    "\n".join(
        [
            f"candidates={len(candidates)}",
            f"hand_matches={fit[0]}/{len(hand_indices)}",
            "matched_rows=" + ",".join(
                str(hand_indices[index] + 1) for index in sorted(fit[1])
            ),
            "missed_rows=" + ",".join(
                str(hand_indices[index] + 1)
                for index in range(len(hand_indices))
                if index not in set(fit[1])
            ),
            "false_positive_candidates=" + ",".join(
                str(index + 1)
                for index in range(len(candidates))
                if index not in set(fit[2])
            ),
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(f"wrote {OUTPUT_PATH}: {fit[0]}/{len(hand_indices)} hand holds matched")
