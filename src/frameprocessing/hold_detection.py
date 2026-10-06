from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceConfiguration, InferenceHTTPClient


@dataclass
class SpeedHold:
    center: tuple[float, float]
    kind: str
    area: float


class SpeedWallHoldDetector:
    """Align a reference speed-wall layout and render expected hold positions."""

    def __init__(self, reference_path: str | Path, lane: str = "auto") -> None:
        self.reference = cv2.imread(str(reference_path))
        if self.reference is None:
            raise FileNotFoundError(f"Could not open hold reference image: {reference_path}")
        if lane not in {"auto", "left", "right", "all"}:
            raise ValueError("lane must be 'auto', 'left', 'right', or 'all'")

        self.lane = lane
        self.reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=2500)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(
            self.reference_gray, None
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference_holds = self._find_reference_holds()
        self.last_homography = None
        self.contact_streak: dict[int, int] = {}
        self.previous_contact_points = None

        if len(self.reference_holds) == 0:
            raise RuntimeError("No colored hold candidates were found in the reference image")

    def _find_reference_holds(self) -> list[SpeedHold]:
        hsv = cv2.cvtColor(self.reference, cv2.COLOR_BGR2HSV)
        saturation_mask = cv2.inRange(hsv, np.array([0, 55, 45]), np.array([179, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[tuple[float, float], float]] = []
        reference_height, reference_width = self.reference.shape[:2]
        minimum_area = max(20.0, reference_width * reference_height * 0.00002)
        maximum_area = reference_width * reference_height * 0.02
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_area or area > maximum_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
            candidates.append((center, area))

        candidates.sort(key=lambda item: (item[0][1], item[0][0]))
        deduplicated: list[tuple[tuple[float, float], float]] = []
        for center, area in candidates:
            if all(np.linalg.norm(np.subtract(center, other_center)) > 12 for other_center, _ in deduplicated):
                deduplicated.append((center, area))

        if not deduplicated:
            return []
        areas = np.array([area for _, area in deduplicated])
        large_threshold = float(np.median(areas) * 1.8)
        return [
            SpeedHold(center, "large" if area >= large_threshold else "small", area)
            for center, area in deduplicated
        ]

    def _align_to_frame(self, frame: np.ndarray) -> np.ndarray | None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self.orb.detectAndCompute(frame_gray, None)
        if frame_descriptors is None or self.reference_descriptors is None:
            return self.last_homography

        matches = self.matcher.knnMatch(self.reference_descriptors, frame_descriptors, k=2)
        good_matches = [
            first
            for first, second in matches
            if first.distance < 0.72 * second.distance
        ]
        if len(good_matches) >= 4:
            reference_points = np.float32(
                [self.reference_keypoints[m.queryIdx].pt for m in good_matches]
            )
            frame_points = np.float32(
                [frame_keypoints[m.trainIdx].pt for m in good_matches]
            )
            homography, inlier_mask = cv2.findHomography(
                reference_points, frame_points, cv2.RANSAC, 5.0
            )
            if homography is not None and int(inlier_mask.sum()) >= 4:
                self.last_homography = homography

        return self.last_homography

    def _select_lane(self, holds: list[SpeedHold], centers: np.ndarray, focus_x: float | None) -> list[int]:
        if self.lane == "all" or not holds:
            return list(range(len(holds)))
        if self.lane in {"left", "right"}:
            median_x = float(np.median(centers[:, 0]))
            is_left = centers[:, 0] < median_x
            wanted_left = self.lane == "left"
            return [index for index, value in enumerate(is_left) if value == wanted_left]
        if focus_x is None:
            return list(range(len(holds)))

        median_x = float(np.median(centers[:, 0]))
        left_indices = [index for index, x in enumerate(centers[:, 0]) if x < median_x]
        right_indices = [index for index, x in enumerate(centers[:, 0]) if x >= median_x]
        if not left_indices or not right_indices:
            return list(range(len(holds)))
        left_center = float(np.mean(centers[left_indices, 0]))
        right_center = float(np.mean(centers[right_indices, 0]))
        return left_indices if abs(focus_x - left_center) < abs(focus_x - right_center) else right_indices

    def annotate(
        self,
        frame: np.ndarray,
        focus_x: float | None = None,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        homography = self._align_to_frame(frame)
        if homography is None:
            return frame

        reference_centers = np.float32([hold.center for hold in self.reference_holds]).reshape(-1, 1, 2)
        centers = cv2.perspectiveTransform(reference_centers, homography).reshape(-1, 2)
        selected_indices = self._select_lane(self.reference_holds, centers, focus_x)
        contact_indices = self._find_contacts(centers, selected_indices, keypoints)
        output = frame.copy()
        height, width = output.shape[:2]
        for index in selected_indices:
            center_x, center_y = centers[index]
            if not (0 <= center_x < width and 0 <= center_y < height):
                continue
            hold = self.reference_holds[index]
            is_contact = index in contact_indices
            color = (0, 0, 255) if is_contact else (0, 220, 0) if hold.kind == "large" else (0, 200, 255)
            center = (int(center_x), int(center_y))
            cv2.circle(output, center, 10 if hold.kind == "large" else 6, color, 2)
            label = f"{index + 1}:{hold.kind[0].upper()}"
            if is_contact:
                label += " CONTACT"
            cv2.putText(
                output,
                label,
                (center[0] + 10, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return output

    def _find_contacts(
        self,
        centers: np.ndarray,
        selected_indices: list[int],
        keypoints: np.ndarray | None,
    ) -> set[int]:
        if keypoints is None or len(keypoints) == 0:
            self.contact_streak.clear()
            self.previous_contact_points = None
            return set()

        contact_points = keypoints[[9, 10, 15, 16], :2]
        contact_confidence = keypoints[[9, 10, 15, 16], 2]
        if self.previous_contact_points is None:
            movement = np.zeros(len(contact_points), dtype=np.float32)
        else:
            movement = np.linalg.norm(contact_points - self.previous_contact_points, axis=1)
        self.previous_contact_points = contact_points.copy()

        contacts: set[int] = set()
        for index in selected_indices:
            distances = np.linalg.norm(contact_points - centers[index], axis=1)
            close_and_stable = (
                np.any((distances < 45) & (contact_confidence > 0.35) & (movement < 20))
            )
            self.contact_streak[index] = self.contact_streak.get(index, 0) + 1 if close_and_stable else 0
            if self.contact_streak[index] >= 3:
                contacts.add(index)
        return contacts


class RoboflowSpeedWallHoldDetector:
    """Use the Roboflow workflow and keep a fixed 20-large/11-small route set."""

    def __init__(
        self,
        lane: str = "auto",
        large_count: int = 20,
        small_count: int = 11,
    ) -> None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is not set")
        if lane not in {"auto", "left", "right", "all"}:
            raise ValueError("lane must be 'auto', 'left', 'right', or 'all'")

        self.lane = lane
        self.large_count = large_count
        self.small_count = small_count
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        ).configure(InferenceConfiguration(api_key_transport="header"))
        self.last_holds: list[SpeedHold] = []
        self.contact_streak: dict[int, int] = {}
        self.previous_contact_points = None

    def _workflow_predictions(self, frame: np.ndarray) -> list[dict]:
        result = self.client.run_workflow(
            workspace_name="simon-alexander",
            workflow_id="climbing-holds-and-volumes-xyr5x-zrceg",
            images={"image": frame},
            use_cache=False,
        )
        if isinstance(result, list) and result:
            result = result[0]
        prediction_block = result.get("predictions", {}) if isinstance(result, dict) else {}
        predictions = prediction_block.get("predictions", [])
        return [
            prediction
            for prediction in predictions
            if str(prediction.get("class_id", prediction.get("class"))) == "0"
        ]

    def _select_route_holds(self, predictions: list[dict]) -> list[SpeedHold]:
        ranked = sorted(
            predictions,
            key=lambda prediction: (
                float(prediction.get("width", 0)) * float(prediction.get("height", 0)),
                float(prediction.get("confidence", 0)),
            ),
            reverse=True,
        )
        selected = ranked[: self.large_count + self.small_count]
        holds = [
            SpeedHold(
                center=(float(prediction["x"]), float(prediction["y"])),
                kind="large" if index < self.large_count else "small",
                area=float(prediction.get("width", 0)) * float(prediction.get("height", 0)),
            )
            for index, prediction in enumerate(selected)
        ]
        return holds

    def _select_lane(self, centers: np.ndarray, focus_x: float | None) -> list[int]:
        if self.lane in {"all", "auto"} and (self.lane == "all" or focus_x is None):
            return list(range(len(centers)))
        median_x = float(np.median(centers[:, 0]))
        left = [index for index, x in enumerate(centers[:, 0]) if x < median_x]
        right = [index for index, x in enumerate(centers[:, 0]) if x >= median_x]
        if self.lane == "left":
            return left
        if self.lane == "right":
            return right
        if not left or not right:
            return list(range(len(centers)))
        return left if abs(focus_x - np.mean(centers[left, 0])) < abs(focus_x - np.mean(centers[right, 0])) else right

    def annotate(
        self,
        frame: np.ndarray,
        focus_x: float | None = None,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        predictions = self._workflow_predictions(frame)
        current_holds = self._select_route_holds(predictions)
        if current_holds:
            self.last_holds = current_holds
        holds = self.last_holds
        if not holds:
            return frame

        centers = np.float32([hold.center for hold in holds])
        selected_indices = self._select_lane(centers, focus_x)
        contact_indices = self._find_contacts(centers, selected_indices, keypoints)
        output = frame.copy()
        height, width = output.shape[:2]
        for index in selected_indices:
            center_x, center_y = centers[index]
            if not (0 <= center_x < width and 0 <= center_y < height):
                continue
            hold = holds[index]
            is_contact = index in contact_indices
            color = (0, 0, 255) if is_contact else (0, 220, 0) if hold.kind == "large" else (0, 200, 255)
            center = (int(center_x), int(center_y))
            cv2.circle(output, center, 10 if hold.kind == "large" else 6, color, 2)
            label = f"{index + 1}:{hold.kind[0].upper()}"
            if is_contact:
                label += " CONTACT"
            cv2.putText(output, label, (center[0] + 10, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return output

    def _find_contacts(
        self,
        centers: np.ndarray,
        selected_indices: list[int],
        keypoints: np.ndarray | None,
    ) -> set[int]:
        if keypoints is None or len(keypoints) == 0:
            self.contact_streak.clear()
            self.previous_contact_points = None
            return set()
        contact_points = keypoints[[9, 10, 15, 16], :2]
        confidence = keypoints[[9, 10, 15, 16], 2]
        movement = np.zeros(len(contact_points), dtype=np.float32) if self.previous_contact_points is None else np.linalg.norm(contact_points - self.previous_contact_points, axis=1)
        self.previous_contact_points = contact_points.copy()
        contacts = set()
        for index in selected_indices:
            distances = np.linalg.norm(contact_points - centers[index], axis=1)
            stable = np.any((distances < 45) & (confidence > 0.35) & (movement < 20))
            self.contact_streak[index] = self.contact_streak.get(index, 0) + 1 if stable else 0
            if self.contact_streak[index] >= 3:
                contacts.add(index)
        return contacts
