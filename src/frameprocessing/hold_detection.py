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
