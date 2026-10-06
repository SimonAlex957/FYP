import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.frameprocessing.hold_detection import SpeedWallHoldDetector

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("speed_climb_bad_angle.jpg")
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("hold_candidates.png")
image = cv2.imread(str(image_path))
detector = SpeedWallHoldDetector(image_path)
for index, hold in enumerate(detector.reference_holds, start=1):
    center = tuple(round(value) for value in hold.center)
    color = (0, 255, 0) if hold.kind == "large" else (0, 200, 255)
    cv2.circle(image, center, 10 if hold.kind == "large" else 6, color, 2)
    cv2.putText(image, str(index), (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
cv2.imwrite(str(output_path), image)
print(f"wrote {output_path} with {len(detector.reference_holds)} candidates")
