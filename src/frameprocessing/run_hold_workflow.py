import json
import os
from pathlib import Path

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

def run_hold_workflow(image_path: Path, output_path: Path) -> dict:
    """Run the Roboflow workflow and save its raw result."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in this terminal")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(InferenceConfiguration(api_key_transport="header"))

    result = client.run_workflow(
        workspace_name="simon-alexander",
        workflow_id="climbing-holds-and-volumes-xyr5x-zrceg",
        images={"image": str(image_path)},
        use_cache=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_path = (
        project_root
        / "outputs"
        / "RoboflowHoldDetection"
        / "Raw"
        / "roboflow_hold_result.json"
    )
    run_hold_workflow(project_root / "inputs" / "FarBack.jpg", output_path)
    print(f"Saved workflow result to {output_path}")
