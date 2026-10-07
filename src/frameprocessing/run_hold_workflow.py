import json
import os
from pathlib import Path

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

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
    images={"image": "inputs/FarBack.jpg"},
    use_cache=False,
)

output_path = Path("roboflow_hold_result.json")
output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
print(f"Saved workflow result to {output_path}")
