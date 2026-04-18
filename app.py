import base64
import io
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


app = Flask(__name__)


class AssistiveVisionEngine:
    """
    Small AI manager that keeps the architecture easy to explain:
    - loads YOLOv8
    - supports dynamic model hot-reload
    - runs inference in worker threads
    - applies a 3-layer logic pipeline
    """

    def __init__(self) -> None:
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model = None
        self.model_name = "Unavailable"
        self.model_source = ""
        self.last_error = ""
        self.default_model_name = "yolov8n.pt"
        self.min_confidence = 0.30
        self.runtime_model_dir = os.path.join(tempfile.gettempdir(), "assistive_vision_models")
        os.makedirs(self.runtime_model_dir, exist_ok=True)
        self._load_startup_model()

    def _load_startup_model(self) -> None:
        """
        Load a default general-purpose YOLOv8 model.

        yolov8n.pt is a compact CNN checkpoint that still recognizes many
        real-world street objects useful for assistive navigation demos.
        """
        if YOLO is None:
            self.last_error = "ultralytics is not installed. Install dependencies first."
            return

        try:
            self.model = YOLO(self.default_model_name)
            self.model_name = self.default_model_name
            self.model_source = self.default_model_name
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"Failed to load default model: {exc}"

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Hot-swap the neural network in memory without restarting Flask.

        This powers the Training Hub. The UI uploads a .pt file, the server
        saves it, and this function replaces the active YOLO model instantly.
        """
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed in this environment.")

        new_model = YOLO(model_path)
        with self.model_lock:
            self.model = new_model
            self.model_name = os.path.basename(model_path)
            self.model_source = model_path
            self.last_error = ""

        return {
            "status": "success",
            "message": "Model reloaded successfully.",
            "model_name": self.model_name,
            "model_source": self.model_source,
        }

    @staticmethod
    def decode_base64_frame(frame_data_url: str) -> np.ndarray:
        """
        Decode a browser-captured JPEG data URL into an RGB array for YOLO.
        """
        if "," in frame_data_url:
            frame_data_url = frame_data_url.split(",", 1)[1]

        image_bytes = base64.b64decode(frame_data_url)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    @staticmethod
    def estimate_distance_meters(box_height: float, frame_height: float) -> float:
        """
        Layer 2: Monocular distance estimation from bounding-box height.

        Nearby objects appear taller in the image. Distant objects appear
        shorter. This heuristic is simple and presentation-friendly.
        """
        if frame_height <= 0:
            return 99.0

        normalized_height = max(box_height / frame_height, 0.01)
        estimated_distance = 1.35 / normalized_height
        return round(min(max(estimated_distance, 0.2), 15.0), 2)

    @staticmethod
    def is_relevant_obstacle(class_name: str) -> bool:
        """
        Keep only classes that matter for street mobility assistance.
        """
        important_classes = {
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "bench",
            "chair",
            "traffic light",
            "stop sign",
            "dog",
            "cat",
            "potted plant",
            "fire hydrant",
            "parking meter",
            "suitcase",
            "dining table",
        }
        return class_name in important_classes

    @staticmethod
    def estimate_direction(x1: float, x2: float, frame_width: float) -> str:
        """
        Estimate horizontal direction from the bounding-box center.

        This supports assistive phrases such as:
        - obstacle on your left
        - obstacle ahead
        - obstacle on your right
        """
        if frame_width <= 0:
            return "ahead"

        center_x = (x1 + x2) / 2.0
        normalized_x = center_x / frame_width

        if normalized_x < 0.36:
            return "left"
        if normalized_x > 0.64:
            return "right"
        return "ahead"

    def _run_inference(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Full 3-layer logic:
        1. Detection with the YOLOv8 CNN.
        2. Distance estimation from box height.
        3. Decision filtering: confidence > 50% and distance < 2 meters.
        """
        started_at = time.perf_counter()

        with self.model_lock:
            active_model = self.model
            current_model_name = self.model_name

        if active_model is None:
            raise RuntimeError(self.last_error or "No YOLO model is loaded.")

        results = active_model.predict(frame, verbose=False)
        inference_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

        frame_height, frame_width = frame.shape[:2]
        detections: List[Dict[str, Any]] = []
        hazards: List[Dict[str, Any]] = []

        if results:
            result = results[0]
            names = result.names
            boxes = getattr(result, "boxes", None)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    class_name = names.get(class_id, str(class_id))

                    box_height = max(y2 - y1, 1.0)
                    estimated_distance = self.estimate_distance_meters(box_height, frame_height)
                    is_relevant = self.is_relevant_obstacle(class_name)
                    direction = self.estimate_direction(x1, x2, frame_width)

                    detection = {
                        "label": class_name,
                        "confidence": round(confidence, 3),
                        "distance_m": estimated_distance,
                        "direction": direction,
                        "bbox": {
                            "x": round(x1, 1),
                            "y": round(y1, 1),
                            "w": round(x2 - x1, 1),
                            "h": round(y2 - y1, 1),
                        },
                        "is_hazard": confidence >= self.min_confidence and estimated_distance < 3.5,
                        "priority_hazard": is_relevant and confidence >= self.min_confidence and estimated_distance < 2.0,
                    }

                    # Show broader detections on the overlay so demos remain
                    # visually informative even when the scene contains objects
                    # outside the smaller mobility-priority class list.
                    if confidence >= self.min_confidence:
                        detections.append(detection)

                    if detection["is_hazard"]:
                        hazards.append(detection)

        hazards.sort(
            key=lambda item: (
                0 if item.get("priority_hazard") else 1,
                item["distance_m"],
            )
        )
        return {
            "detections": detections,
            "hazards": hazards,
            "fps": round(1000 / max(inference_latency_ms, 1), 2),
            "latency_ms": inference_latency_ms,
            "model_name": current_model_name,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    def infer_async(self, frame: np.ndarray):
        """
        Run CNN inference in a background worker so the web app stays responsive.
        """
        return self.executor.submit(self._run_inference, frame)


engine = AssistiveVisionEngine()


@app.route("/")
def home():
    return send_file(os.path.join(os.getcwd(), "index.html"))


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(
        {
            "ready": engine.model is not None,
            "model_name": engine.model_name,
            "model_source": engine.model_source,
            "last_error": engine.last_error,
        }
    )


@app.route("/api/load-model", methods=["POST"])
def load_model():
    """
    Web browsers do not expose raw local file paths directly for security.
    So the Training Hub uploads the .pt file, the backend saves it, and then
    the model is reloaded from that saved server-side path.
    """
    uploaded_file = request.files.get("model")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"status": "error", "message": "No model file selected."}), 400

    filename = secure_filename(uploaded_file.filename)
    if not filename.lower().endswith(".pt"):
        return jsonify({"status": "error", "message": "Only .pt files are allowed."}), 400

    saved_path = os.path.join(engine.runtime_model_dir, filename)
    uploaded_file.save(saved_path)

    try:
        result = engine.load_model(saved_path)
        result["saved_path"] = saved_path
        return jsonify(result)
    except Exception as exc:
        engine.last_error = str(exc)
        return jsonify({"status": "error", "message": f"Model reload failed: {exc}"}), 500


@app.route("/api/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame")

    if not frame_data:
        return jsonify({"status": "error", "message": "Frame data is missing."}), 400

    try:
        frame = engine.decode_base64_frame(frame_data)
        future = engine.infer_async(frame)
        result = future.result(timeout=20)
        result["status"] = "success"
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
