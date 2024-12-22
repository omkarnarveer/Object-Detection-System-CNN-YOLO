from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from yolo_inference import detect_objects  # Ensure yolo_inference.py is in the same folder

app = Flask(__name__)

# Configure upload and output directories
UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Perform object detection
        detections, output_image_path = detect_objects(file_path, app.config["OUTPUT_FOLDER"])
        return jsonify({
            "message": f"Detected objects: {', '.join([d['label'] for d in detections])}",
            "image_path": f"/{output_image_path.replace(os.sep, '/')}"
        })

if __name__ == "__main__":
    app.run(debug=True)