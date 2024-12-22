from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure the YOLO model file is in the correct location

def detect_objects(image_path, output_path):
    # Perform object detection
    results = model(image_path)

    # Extract the image with detections (plot) from the results
    detection_image = results[0].plot()

    # Save the detection image to the output path
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR))

    # Extract detected object labels and confidences
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 2)
            })

    return detections, output_image_path
