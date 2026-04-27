from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Run detection
results = model('/home/msi/go2_ws/scripts/robot_frame.jpg', save=True)

# Print results
print("\n--- Detections ---")
for r in results:
    if len(r.boxes) == 0:
        print("No objects detected.")
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  → {cls} (confidence: {conf:.2f})")

# Show image with bounding boxes in a window
annotated = results[0].plot()
cv2.imshow('YOLOv8 Detections', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()