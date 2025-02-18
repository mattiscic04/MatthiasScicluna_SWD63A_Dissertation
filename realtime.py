import cv2
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt") 

# Opens the webcam / iphone camera
cap = cv2.VideoCapture(0)

class_labels = {
    0: "AM4",
    1: "AM5",
    2: "LGA-1150",
    3: "LGA-1700",
    4: "Not Clear"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        class_ids = result.boxes.cls  # Class IDs
        confidences = result.boxes.conf  # Confidence scores

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box) 

            # Check if the detected socket exists in class ids
            label = class_labels.get(int(class_id), f"Unknown ({class_id})")  # Handle unexpected IDs
            label = f"{label} ({conf:.2f})"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Text label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("CPU Slot Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
