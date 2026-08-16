from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("../lpr_model/weights/best.pt")   # Replace with your trained weights

# Load image
image_path = "../frames_no_plate/frame_00000.jpg"
img = cv2.imread(image_path)

# Run inference
results = model(img)

# Loop through detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

# Show output
cv2.imshow("Detection", img)
cv2.imwrite("output.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
