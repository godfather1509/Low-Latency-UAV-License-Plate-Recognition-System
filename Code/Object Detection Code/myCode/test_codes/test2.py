import cv2
from ultralytics import YOLO

model = YOLO("../lpr_model/weights/best.pt")

cap = cv2.VideoCapture("../Data/testVideo.mp4")

frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_clean = frame.copy().astype("uint8")

    # --- TEMPORARY DEBUG SAVE ---

    # LP detection
    res = model(frame_clean, conf=0.25)[0]

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("test", frame)
    cv2.imwrite("debug_frame.jpg", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
