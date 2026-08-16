# this is not part of website
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import multiprocessing as mp
import time
from sort_master.sort import Sort

# Class labels (unchanged)
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# Detection process
def detection_worker(frame_queue, result_queue):
    model = YOLO(r"C:\Users\ayush\OneDrive\Desktop\major project\quadcopter\Code\Live Streaming\quadcopter\camera\Yolo-Weights\yolov8l.pt")
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        results = model(frame, stream=True)
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if classNames[cls] == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
        result_queue.put(detections)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Needed for some platforms like Windows

    frame_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue()

    # Start detection process
    detection_process = mp.Process(
        target=detection_worker, args=(frame_queue, result_queue)
    )
    detection_process.start()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    if not success:
        print("❌ Webcam error")
        exit()

    # Load mask
    mask = cv2.imread(
        r"C:\Users\ayush\OneDrive\Desktop\major project\quadcopter\Code\Live Streaming\quadcopter\camera\sort_master\mask.png"
    )
    if mask is None:
        print("❌ mask.png not found")
        exit()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Optional overlay
    imgGraphics = cv2.imread(
        r"C:\Users\ayush\OneDrive\Desktop\major project\quadcopter\Code\Live Streaming\quadcopter\camera\sort_master\graphics.png",
        cv2.IMREAD_UNCHANGED,
    )

    # Tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    limitsUp = [100, 150, 300, 150]
    limitsDown = [500, 450, 700, 450]
    totalCountUp = []
    totalCountDown = []

    last_detection = np.empty((0, 5))

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRegion = cv2.bitwise_and(img, mask)

        # Add overlay
        if imgGraphics is not None:
            img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

        # Send current frame to detection process if queue is empty
        if frame_queue.empty():
            frame_queue.put(imgRegion.copy())

        # Get latest detection result
        if not result_queue.empty():
            last_detection = result_queue.get()

        resultsTracker = tracker.update(last_detection)

        # Draw count lines
        cv2.line(
            img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5
        )
        cv2.line(
            img,
            (limitsDown[0], limitsDown[1]),
            (limitsDown[2], limitsDown[3]),
            (0, 0, 255),
            5,
        )

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(
                img,
                f" {int(id)}",
                (max(0, x1), max(35, y1)),
                scale=2,
                thickness=3,
                offset=10,
            )
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if (
                limitsUp[0] < cx < limitsUp[2]
                and limitsUp[1] - 15 < cy < limitsUp[1] + 15
            ):
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    cv2.line(
                        img,
                        (limitsUp[0], limitsUp[1]),
                        (limitsUp[2], limitsUp[3]),
                        (0, 255, 0),
                        5,
                    )

            if (
                limitsDown[0] < cx < limitsDown[2]
                and limitsDown[1] - 15 < cy < limitsDown[1] + 15
            ):
                if totalCountDown.count(id) == 0:
                    totalCountDown.append(id)
                    cv2.line(
                        img,
                        (limitsDown[0], limitsDown[1]),
                        (limitsDown[2], limitsDown[3]),
                        (0, 255, 0),
                        5,
                    )

        # Display count
        cv2.putText(
            img,
            str(len(totalCountUp)),
            (929, 345),
            cv2.FONT_HERSHEY_PLAIN,
            5,
            (139, 195, 75),
            7,
        )
        cv2.putText(
            img,
            str(len(totalCountDown)),
            (1191, 345),
            cv2.FONT_HERSHEY_PLAIN,
            5,
            (50, 50, 230),
            7,
        )

        cv2.imshow("People Counter", img)

        # Exit on 'q' or when window is closed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    frame_queue.put(None)  # Signal detection process to exit
    detection_process.join()
    cv2.destroyAllWindows()
