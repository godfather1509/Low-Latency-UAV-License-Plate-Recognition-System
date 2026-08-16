import cv2
import numpy as np
import os
from ultralytics import YOLO
from util import read_license_plate, write_csv
from sort.sort import Sort
from multiprocessing import Process, Queue
import time

def get_car_relaxed(plate_box, tracks, iou_threshold=0.05):
    px1, py1, px2, py2, score, cls = plate_box

    best_iou = 0.0
    best_car_id = -1
    best_bbox = (-1, -1, -1, -1)

    if tracks is None or len(tracks) == 0:
        return -1, -1, -1, -1, -1

    for t in tracks:
        # ensure we can unpack whether it's a numpy array row or list
        try:
            x1, y1, x2, y2, car_id = t
        except Exception:
            # skip unexpected row
            continue

        # compute intersection between plate and this car bbox
        ix1 = max(px1, x1)
        iy1 = max(py1, y1)
        ix2 = min(px2, x2)
        iy2 = min(py2, y2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        plate_area = max((px2 - px1) * (py2 - py1), 1e-6)  # avoid div-by-zero
        iou = inter_area / plate_area

        if iou > best_iou:
            best_iou = iou
            try:
                best_car_id = int(car_id)
            except:
                best_car_id = car_id
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    if best_iou >= iou_threshold:
        return (*best_bbox, best_car_id)

    return -1, -1, -1, -1, -1


def clamp_bbox_to_frame(x1, y1, x2, y2, w, h):
    # ensure coords are ints and within image bounds
    x1c = max(0, min(int(x1), w - 1))
    y1c = max(0, min(int(y1), h - 1))
    x2c = max(0, min(int(x2), w - 1))
    y2c = max(0, min(int(y2), h - 1))
    # ensure x2 > x1 and y2 > y1
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c

def display_process(frame_queue):
    window_name = "Vehicle and License Plate Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_open=True
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # stop signal
        if cv2.getWindowProperty("Vehicle and License Plate Detection", cv2.WND_PROP_VISIBLE) < 1:
            window_open=False
            cv2.destroyAllWindows()
            break
        if window_open:
            try:
                cv2.imshow("Vehicle and License Plate Detection", frame)
                key = cv2.waitKey(1)
            except cv2.error:
                window_open = False   # stop showing frames
    cv2.destroyAllWindows()

def main_start():
    frame_queue = Queue(maxsize=20)  # small queue to keep memory low
    p1 = Process(target=display_process, args=(frame_queue,))
    p2= Process(target=main_lpr_detection, args=(frame_queue,))

    p2.start()
    p1.start()

    p1.join()
    p2.join()

def main_lpr_detection(frame_queue):

    # Output folders
    detected_folder = "frames_with_plate"
    no_plate_folder = "frames_no_plate"
    os.makedirs(detected_folder, exist_ok=True)
    os.makedirs(no_plate_folder, exist_ok=True)

    images_folder = os.path.join(detected_folder, "images")
    labels_folder = os.path.join(detected_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    results = {}

    # Models
    car_model = YOLO('car_model/yolov8s.pt')
    plate_model = YOLO('lpr_model/weights/best.pt')

    motion_tracker = Sort()

    # Load video (camera index 1). Change to filename if needed.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open camera 1. Try changing index or using a file path.")
        return

    frame_no = -1

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_detected_video.mp4", fourcc, fps, (orig_w, orig_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to read frame.")
            break

        frame_no += 1
        results.setdefault(frame_no, {})
        plate_found = False

        # Car detection (safe inference)
        try:
            car_preds = car_model(frame)[0]
        except Exception as e:
            print(f"car model inference failed on frame {frame_no}: {e}")
            car_preds = None

        detections = []
        if car_preds is not None:
            boxes = getattr(car_preds, "boxes", None)
            if boxes is not None:
                box_list = boxes.data.tolist() if hasattr(boxes, "data") else []
                for det in box_list:
                    if len(det) < 6:
                        continue
                    x1, y1, x2, y2, score, cls = det
                    if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        detections.append([x1, y1, x2, y2, score])
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Update SORT tracker (expects Nx5 array)
        det_array = np.asarray(detections) if len(detections) > 0 else np.empty((0, 5))
        try:
            tracks = motion_tracker.update(det_array)
        except Exception as e:
            print(f"SORT tracker update failed: {e}")
            tracks = []

        # Plate detection
        try:
            plate_preds = plate_model(frame)[0]
        except Exception as e:
            print(f"plate model inference failed on frame {frame_no}: {e}")
            plate_preds = None

        plate_boxes = []
        if plate_preds is not None and hasattr(plate_preds, "boxes"):
            plate_boxes = plate_preds.boxes.data.tolist() if hasattr(plate_preds.boxes, "data") else []

        for plate in plate_boxes:
            if len(plate) < 6:
                continue
            x1, y1, x2, y2, pscore, cls = plate

            # draw yellow box (pre-match)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            # relaxed association with cars
            cx1, cy1, cx2, cy2, car_id = get_car_relaxed(plate, tracks)

            # clamp crop to frame and check size
            clipped = clamp_bbox_to_frame(x1, y1, x2, y2, orig_w, orig_h)
            if clipped is None:
                # invalid crop, skip
                continue
            sx1, sy1, sx2, sy2 = clipped
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10:
                # too small or invalid
                continue

            # basic preprocessing for LPR - keep as grayscale or appropriate for your read_license_plate
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
            lp_text, lp_score = read_license_plate(thresh)
            print("RED BOX:", sx1, sy1, sx2, sy2)
            if lp_text:
                plate_found = True

                # final red bbox and text
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                cv2.putText(frame, lp_text, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Save result if matched with car
                if car_id != -1:
                    # ensure keys are strings or ints consistently; use int(car_id)
                    try:
                        car_key = int(car_id)
                    except:
                        car_key = car_id

                    results[frame_no][car_key] = {
                        "car": {"bounding_box_car": [int(cx1), int(cy1), int(cx2), int(cy2)]},
                        "license_plate": {
                            "bounding_box": [int(sx1), int(sy1), int(sx2), int(sy2)],
                            "bounding_box_score": float(pscore),
                            "text": lp_text,
                            "text_score": float(lp_score) if lp_score is not None else None
                        }
                    }

                # Save image + label (original behavior)
                img_name = f"frame_{frame_no:05d}.jpg"
                img_path = os.path.join(images_folder, img_name)
                cv2.imwrite(img_path, frame)

                # YOLO YOLO-format txt label (normalize relative to orig image size)
                x_center = ((sx1 + sx2) / 2.0) / orig_w
                y_center = ((sy1 + sy2) / 2.0) / orig_h
                w = (sx2 - sx1) / orig_w
                h = (sy2 - sy1) / orig_h
                label_path = os.path.join(labels_folder, img_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    # class 0 reserved for plate (your choice)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # write CSV once per frame (if you prefer once at end, move this)
        try:
            write_csv(results, "results.csv")
        except Exception as e:
            print(f"write_csv failed: {e}")
        out.write(frame)
        # Send frame to display process if queue is not full
        if not frame_queue.full():
            frame_queue.put(frame.copy())
    cap.release()
    out.release()

if __name__ == "__main__":
    main_start()