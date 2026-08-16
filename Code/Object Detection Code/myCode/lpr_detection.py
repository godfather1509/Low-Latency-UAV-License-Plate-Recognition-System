import cv2
import numpy as np
import os
from ultralytics import YOLO
from util import get_car, read_license_plate, write_csv
from sort.sort import *

def main_lpr_detection():

    # --- Folders for saving images ---
    detected_folder = "frames_with_plate"
    no_plate_folder = "frames_no_plate"
    os.makedirs(detected_folder, exist_ok=True)
    os.makedirs(no_plate_folder, exist_ok=True)

    # Create subfolders for frames with plates
    images_folder = os.path.join(detected_folder, "images")
    labels_folder = os.path.join(detected_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    results = {}
    car_model = YOLO('car_model/yolov8s.pt')
    license_plate_detector = YOLO('lpr_model/weights/best.pt')
    motion_tracker = Sort()

    # cap = cv2.VideoCapture('Data/testVideo.mp4')
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    frame_no = -1
    ret = True
    vehicle_indexes = [2, 3, 5, 7]

    # Video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Resize for output video
    output_width = 800
    scaling_factor = output_width / original_width
    output_height = int(original_height * scaling_factor)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_detected_video.mp4', fourcc, fps, (output_width, output_height))

    # Display window
    display_window = True
    cv2.namedWindow('Vehicle and License Plate Detection', cv2.WINDOW_NORMAL)

    while ret:
        frame_no += 1
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_no] = {}
        car_detections = car_model(frame)[0]
        detections_ = []

        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_indexes:
                detections_.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {score:.2f}', (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ---- SAFE SORT INPUT ----
        if len(detections_) == 0:
            # SORT expects array shape (N, 5), even if empty
            dets_for_sort = np.empty((0, 5))
        else:
            dets_for_sort = np.asarray(detections_, dtype=float)

        track_ids = motion_tracker.update(dets_for_sort)


        # License plate detection
        license_plate_detection = license_plate_detector(frame)[0]
        plate_detected_in_frame = False

        for license_plate in license_plate_detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, f'LP {score:.2f}', (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
                lp_text, lp_score= read_license_plate(thresh)
                # print(lp_country)
                if lp_text is not None:
                    plate_detected_in_frame = True
                    results[frame_no][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': lp_text,
                            'bbox_score': score,
                            'text_score': lp_score,
                        }
                    }

                    # Draw rectangle and text
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f'{lp_text}', (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Save frame with YOLO-style label
                    img_name = f"frame_{frame_no:05d}.jpg"
                    img_path = os.path.join(images_folder, img_name)
                    cv2.imwrite(img_path, frame)
                    # Save results CSV
                    write_csv(results, 'results.csv')
                    # Compute YOLO format: class_id=0 (plate), x_center, y_center, width, height (normalized)
                    x_center = (x1 + x2) / 2 / original_width
                    y_center = (y1 + y2) / 2 / original_height
                    w = (x2 - x1) / original_width
                    h = (y2 - y1) / original_height

                    label_path = os.path.join(labels_folder, img_name)
                    new_label_path = label_path.replace('.jpg', '.txt')
                    with open(new_label_path, 'w') as f:
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")

        # If no plate detected, save frame in separate folder with label indicating -1
        if not plate_detected_in_frame:
            img_name = f"frame_{frame_no:05d}.jpg"
            img_path = os.path.join(no_plate_folder, img_name)
            cv2.imwrite(img_path, frame)
            
        # Resize frame for output video
        frame_resized = cv2.resize(frame, (output_width, output_height))
        out.write(frame_resized)

        # Optional display
        if display_window:
            if cv2.getWindowProperty('Vehicle and License Plate Detection', cv2.WND_PROP_VISIBLE) < 1:
                print("Display window closed by user.")
                display_window = False
            else:
                cv2.imshow('Vehicle and License Plate Detection', frame_resized)
                cv2.waitKey(1)

   
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved, frames saved with YOLO-style labels, CSV written.")