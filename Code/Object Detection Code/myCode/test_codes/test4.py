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

    images_folder = os.path.join(detected_folder, "images")
    labels_folder = os.path.join(detected_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    results = {}

    # Only license plate model is needed
    license_plate_detector = YOLO('lpr_model/weights/best.pt')

    # Track only license plates
    motion_tracker = Sort()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_no = -1
    ret = True

    # Video settings
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_width = 800
    scaling_factor = output_width / original_width
    output_height = int(original_height * scaling_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_detected_video.mp4', fourcc, fps, (output_width, output_height))

    cv2.namedWindow('LP Detection', cv2.WINDOW_NORMAL)

    while ret:
        frame_no += 1
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_no] = {}

        # --------------------------
        # 1️⃣ DETECT LICENSE PLATES
        # --------------------------
        lp_detections = license_plate_detector(frame)[0]
        detections_ = []

        for det in lp_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            detections_.append([x1, y1, x2, y2, score])

        # Safe SORT input
        if len(detections_) == 0:
            dets_for_sort = np.empty((0, 5))
        else:
            dets_for_sort = np.asarray(detections_, dtype=float)

        # --------------------------
        # 2️⃣ TRACK LICENSE PLATES
        # --------------------------
        track_ids = motion_tracker.update(dets_for_sort)

        # --------------------------
        # 3️⃣ PROCESS EACH TRACKED PLATE
        # --------------------------
        plate_detected_in_frame = False

        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            plate_detected_in_frame = True

            # crop LP
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

            # OCR
            lp_text, lp_score = read_license_plate(thresh)

            # Draw box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 255), 2)
            cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # If OCR success → show text
            if lp_text:
                cv2.putText(frame, lp_text, (int(x1), int(y2)+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                results[frame_no][int(track_id)] = {
                    'license_plate': {
                        'bounding_box': [x1, y1, x2, y2],
                        'text': lp_text,
                        'text_score': lp_score,
                        'lp_country': lp_country
                    }
                }

                # Save image
                img_name = f"frame_{frame_no:05d}.jpg"
                cv2.imwrite(os.path.join(images_folder, img_name), frame)

                # Save YOLO label
                x_center = (x1 + x2) / 2 / original_width
                y_center = (y1 + y2) / 2 / original_height
                w = (x2 - x1) / original_width
                h = (y2 - y1) / original_height

                with open(os.path.join(labels_folder, img_name.replace('.jpg', '.txt')), 'w') as f:
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")

                write_csv(results, 'results.csv')

        # --------------------------
        # 4️⃣ SAVE NON-LP FRAMES
        # --------------------------
        if not plate_detected_in_frame:
            cv2.imwrite(os.path.join(no_plate_folder, f"frame_{frame_no:05d}.jpg"), frame)

        # Output video
        frame_resized = cv2.resize(frame, (output_width, output_height))
        out.write(frame_resized)
        cv2.imshow("LP Detection", frame_resized)
        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("DONE — LP tracking + OCR complete.")



main_lpr_detection()