import sys
import cv2
import time
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("USB2.0 PC CAMERA Feed")

        # Video capture settings
        self.width, self.height = 640, 480
        self.fps = 15  # target FPS

        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # VideoWriter for recording
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter("recorded_output.avi", fourcc, self.fps, (self.width, self.height))

        # QLabel to display video frames
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # Timer for capturing frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.fps))

        self.is_fullscreen = False

        # --- Metrics ---
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()

        # Open CSV file for logging
        self.csv_file = open("metrics_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "CapturedFrames", "ExpectedFrames", "FPS", "Latency_ms", "FrameDrop_%"])

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.close()
            return

        self.frame_count += 1
        now = time.time()

        # --- Latency (ms between frames) ---
        latency_ms = (now - self.last_frame_time) * 1000
        self.last_frame_time = now

        # --- Every 5 seconds, calculate FPS & Frame drop ---
        elapsed = now - self.start_time
        if elapsed >= 5:
            actual_fps = self.frame_count / elapsed
            expected_frames = self.fps * elapsed
            drop_rate = 100 * (expected_frames - self.frame_count) / expected_frames

            print(f"[INFO] Captured: {self.frame_count}, Expected: {expected_frames:.1f}, "
                  f"Avg FPS: {actual_fps:.2f}, Frame Drop: {drop_rate:.2f}%")

            # log to CSV
            self.csv_writer.writerow([
                time.strftime("%H:%M:%S"),
                self.frame_count,
                f"{expected_frames:.1f}",
                f"{actual_fps:.2f}",
                f"{latency_ms:.2f}",
                f"{drop_rate:.2f}"
            ])
            self.csv_file.flush()

            # reset counters
            self.frame_count = 0
            self.start_time = now

        # print latency each frame
        print(f"[DEBUG] Frame latency: {latency_ms:.2f} ms")

        # --- Save frame to video ---
        self.out.write(frame)

        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit window
        window_size = self.video_label.size()
        frame_height, frame_width, _ = rgb_frame.shape
        scale_w = window_size.width() / frame_width
        scale_h = window_size.height() / frame_height
        scale = min(scale_w, scale_h)
        new_w = int(frame_width * scale)
        new_h = int(frame_height * scale)
        resized_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to QImage and set
        qimg = QImage(resized_frame.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_F:
            if self.is_fullscreen:
                self.showNormal()
                self.is_fullscreen = False
            else:
                self.showFullScreen()
                self.is_fullscreen = True

    def closeEvent(self, event):
        self.cap.release()
        self.out.release()
        self.csv_file.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.resize(800, 600)  # Initial window size
    window.show()
    sys.exit(app.exec_())
