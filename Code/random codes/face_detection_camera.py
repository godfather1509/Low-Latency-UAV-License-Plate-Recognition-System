import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("USB2.0 PC CAMERA Feed with Face Detection")

        # Face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Video capture settings
        self.width, self.height = 640, 480
        self.fps = 15

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

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            self.close()
            return

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write frame to file
        self.out.write(frame)

        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get window size and resize frame to fit, preserving aspect ratio
        window_size = self.video_label.size()
        frame_height, frame_width, _ = rgb_frame.shape

        scale_w = window_size.width() / frame_width
        scale_h = window_size.height() / frame_height
        scale = min(scale_w, scale_h)

        new_w = int(frame_width * scale)
        new_h = int(frame_height * scale)

        resized_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to QImage
        qimg = QImage(resized_frame.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)

        # Set pixmap
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_F:
            # Toggle fullscreen
            if self.is_fullscreen:
                self.showNormal()
                self.is_fullscreen = False
            else:
                self.showFullScreen()
                self.is_fullscreen = True

    def closeEvent(self, event):
        self.cap.release()
        self.out.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.resize(800, 600)  # Start window size
    window.show()
    sys.exit(app.exec_())
