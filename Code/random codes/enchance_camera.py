import cv2
import numpy as np

# === Settings ===
width, height = 640, 480
fps = 15
output_filename = "recorded_output.avi"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# === CLAHE setup ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# === Sharpening kernel ===
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to LAB and apply CLAHE to luminance
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Optional: Resize up slightly (adds perception of detail)
    upscaled = cv2.resize(frame_clahe, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Sharpen the upscaled frame
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)

    # Show and write
    cv2.imshow("Enhanced Feed", sharpened)
    out.write(cv2.resize(sharpened, (width, height)))  # Resize back before saving

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
