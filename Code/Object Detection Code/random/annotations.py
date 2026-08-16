import cv2
import os
import matplotlib as plt
img_dir = "Data/License Plate Recognition/train/images"
label_dir = "Data/License Plate Recognition/train/labels"

for label_file in os.listdir(label_dir):
    img_file = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, label_file)
    
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
            x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
