from ultralytics import YOLO


model=YOLO("yolov8n.yaml") # build a new model from scratch

results=model.train(data="myCode/config.yaml",epochs=1) # train the model

# yolo detect train data=config.yaml model="yolov8n.yaml" epochs=1
# this command can train yolo model from command line
