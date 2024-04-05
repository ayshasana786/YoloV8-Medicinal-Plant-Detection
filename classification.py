from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data=r'C:\Users\aysha\OneDrive\Desktop\Yolo_v8\splitted_leaves9', epochs=5)

metrics = model.val()