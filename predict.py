from ultralytics import YOLO

model = YOLO(r'runs\classify\train\weights\best.onnx',task='classify')

#prediction

p1=model.predict(r"C:\Users\aysha\OneDrive\Desktop\Yolo_v8\splitted_leaves\val\Alpinia Galanga (Rasna)\AG-S-011.jpg",save=True, imgsz=224, conf=0.5)