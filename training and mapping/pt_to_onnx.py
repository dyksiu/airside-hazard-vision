from ultralytics import YOLO

model = YOLO(r"C:\Users\dyksi\OneDrive\Pulpit\eksport_do_rpi\best_yolov11n_linie.pt")
model.export(format="onnx", imgsz=512, opset=11)