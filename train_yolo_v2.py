from ultralytics import YOLO

def main():
    #model = YOLO("D:/ICR/yolo_test/runs/detect/train22/weights/best.pt")
    model = YOLO("yolo11s-seg.pt")
    #model = YOLO("C:/Users/dyksi/OneDrive/Pulpit/magisterka/yolo_test/runs/segment/train/weights/best.pt")
    #model = YOLO("D:\ICR\yolo_test\trening\runs\segment\train\weights\best.pt")
    model.train(
        #data="datasets/traffic_dataset/data.yaml",
        data=r"C:\Users\dyksi\OneDrive\Pulpit\magisterka\yolo_test\trening\dataset_v11_naw_500\data.yaml",
        epochs=50,
        imgsz=640,
        device=0, # WYBÓR GPU
        patience=20

    )

if __name__ == "__main__":
    main()
