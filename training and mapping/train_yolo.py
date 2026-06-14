from ultralytics import YOLO
import time

def main():
    #model = YOLO("D:/ICR/yolo_test/runs/detect/train22/weights/best.pt")
    model = YOLO('yolo11m-seg.pt')
    start_time = time.time()
    #model = YOLO("C:/Users/dyksi/OneDrive/Pulpit/magisterka/yolo_test/runs/segment/train/weights/best.pt")
    model.train(
        #data="datasets/traffic_dataset/data.yaml",
        data = "datasets/surface_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        device=0, # WYBÓR GPU
        patience=20

    )
    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print(f"\nŁączny czas treningu: {hours}h {minutes}min {seconds:.2f}s")

if __name__ == "__main__":
    main()
