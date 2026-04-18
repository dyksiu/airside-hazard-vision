import csv


def save_results_csv(
    save_path,
    surface_backend,
    line_backend,
    surface_model,
    line_model,
    unet_num_classes,
    unet_names,
    line_unet_num_classes,
    line_unet_names,
    confidences_copy,
    detections_copy,
    current_fps=0.0,
    avg_fps=0.0,
    event_log_copy=None,
):
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(["# LEGENDA KLAS (format: typ - ID = Nazwa)"])

        if surface_backend == "YOLO":
            if surface_model is not None:
                for cls_id, name in surface_model.names.items():
                    writer.writerow([f"# surface: {cls_id} = {name}"])
        else:
            writer.writerow(["# surface: 0 = tlo"])
            for cls_id in range(1, unet_num_classes):
                writer.writerow([f"# surface: {cls_id} = {unet_names.get(cls_id, str(cls_id))}"])

        if line_backend == "YOLO":
            if line_model is not None:
                for cls_id, name in line_model.names.items():
                    writer.writerow([f"# line: {cls_id} = {name}"])
        else:
            writer.writerow(["# line: 0 = tlo"])
            for cls_id in range(1, line_unet_num_classes):
                writer.writerow([f"# line: {cls_id} = {line_unet_names.get(cls_id, str(cls_id))}"])

        writer.writerow([])
        writer.writerow(["STATYSTYKI FPS"])
        writer.writerow(["Metryka", "Wartosc"])
        writer.writerow(["Aktualne FPS", round(current_fps, 4)])
        writer.writerow(["srednie FPS", round(avg_fps, 4)])

        writer.writerow([])
        writer.writerow(["PODSUMOWANIE"])
        writer.writerow(["Typ modelu", "Klasa", "Średni confidence", "Liczba wykryć"])

        for key, confs in confidences_copy.items():
            model_type, class_name = key.split(":", 1)
            avg_conf = sum(confs) / len(confs)
            writer.writerow([model_type, class_name, round(avg_conf, 4), len(confs)])

        writer.writerow([])
        writer.writerow(["WSZYSTKIE WYKRYCIA"])
        writer.writerow(["Numer klatki", "Typ modelu", "Klasa", "Confidence"])

        for detection in detections_copy:
            writer.writerow(detection)

        if event_log_copy:
            writer.writerow([])
            writer.writerow(["LOG ZDARZEŃ"])
            writer.writerow(["Czas", "Typ zdarzenia", "Opis"])
            for entry in event_log_copy:
                writer.writerow(list(entry))