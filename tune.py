from ultralytics import YOLO

model = YOLO('yolov8n.pt')
result_grid = model.tune(data='/home/tavi/Dizertatie/dataset2+dataset4/data.yaml', epochs=15, use_ray=True)