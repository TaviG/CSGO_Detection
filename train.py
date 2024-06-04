from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='../../dataset2+dataset3/data.yaml', epochs=300, imgsz=640, box=0.04729, cls=0.42352, lr0=0.00452, lrf=0.74841,
                                                                            momentum=0.83102, warmup_epochs=2.69199, warmup_momentum=0.17677, weight_decay=0.00077, batch=8, amp=False)

  