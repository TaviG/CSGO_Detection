from ultralytics import YOLO

# Load a model
#model = YOLO('../../../.pyenv/runs/detect/train124/weights/best.pt')  # load a custom model
#model = YOLO('newdataset_nmodel.pt')  # load a custom model
model = YOLO('last_nmodel.pt')  # load a custom model
# Validate the model
metrics = model.val( split = 'test')  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

