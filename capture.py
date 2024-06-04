import numpy as np
import cv2 as cv
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from ultralytics import YOLO
import ctypes
import torch

wincap = WindowCapture("Counter-Strike: Global Offensive - Direct3D 9")
#wincap = WindowCapture("Counter-Strike 2")

#vision = Vision()
model = YOLO('last_nmodel.pt')
# model = YOLO('newdataset_nmodel.pt')
# model = YOLO('../../../.pyenv/runs/detect/train124/weights/best.pt')
side = input("Choose enemy CT/T:  ")
if side == "CT":
   body_class = 0
   head_class = 1
elif side == "T":
   body_class = 2
   head_class = 3
else:
   print("Incorrect input")
   exit()


user32 = ctypes.windll.user32
screensize_width = user32.GetSystemMetrics(0)
screensize_height = user32.GetSystemMetrics(1)
crosshair_X = wincap.w//2
crosshair_Y = wincap.h//2

loop_time = time()
while(True):
    boxes = []
    screenshot = wincap.get_screenshot()
    model_time = time()
    results = model(screenshot)
    print('Model FPS {}'.format(1 / (time() - model_time)))
    annotated_frame = results[0].plot()
    if head_class in results[0].boxes.cls:
       idx = (results[0].boxes.cls == head_class).nonzero(as_tuple=True)
       boxes = results[0].boxes[idx].xywh
    elif body_class in results[0].boxes.cls:
       idx = (results[0].boxes.cls == body_class).nonzero(as_tuple=True)
       boxes = results[0].boxes[idx].xywh

    boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)
    print(boxes)
   #  for box in boxes:
   #   print(torch.round(box[0]).type(torch.int16) - crosshair_X,torch.round(box[1]).type(torch.int16) - crosshair_Y)
   #   wincap.click(torch.round(box[0]).type(torch.int16) - crosshair_X,torch.round(box[1]).type(torch.int16) - crosshair_Y)
    if len(boxes):
      print(torch.round(boxes[0][0]).type(torch.int16) - crosshair_X,torch.round(boxes[0][1]).type(torch.int16) - crosshair_Y)
      wincap.click(torch.round(boxes[0][0]).type(torch.int16) - crosshair_X,torch.round(boxes[0][1]).type(torch.int16) - crosshair_Y)
         
    
#    print(results[0].boxes[idx].cls)
 #   print(results[0].boxes.xywh)
 #   print(results[0].boxes.cls)
    # Display the annotated frame
    cv.imshow("YOLOv8 Inference", annotated_frame)
  #  cv.imshow('Computer Vision', screenshot)
  #  vision.find('', screenshot, threshold=0.5, debug_mode=None)
    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
