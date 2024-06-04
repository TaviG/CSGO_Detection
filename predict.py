from ultralytics import YOLO
from PIL import Image
import cv2 as cv

model = YOLO('last_nmodel.pt')
screenshot = Image.open('../../Dizertatie_final/frame_004090_PNG.rf.f32f32f4da689a4815da0f5d4fe6c03e.jpg')
# screenshot.show()
results = model(screenshot)
annotated_frame = results[0].plot()
rgb_img = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
im = Image.fromarray(rgb_img)
im.save("img_predict.jpeg")

# cv.imshow("YOLOv8 Inference", annotated_frame)