from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()

print(torch.cuda.is_available())