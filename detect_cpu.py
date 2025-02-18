from ultralytics import YOLO

# Loading YOLOv8 model (pre-trained from Roboflow)
model = YOLO("yolov8n.pt") 

# Training the model on my dataset
model.train(
    data="/Users/matthiasscicluna/Downloads/Matthias_Scicluna_Thesis_SWD6-2/data.yaml", #My Path to the dataset file
    epochs=35,
    imgsz=640,
    batch=8,
    device="mps"  # Using Apple Silicon GPU acceleration - M2 Chip
)
