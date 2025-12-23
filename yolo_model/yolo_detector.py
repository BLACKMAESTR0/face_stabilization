from ultralytics import YOLO

class YoloFaceDetector:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        if len(results[0].boxes) == 0:
            return None

        box = results[0].boxes.xywh[0].cpu().numpy()
        return box