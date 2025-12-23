import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda/cpu

YOLO_MODEL = "models/weights/face_yolov8n.pt"
LSTM_WEIGHTS = "models/weights/best_val_model_0.000093.pth"

SEQ_LEN = 20        # the number of bboxes be processing
PRED_LEN = 10        # the number of bboxes be predicted

ALPHA_YOLO = 0.5    # YOLO's weight
ALPHA_LSTM = 0.5    # LSTM's weight
