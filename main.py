import cv2
import torch
import numpy as np

from config import *
from models.lstm_model import Seq2SeqLSTMPredictor
from models.lstm_inference import inference_lstm
from yolo_model.yolo_detector import YoloFaceDetector
from utils.bbox_buffer import BBoxBuffer
from utils.fusion import fuse_bboxes

device = DEVICE

yolo = YoloFaceDetector(YOLO_MODEL, device)
lstm = torch.load(LSTM_WEIGHTS, weights_only=False)
lstm.to(device)
lstm.eval()

buffer = BBoxBuffer(SEQ_LEN)
cap = cv2.VideoCapture(0)

alpha_yolo = 0.5
alpha_lstm = 0.5
seq_len = SEQ_LEN
pred_len = PRED_LEN

ALPHA_STEP = 0.05
SEQ_STEP = 1
PRED_STEP = 1


def update_buffer_size(new_size):
    """change buffer size"""
    global buffer, seq_len
    seq_len = max(1, min(30, new_size))
    buffer = BBoxBuffer(seq_len)
    return seq_len


def print_controls():
    """show help menu in console"""
    controls = """
    ========== CONTROL ==========
    Q/W - change ALPHA_YOLO (↓/↑)
    A/S - change ALPHA_LSTM (↓/↑)
    Z/X - change SEQ_LEN (↓/↑)
    C/V - change PRED_LEN (↓/↑)
    SPACE - normalize weights (0.5/0.5)
    H - show this table
    ESC - exit
    ================================
    """
    print(controls)


def clamp_alpha(value):
    return max(0.0, min(1.0, value))


def normalize_alphas():
    """normalize alpha's"""
    global alpha_yolo, alpha_lstm
    total = alpha_yolo + alpha_lstm
    if total > 0:
        alpha_yolo /= total
        alpha_lstm /= total
    else:
        alpha_yolo = 0.5
        alpha_lstm = 0.5


def draw_info_panel(frame, alpha_yolo, alpha_lstm, seq_len, pred_len):
    """draw panel of information"""
    h, w = frame.shape[:2]

    # Полупрозрачный фон для текста
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Текст информации
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 255, 0)
    y_offset = 30

    info_lines = [
        f"ALPHA_YOLO: {alpha_yolo:.2f} (Q/W)",
        f"ALPHA_LSTM: {alpha_lstm:.2f} (A/S)",
        f"SEQ_LEN: {seq_len} (Z/X)",
        f"PRED_LEN: {pred_len} (C/V)",
    ]

    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (20, y_offset + i * 25), font, font_scale, color, thickness)

    # Подсказка внизу
    cv2.putText(frame, "Press H for help | ESC to exit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame



print_controls()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    yolo_bbox = yolo.detect(frame)

    if yolo_bbox is not None:
        buffer.add(yolo_bbox)

        if buffer.ready():
            seq = buffer.tensor()


            scale = torch.tensor([w_frame, h_frame, w_frame, h_frame], dtype=torch.float32)
            seq_normalized = torch.tensor(seq, dtype=torch.float32) / scale
            seq_normalized = seq_normalized.unsqueeze(1)


            with torch.no_grad():
                preds_normalized = inference_lstm(
                    lstm, seq_normalized, device, pred_len
                )


            preds_normalized = preds_normalized.squeeze(1)
            lstm_bbox_normalized = preds_normalized[-1].detach().cpu().numpy()
            lstm_bbox = lstm_bbox_normalized * np.array([w_frame, h_frame, w_frame, h_frame])

            final_bbox = fuse_bboxes(
                yolo_bbox, lstm_bbox,
                alpha_yolo, alpha_lstm
            )

            x, y, w, h = final_bbox.astype(int)

            cv2.rectangle(
                frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                (0, 255, 0), 2
            )

    # Добавить информационную панель
    frame = draw_info_panel(frame, alpha_yolo, alpha_lstm, seq_len, pred_len)

    cv2.imshow("Face Stabilization", frame)


    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("Exit...")
        break

    # ALPHA_YOLO
    elif key == ord('q'):  # Q - decrease YOLO's alpha
        alpha_yolo = clamp_alpha(alpha_yolo - ALPHA_STEP)
        alpha_lstm = clamp_alpha(alpha_lstm + ALPHA_STEP)
        print(f"ALPHA_YOLO: {alpha_yolo:.3f}")

    elif key == ord('w'):  # W - increase YOLO's alpha
        alpha_yolo = clamp_alpha(alpha_yolo + ALPHA_STEP)
        alpha_lstm = clamp_alpha(alpha_lstm - ALPHA_STEP)
        print(f"ALPHA_YOLO: {alpha_yolo:.3f}")

    # ALPHA_LSTM
    elif key == ord('a'):  # A - decrease LSTM's alpha
        alpha_lstm = clamp_alpha(alpha_lstm - ALPHA_STEP)
        alpha_yolo = clamp_alpha(alpha_yolo + ALPHA_STEP)
        print(f"ALPHA_LSTM: {alpha_lstm:.3f}")

    elif key == ord('s'):  # S - increase LSTM's alpha
        alpha_lstm = clamp_alpha(alpha_lstm + ALPHA_STEP)
        alpha_yolo = clamp_alpha(alpha_yolo - ALPHA_STEP)
        print(f"ALPHA_LSTM: {alpha_lstm:.3f}")

    # SEQ_LEN
    elif key == ord('z'):  # Z - decrease SEQ_LEN
        seq_len = update_buffer_size(seq_len - SEQ_STEP)
        print(f"SEQ_LEN: {seq_len}")

    elif key == ord('x'):  # X - increase SEQ_LEN
        seq_len = update_buffer_size(seq_len + SEQ_STEP)
        print(f"SEQ_LEN: {seq_len}")

    # PRED_LEN
    elif key == ord('c'):  # C - decrease PRED_LEN
        pred_len = max(1, pred_len - PRED_STEP)
        print(f"PRED_LEN: {pred_len}")

    elif key == ord('v'):  # V - increase PRED_LEN
        pred_len = min(30, pred_len + PRED_STEP)
        print(f"PRED_LEN: {pred_len}")

    # Нормализовать веса
    elif key == ord(' '):  # SPACE
        normalize_alphas()
        print(f"Weights have been normalized: YOLO={alpha_yolo:.3f}, LSTM={alpha_lstm:.3f}")

    # Справка
    elif key == ord('h'):  # H
        print_controls()

cap.release()
cv2.destroyAllWindows()