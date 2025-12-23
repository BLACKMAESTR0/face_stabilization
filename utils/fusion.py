def fuse_bboxes(yolo_bbox, lstm_bbox, alpha_yolo, alpha_lstm):

    return (
        alpha_yolo * yolo_bbox +
        alpha_lstm * lstm_bbox
    )