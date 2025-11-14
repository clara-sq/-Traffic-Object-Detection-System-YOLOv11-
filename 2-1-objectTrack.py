import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- 配置部分 ----------------
OBJ_LIST = ['person', 'car', 'bus', 'truck']
DETECTOR_PATH = 'weights/yolo11n.pt'
# VIDEO_PATH = 'videos/test_person.mp4'
VIDEO_PATH = 'videos/test_person.mp4'
RESULT_PATH = "output/2-output-1-test.mp4"

# ---------------- 工具类定义 ----------------
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

# ---------------- Tracker 基类 ----------------
class baseTracker(object):
    def __init__(self):
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.70

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")

# ---------------- 绘图函数 ----------------
def draw_bboxes(im, pred_boxes):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]  # Green, Blue, Red, Yellow
    magenta_color = (255, 0, 255)
    for box in pred_boxes:
        x1, y1, x2, y2, lbl, _, track_id = box
        class_idx = OBJ_LIST.index(lbl) if lbl in OBJ_LIST else -1
        color = colors[class_idx] if class_idx != -1 else (0, 0, 0)
        thickness = 2
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text = f'{lbl} (ID:{track_id})'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(im, (int(x1), int(y1 - text_size[1] - 5)),
                      (int(x1 + text_size[0]), int(y1 - 5)), color, -1)
        cv2.putText(im, text, (int(x1), int(y1 - 5)), font, font_scale,
                    magenta_color, font_thickness, lineType=cv2.LINE_AA)
    return im

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for track_id in trail_points:
        points = trail_points[track_id]
        color = trail_color.get(track_id, (0, 0, 255))
        for i in range(1, len(points)):
            cv2.line(output_image_frame,
                     (int(points[i - 1][0]), int(points[i - 1][1])),
                     (int(points[i][0]), int(points[i][1])),
                     color, thickness=2)
        if len(points) > trail_length:
            trail_points[track_id] = points[-trail_length:]

# ---------------- YOLO + Tracker 类 ----------------
class yoloTracker(baseTracker):
    def __init__(self):
        super(yoloTracker, self).__init__()
        self.init_model()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.names = self.model.names

    def track(self, im):
        res = self.model.track(im, tracker="bytetrack.yaml", persist=True,
                               imgsz=self.img_size, conf=self.conf,
                               iou=self.iou, device=self.device)
        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            confidence = box.conf.cpu().item()
            class_id = int(box.cls.cpu().item())
            lbl = self.names[class_id]
            if lbl not in OBJ_LIST:
                continue
            track_id = int(box.id.cpu().item()) if box.id is not None else -1
            pred_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], lbl, confidence, track_id))
        im = draw_bboxes(im, pred_boxes)
        return im, pred_boxes

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查路径")
        cap.release()
        exit()

    frame_height, frame_width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(RESULT_PATH, fourcc, 30, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = yoloTracker()
    trail_points = {}  # {track_id: [(x,y), (x,y), ...]}
    trail_color = {}   # {track_id: color}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        output_frame, boxes = tracker.track(frame)

        for box in boxes:
            x1, y1, x2, y2, lbl, conf, track_id = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if track_id not in trail_points:
                trail_points[track_id] = []
                trail_color[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            trail_points[track_id].append((cx, cy))

        draw_trail(output_frame, trail_points, trail_color)
        out.write(output_frame)
        cv2.imshow("Tracking", output_frame)

        key = cv2.waitKey(1)
        if key == 13:  # 13 是 Enter 键
            print("用户中止处理，已保存处理结果。")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
