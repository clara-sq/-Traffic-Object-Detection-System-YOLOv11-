# # #视频轨迹
#绘制方框和文字，增加透明度、padding、颜色等参数
# #增加消除轨迹功能：

import torch
import cv2
from ultralytics import YOLO

VIDEO_PATH = 'videos/test_person.mp4'
RESULT_PATH = "output/2-output-removeTrace.mp4"

OBJ_LIST = ['person', 'car', 'bus', 'truck']
DETECTOR_PATH = 'weights/yolo11n.pt'

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


class yoloTracker4(baseTracker):
    def __init__(self, model=None):   #修改：原本的类写死为yolo11.pt, 改为可选参数(video2需要选择)
        super(yoloTracker4, self).__init__()
        self.model = model
        if self.model is None:
            self.init_model()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names


    def track(self, im):
        results = self.model.track(im, tracker="bytetrack.yaml", persist=True, imgsz=self.img_size,
                                   conf=self.conf, iou=self.iou, device=self.device)
        detected_boxes = results[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            class_id = box.cls.int().cpu().item()
            lbl = self.names[class_id]
            if not lbl in OBJ_LIST:
                continue
            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = box.conf.cpu().item()
            track_id = box.id.int().cpu().item()
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))
        return im, pred_boxes

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, tracker_id):
        self.detections.append((xyxy, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_length=50):
    trail_colors = [(255, 0, 255)] * len(trail_points)  # Red color for all trails
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j - 1][0]), int(trail_points[i][j - 1][1])),
                            (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_colors[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def draw_bboxes(im, pred_boxes):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]  # Green, Blue, Red, Yellow
    for box in pred_boxes:
        x1, y1, x2, y2, lbl, _, track_id = box
        class_idx = OBJ_LIST.index(lbl) if lbl in OBJ_LIST else -1
        if class_idx != -1:
            color = colors[class_idx]
        else:
            color = (0, 0, 0)  # Default color if class not in OBJ_LIST
        thickness = 2
        padding = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        # 画目标框
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        # 构建文字
        text = f'{lbl} (ID:{track_id})'
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # 计算颜色亮度以决定文字颜色
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
        # 半透明文字背景框位置
        text_x = int(x1)
        text_y = int(y1) - 5
        box_start = (text_x, text_y - text_size[1] - padding)
        box_end = (text_x + text_size[0] + padding, text_y + padding)
        # 绘制半透明背景
        overlay = im.copy()
        cv2.rectangle(overlay, box_start, box_end, color, -1)
        alpha = 0.6  # 背景透明度
        cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
        # 绘制文字
        text_pos = (text_x + padding // 2, text_y)
        cv2.putText(im, text, text_pos, font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    return im


if __name__ == '__main__':
    # 开启影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    # 读取第一帧，确保 frame 有值
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查影片路径或影片格式")
        cap.release()
        exit()
    # 获取影像高度、宽度
    frame_height, frame_width = frame.shape[:2]
    # 设定输出影片格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(RESULT_PATH, fourcc, 30, (frame_width, frame_height))
    # 重新放回第一帧，确保循环内可以处理完整影片
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Dictionary to store the trail points of each object
    object_trails = {}

    # 记录每个目标未侦测到的连续帧
    lost_frames_counter = {}

    tracker = yoloTracker4()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # 利用 YOLO Tracker 来读取 frame 中的目标框
            myDetections = Detections()  # 移动到循环内
            output_image_frame, list_boxes = tracker.track(frame)
            for item_bbox in list_boxes:
                x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
                myDetections.add((x1, y1, x2, y2), track_id)

            # Add the current object’s position to the trail
            current_ids = []
            for xyxy, track_id in myDetections.detections:
                x1, y1, x2, y2 = xyxy
                center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
                current_ids.append(track_id)  # 收集目前有侦测到的 ID
                if track_id in object_trails:
                    object_trails[track_id].append((center.x, center.y))
                else:
                    object_trails[track_id] = [(center.x, center.y)]

            # Draw the trail for each object
            draw_trail(output_image_frame, list(object_trails.values()))

            # 修改 trail 删除逻辑：加上等待时间
            remove_ids = []
            for track_id in object_trails:
                if track_id not in current_ids:
                    # 记录未侦测帧数，若超过阈值才准备删除
                    lost_frames_counter[track_id] = lost_frames_counter.get(track_id, 0) + 1
                    if lost_frames_counter[track_id] > 20:  # 你可以调整这个阈值（这里是 20 帧）
                        remove_ids.append(track_id)

            for tid in remove_ids:
                object_trails.pop(tid)
                lost_frames_counter.pop(tid)  # 清除对应的计数器

            # 绘制边界框
            output_image_frame = draw_bboxes(output_image_frame, list_boxes)

            # 存储结果
            out.write(output_image_frame)
            # 显示结果
            cv2.imshow("Demo", output_image_frame)
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
