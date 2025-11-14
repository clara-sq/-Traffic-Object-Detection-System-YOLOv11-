#计数：方框+线偏移+速度

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO

OBJ_LIST = ['person', 'car', 'bus', 'truck']

# 定义 Point 类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 定义 Detections 类
class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, tracker_id):
        self.detections.append((xyxy, tracker_id))


# 定义 draw_trail 函数
def draw_trail4(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame,
                         (int(trail_points[i][j - 1][0]), int(trail_points[i][j - 1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])),
                         trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)
    return trail_points


# 定义 is_in_line 函数
def is_in_line(pt1, pt2, pt):
    x1, y1 = pt1.x, pt1.y
    x2, y2 = pt2.x, pt2.y
    x, y = pt.x, pt.y
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


# 定义 trigger 函数
def trigger(detections: Detections, pt1, pt2, prev_tracker_state, tracker_state, crossing_ids, in_count, out_count):
    for xyxy, tracker_id in detections.detections:
        x1, y1, x2, y2 = xyxy
        center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
        tracker_state_new = is_in_line(pt1, pt2, center) >= 0  # Use the center to determine the side

        # Handle new detection
        if tracker_id not in tracker_state or tracker_state[tracker_id] is None:
            tracker_state[tracker_id] = {'state': tracker_state_new, 'direction': None}
            if tracker_id in prev_tracker_state and prev_tracker_state[tracker_id] is not None:
                # If the object was previously tracked and has a known direction,
                # we restore its direction.
                tracker_state[tracker_id]['direction'] = prev_tracker_state[tracker_id]['direction']

        # Handle detection on the same side of the line
        elif tracker_state[tracker_id]['state'] == tracker_state_new:
            continue

        # If the object has completely crossed the line
        else:
            if tracker_state[tracker_id]['state'] and not tracker_state_new:  # From up to down
                if tracker_state[tracker_id]['direction'] != 'down':
                    # Only count if the previous direction was not 'down'
                    in_count += 1  # Increment in_count for crossing from up to down
                tracker_state[tracker_id]['direction'] = 'down'
            elif not tracker_state[tracker_id]['state'] and tracker_state_new:  # From down to up
                if tracker_state[tracker_id]['direction'] != 'up':  # Only count if the previous direction was not 'up'
                    out_count += 1  # Increment out_count for crossing from down to up
                tracker_state[tracker_id]['direction'] = 'up'

            tracker_state[tracker_id]['state'] = tracker_state_new  # Update the tracker state

    # 更新已经消失的检测对象状态
    for tracker_id in list(tracker_state.keys()):
        if tracker_id not in [item[1] for item in detections.detections]:
            prev_tracker_state[tracker_id] = tracker_state[tracker_id]  # Save the state of the disappeared object
            tracker_state[tracker_id] = None

    return in_count, out_count, prev_tracker_state, tracker_state


def draw_bboxes3(im, pred_boxes):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    for box in pred_boxes:
        x1, y1, x2, y2, lbl, _, track_id = box
        class_idx = OBJ_LIST.index(lbl) if lbl in OBJ_LIST else -1
        if class_idx != -1:
            color = colors[class_idx]
        else:
            color = (0, 0, 0)
        thickness = 1
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text = f'{lbl} (ID:{track_id})'
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        padding = 5
        text_x = int(x1)
        text_y = int(y1) - 5
        box_start = (text_x, text_y - text_size[1] - 2 * padding)
        box_end = (text_x + text_size[0] + 2 * padding, text_y)
        overlay = im.copy()
        cv2.rectangle(overlay, box_start, box_end, color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
        text_pos = (text_x + padding, text_y - padding)
        cv2.putText(im, text, text_pos,
                    font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
    return im


class YoloTracker3:
    def __init__(self, weights, device='cpu'):
        self.weights = weights
        self.device = device
        self.model = YOLO(self.weights)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def track(self, im, imgsz=640, conf=0.25, iou=0.70):
        results = self.model.track(im, tracker="bytetrack.yaml", persist=True, imgsz=imgsz, conf=conf, iou=iou,
                                   device=self.device)
        detected_boxes = results[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            class_id = box.cls.int().cpu().item()
            lbl = self.names[class_id]
            if lbl not in OBJ_LIST:
                continue
            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = box.conf.cpu().item()
            track_id = box.id.int().cpu().item()
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))
        im = draw_bboxes3(im, pred_boxes)
        return im, pred_boxes


if __name__ == '__main__':
    VIDEO_PATH = 'videos/test_traffic.mp4'
    RESULT_PATH = "output/3-output.mp4"

    DETECTOR_PATH = 'weights/yolo11n.pt'
    offset = 150  # 偏移量，单位为像素，可依需要调整

    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查影片路径或影片格式")
        cap.release()
        exit()

    # 获取影像高度、宽度
    frame_height, frame_width = frame.shape[:2]

    # 获取影片帧率，供后续速度计算
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = YoloTracker3(DETECTOR_PATH)

    object_trails = {}
    lost_frames_counter = {}
    in_count = 0
    out_count = 0
    prev_tracker_state = {}
    tracker_state = {}
    crossing_ids = set()
    pt1 = Point(0, frame_height // 2 - offset)  # Line starting point
    pt2 = Point(frame_width, frame_height // 2 - offset)  # Line ending point

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        output_image_frame, list_boxes = tracker.track(frame)
        cv2.line(output_image_frame, (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 0, 255), thickness=2)

        myDetections = Detections()
        current_ids = []

        for item_bbox in list_boxes:
            x1, y1, x2, y2, class_label, confidence, track_id = item_bbox
            myDetections.add((x1, y1, x2, y2), track_id)
            current_ids.append(track_id)

            center = Point(x=(x1 + x2) / 2, y=(y1 + y2) / 2)
            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

            lost_frames_counter[track_id] = 0

        # Draw the trail for each object
        trail_colors = [(255, 0, 255)] * len(object_trails)  # Red color for all trails
        draw_trail(output_image_frame, list(object_trails.values()), trail_colors)

        # 显示速度（单位：像素/秒）
        for track_id in object_trails:
            trail = object_trails[track_id]
            if len(trail) >= 2:
                x1, y1 = trail[-2]
                x2, y2 = trail[-1]
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx ** 2 + dy ** 2)
                speed = distance * fps  # 像素/秒

                # 在目标中心点显示速度文字
                center = (int(x2), int(y2))
                speed_text = f"{speed:.1f} px/s"
                cv2.putText(output_image_frame, speed_text, center,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0, 255, 0), thickness=1)

        # Remove disappeared objects
        remove_ids = []
        for track_id in object_trails:
            if track_id not in current_ids:
                lost_frames_counter[track_id] = lost_frames_counter.get(track_id, 0) + 1
                if lost_frames_counter[track_id] > 20:
                    remove_ids.append(track_id)

        for tid in remove_ids:
            object_trails.pop(tid)
            lost_frames_counter.pop(tid)  # 清除对应的计数器

        print("tracker_state：", len(tracker_state.keys()))
        print("prev_tracker_state：", len(prev_tracker_state.keys()))

        in_count, out_count, prev_tracker_state, tracker_state = trigger(
            myDetections, pt1, pt2, prev_tracker_state, tracker_state, crossing_ids, in_count, out_count)

        text_draw = 'DOWN: ' + str(out_count) + ' , UP: ' + str(in_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw, org=(10, 50),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)

        out.write(output_image_frame)
        cv2.imshow('Output', output_image_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()