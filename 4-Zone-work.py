#入侵后目标框变红色
import cv2
import numpy as np
from YoloTracker4 import yoloTracker4
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 视频路径和输出路径
VIDEO_PATH = 'videos/test_person.mp4'
RESULT_PATH = "output/4-Zone-output-time&print.mp4"
SCREENSHOT_PATH = "../output/screenshots/"  # 截图保存路径

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def drawAndFillPolygon(image, polygonPoints, fillColor):
    """
    绘制并填充多边形区域
    :param image: 输入图像
    :param polygonPoints: 多边形的顶点
    :param fillColor: 填充颜色
    :return: 填充后的图像
    """
    pts = np.array([(p.x, p.y) for p in polygonPoints], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [pts], color=fillColor)
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=5)
    overlaidImage = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return overlaidImage

def isInsidePolygon(point, polygonPoints):
    """
    判断点是否在多边形内
    :param point: 点的坐标
    :param polygonPoints: 多边形的顶点
    :return: 点是否在多边形内
    """
    pts = np.array([(p.x, p.y) for p in polygonPoints], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, (point.x, point.y), False)
    return result >= 0

if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, frame = cap.read()
    if not success:
        print("Error: 无法读取影片，请检查影片路径或影片格式")
        cap.release()
        exit()

    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = yoloTracker4()

    # 定义多边形区域
    polygonPoints = [
        Point(100, 200),
        Point(500, 200),
        Point(800, 400),
        Point(400, 400)
    ]

    # 用于存储每个 track_id 的入侵累计帧数和是否当前在区域中
    intrusion_time_dict = {}
    frame_count = 0  # 初始化帧数计数器

    # 确保截图保存路径存在
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1  # 每帧增加帧数计数器

        # 目标检测与跟踪
        frame_with_track, list_boxes = tracker.track(frame)

        # 绘制多边形区域
        frame_with_track = drawAndFillPolygon(frame_with_track, polygonPoints, fillColor=(255, 0, 0))

        if list_boxes:
            for bbox in list_boxes:
                x1, y1, x2, y2, class_label, confidence, track_id = bbox

                # 计算脚部中心点
                foot_center = Point(x=(x1 + x2) / 2, y=y2)

                # 判断是否在多边形内
                is_in_polygon = isInsidePolygon(foot_center, polygonPoints)

                # 初始化或更新 intrusion_time_dict
                if track_id not in intrusion_time_dict:
                    intrusion_time_dict[track_id] = {"intrusion_frames": 0, "is_in_polygon": is_in_polygon, "screenshot_taken": False}
                else:
                    intrusion_time_dict[track_id]["is_in_polygon"] = is_in_polygon

                # 首次进入区域，截图并保存
                if is_in_polygon and not intrusion_time_dict[track_id]["screenshot_taken"]:
                    screenshot = frame[int(y1):int(y2), int(x1):int(x2)] #截取目标区域
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #获取当前时间戳
                    screenshot_filename = f"{track_id}_{timestamp}.jpg"  # 保存文件名
                    screenshot_path = os.path.join(SCREENSHOT_PATH, screenshot_filename)  # 保存路径
                    cv2.imwrite(screenshot_path, screenshot)   # 保存截图
                    intrusion_time_dict[track_id]["screenshot_taken"] = True  # 标记已保存

                # 如果目标在区域中，增加其入侵累计帧数
                if is_in_polygon:
                    intrusion_time_dict[track_id]["intrusion_frames"] += 1

                # 计算入侵时间（秒）
                intrusion_time_seconds = intrusion_time_dict[track_id]["intrusion_frames"] / fps

                # 显示入侵时间
                intrude_text = f'{intrusion_time_seconds:.1f}s'
                cv2.putText(frame_with_track, intrude_text, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                #绘制检测框为红色
                box_color = (0, 255, 0)  # 默认绿色
                if is_in_polygon:
                    box_color = (0, 0, 255)  # 如果在区域内，变为红色

                # 绘制检测框和ID
                cv2.rectangle(frame_with_track, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.putText(frame_with_track, f'ID:{track_id}', (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        # 处理已经离开区域的目标
        for track_id in list(intrusion_time_dict.keys()):
            if not any(bbox[6] == track_id for bbox in list_boxes):
                intrusion_time_dict[track_id]["is_in_polygon"] = False

        out.write(frame_with_track)
        cv2.imshow("Demo", frame_with_track)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()