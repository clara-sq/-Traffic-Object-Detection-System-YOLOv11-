#识别视频：车的目标框
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("weights/yolo11n.pt")
# 加载 YOLO 模型
# model = YOLO("yolo11s.pt")  # 使用預訓練模型

# 只保留這些類別 (COCO 資料集索引)
TARGET_CLASSES = [2, 5, 7]
video_path = "videos/99397548-1-6-720p.mp4"
cap = cv2.VideoCapture(video_path)
# 先读取第一帧，确保 `frame` 有值
success, frame = cap.read()
if not success:
    print("Error: 无法读取影片，请检查影片路径或影片格式")
    cap.release()
    exit()
# 初始化影片写入
frame_height, frame_width = frame.shape[:2]  # 获取影像高度、宽度
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (frame_width, frame_height))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MP4V"), 30, (frame_width, frame_height))
# 重新放回第一帧，确保循环内可以处理完整影片
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# 开始逐帧处理影片
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 执行 YOLO 追踪
        results = model.track(frame, persist=True)
        print(results[0])
        # 初始化 Annotator
        annotator = Annotator(frame, line_width=1)
        # 标记追踪结果
        # annotated_frame = results[0].plot()
        # 如果有偵測到物件，則手動繪製邊界框
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()  # 边界框坐标
            track_ids = results[0].boxes.id.int().cpu().tolist()  # 追踪 ID
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # 对象类别索引

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id in TARGET_CLASSES:  # car, bus, truck
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[class_id]} ID {track_id}"  # 显示类别名称
                    annotator.box_label((x1, y1, x2, y2), label, color=colors(track_id, True))

        # 取得手動標註後的影像
        annotated_frame = annotator.result()
        # 存储结果
        out.write(annotated_frame)
        # 显示结果
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
# 释放资源
out.release()
cap.release()
cv2.destroyAllWindows()
