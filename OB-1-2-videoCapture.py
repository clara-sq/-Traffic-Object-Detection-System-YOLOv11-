#识别视频：目标框，人和车


import cv2
from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("weights/yolo11n.pt")

# 打开视频文件
video_path = "videos/99397548-1-6-720p.mp4"
# video_path = "videos/people-2-720p.mp4"
cap = cv2.VideoCapture(video_path)


# 循环遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()
    if success:
        # 在帧上运行 YOLO11 跟踪，跨帧持久化跟踪
        results = model.track(frame, persist=True)
        # 在帧上可视化结果
        annotated_frame = results[0].plot()
        # 显示带注释的帧
        cv2.imshow("YOLO11 跟踪", annotated_frame)
        # 如果按下 'q' 键，则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果达到视频末尾，则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
