import cv2
from ultralytics import YOLO

try:
    model = YOLO("weights/yolo11n.pt")
    video_path = "videos/99397548-1-6-720p.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error: 无法读取影片，请检查影片路径或影片格式")

    # 获取影像高度、宽度
    frame_height, frame_width = cap.read()[1].shape[:2]
    out = cv2.VideoWriter("./output/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (frame_width, frame_height))

    # 重新放回第一帧，确保循环内可以处理完整影片
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 开始逐帧处理影片
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 执行 YOLO 追踪
        results = model.track(frame, persist=True)
        if not results:
            print("Warning: 当前帧没有检测到目标")
            continue

        # 标记追踪结果
        annotated_frame = results[0].plot()

        # 存储结果
        out.write(annotated_frame)

        # 显示结果
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"发生错误: {e}")

finally:
    # 释放资源
    out.release()
    cap.release()
    cv2.destroyAllWindows()
