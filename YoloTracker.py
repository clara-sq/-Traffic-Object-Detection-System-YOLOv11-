#修改文字的颜色，增加透明度，增加文字背景框；预测图片
import torch
import cv2
from ultralytics import YOLO

OBJ_LIST = ['person', 'car', 'bus', 'truck']
DETECTOR_PATH = r'E:\PROJECT\pycharmProj\ultralytics-main\ultralytics-main\weights\yolo11n.pt'

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

class yoloTracker(baseTracker):
    def __init__(self):
        super(yoloTracker, self).__init__()
        self.init_model()  # 显式调用模型初始化方法

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
            if lbl not in OBJ_LIST:  # 确保只处理目标列表中的类别
                continue
            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = box.conf.cpu().item()
            track_id = box.id.int().cpu().item()
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence, track_id))
        return im, pred_boxes

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

    return im,pred_boxes


if __name__ == '__main__':
    myTracker = yoloTracker()
    img_bgr = cv2.imread(r'E:\PROJECT\pycharmProj\ultralytics-main\ultralytics-main\ultralytics\assets\bus.jpg')
    if img_bgr is None:
        print("Error: Unable to load image.")
    else:
        img_bgr, pred_boxes = myTracker.track(img_bgr)
        print("Detected boxes:", pred_boxes)  # 打印检测结果
        img_bgr = draw_bboxes(img_bgr, pred_boxes)  # 绘制边界框和文字
        cv2.imshow('Tracked image', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


