import datetime
import os
import sys
import time
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import csv
from ultralytics import YOLO
from ultralytics.engine import predictor
import cv2
from YoloTracker import yoloTracker
from YoloTracker import draw_bboxes
from YoloTracker4 import yoloTracker4, OBJ_LIST
from Count3 import YoloTracker3, OBJ_LIST, draw_trail4
from datetime import datetime
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QGraphicsScene, QTabBar, QDialog, QHBoxLayout, QLabel, \
    QVBoxLayout, QPushButton, QTableWidgetItem, QMessageBox, QGraphicsLineItem, QLineEdit
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QUrl, QTimer, QRectF
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyecharts.charts import HeatMap, Bar3D
from pyecharts import options as opts
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, tracker_id):
        self.detections.append((xyxy, tracker_id))

def draw_trail3(output_image_frame, trail_points, trail_length=50):
    trail_colors = [(255, 0, 255)] * len(trail_points)  # Red color for all trails
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j - 1][0]), int(trail_points[i][j - 1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_colors[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def draw_bboxes4(im, pred_boxes):  #视频
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

def is_in_line(pt1, pt2, pt):
    x1, y1 = pt1.x, pt1.y
    x2, y2 = pt2.x, pt2.y
    x, y = pt.x, pt.y
    return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

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




class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 不要登录
        # 加载qt-designer中设计的ui文件
        self.ui = uic.loadUi("./ui/main_new.ui")
        # 菜单下拉框
        self.actiondefault = self.ui.actiondefault
        self.actionblack = self.ui.actionblack
        self.actionwhite = self.ui.actionwhite
        self.actionblue = self.ui.actionblue
        self.actionintro = self.ui.actionintro
        self.actionversion = self.ui.actionversion
        self.actionexit = self.ui.actionexit
        # 侧边栏
        self.tabWidget = self.ui.tabWidget
        self.tab_image = self.ui.tab_image
        self.tab_video = self.ui.tab_video
        self.tab_track = self.ui.tab_track
        self.tab_count = self.ui.tab_count
        self.test2_btn = self.ui.test2_btn
        self.test3_btn = self.ui.test3_btn

        # tab1_image
        self.raw_img = self.ui.raw_image
        self.res_img = self.ui.res_image
        self.select_btn = self.ui.select_btn
        self.show_btn = self.ui.show_btn
        # tab1_data
        self.model1 = self.ui.combo1
        self.conf1 = self.ui.conf1
        self.conf1.setRange(0.0, 1.0)  # 设置范围
        self.conf1.setSingleStep(0.01)  # 设置步长
        self.conf1.setValue(0.25)  # 设置初始值
        self.IOU1 = self.ui.IOU1
        self.IOU1.setRange(0.0, 1.0)  # 设置范围
        self.IOU1.setSingleStep(0.01)  # 设置步长
        self.IOU1.setValue(0.45)  # 设置初始值
        self.class1 = self.ui.class1
        # tab2_video1
        self.video1 = self.ui.video1
        self.choose_video1 = self.ui.choose_video
        # self.play_pause1 = self.ui.play_pause1
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_frame)
        self.video_path = ""
        # # 创建一个媒体播放器对象和一个视频窗口对象
        # self.media_player1 = QMediaPlayer()
        # # 将视频窗口设置为媒体播放器的显示窗口
        # # self.media_player1.setVideoOutput(self.video1)
        # # 进度条
        # self.media_player1.durationChanged.connect(self.getDuration1)
        # self.media_player1.positionChanged.connect(self.getPosition1)
        self.ui.slider1.sliderMoved.connect(self.updatePosition1)



        # tab2_video2
        self.video2 = self.ui.video2
        self.show_video = self.ui.show_video
        # self.play_pause2 = self.ui.play_pause2
        # self.timer2 = QTimer()
        # self.timer2.timeout.connect(self.showVideo)
        # self.cap2 = None
        # self.out2 = None
        # self.tracker2 = None


        # 创建一个媒体播放器对象和一个视频窗口对象
        # self.media_player2 = QMediaPlayer()
        # 将视频窗口设置为媒体播放器的显示窗口
        # self.media_player2.setVideoOutput(self.video2)
        # 进度条
        # self.media_player2.durationChanged.connect(self.getDuration2)
        # self.media_player2.positionChanged.connect(self.getPosition2)
        self.ui.slider2.sliderMoved.connect(self.updatePosition2)
        # tab2_data
        self.model2 = self.ui.combo2
        self.conf2 = self.ui.conf2
        self.conf2.setRange(0.0, 1.0)  # 设置范围
        self.conf2.setSingleStep(0.01)  # 设置步长
        self.conf2.setValue(0.25)  # 设置初始值
        self.IOU2 = self.ui.IOU2
        self.IOU2.setRange(0.0, 1.0)  # 设置范围
        self.IOU2.setSingleStep(0.01)  # 设置步长
        self.IOU2.setValue(0.45)  # 设置初始值
        self.class2 = self.ui.class2





        # tab3_track
        self.video3 = self.ui.video3
        self.play_pause3 = self.ui.play_pause3
        self.slider3 = self.ui.slider3
        self.time3 = self.ui.time3
        self.choose_btn = self.ui.choose_btn
        self.track_path = self.ui.track_path
        self.track_btn = self.ui.track_btn
        self.export3 = self.ui.export3
        # 创建一个媒体播放器对象和一个视频窗口对象
        # self.media_player3 = QMediaPlayer()
        # 将视频窗口设置为媒体播放器的显示窗口
        # self.media_player3.setVideoOutput(self.video3)
        # 进度条
        # self.media_player3.durationChanged.connect(self.getDuration3)
        # self.media_player3.positionChanged.connect(self.getPosition3)
        self.ui.slider3.sliderMoved.connect(self.updatePosition3)
        # tab3_data
        self.conf3 = self.ui.conf3
        self.conf3.setRange(0.0, 1.0)  # 设置范围
        self.conf3.setSingleStep(0.01)  # 设置步长
        self.conf3.setValue(0.25)  # 设置初始值
        self.IOU3 = self.ui.IOU3
        self.IOU3.setRange(0.0, 1.0)  # 设置范围
        self.IOU3.setSingleStep(0.01)  # 设置步长
        self.IOU3.setValue(0.45)  # 设置初始值
        self.class3 = self.ui.class3
        # tab4_count
        self.video4 = self.ui.video4
        # self.play_pause4 = self.ui.play_pause4
        self.slider4 = self.ui.slider4
        self.time4 = self.ui.time4
        self.choose_btn4 = self.ui.choose_btn4
        self.count_path = self.ui.count_path
        self.count_btn = self.ui.count_btn
        # 创建一个媒体播放器对象和一个视频窗口对象
        # self.media_player4 = QMediaPlayer()
        # 将视频窗口设置为媒体播放器的显示窗口
        # self.media_player4.setVideoOutput(self.video4)
        # 进度条
        # self.media_player4.durationChanged.connect(self.getDuration4)
        # self.media_player4.positionChanged.connect(self.getPosition4)
        self.ui.slider4.sliderMoved.connect(self.updatePosition4)
        # tab4_data
        self.conf4 = self.ui.conf4
        self.conf4.setRange(0.0, 1.0)  # 设置范围
        self.conf4.setSingleStep(0.01)  # 设置步长
        self.conf4.setValue(0.25)  # 设置初始值
        self.IOU4 = self.ui.IOU4
        self.IOU4.setRange(0.0, 1.0)  # 设置范围
        self.IOU4.setSingleStep(0.01)  # 设置步长
        self.IOU4.setValue(0.45)  # 设置初始值
        self.class4 = self.ui.class4

        # tab5_zone
        self.video5 = self.ui.video5
        self.select_btn_5 = self.ui.select_btn_5
        self.zone_btn = self.ui.zone_btn
        self.zone_path = self.ui.zone_path
        #tab5 data
        self.conf5 = self.ui.conf5
        self.conf5.setRange(0.0, 1.0)  # 设置范围
        self.conf5.setSingleStep(0.01)  # 设置步长
        self.conf5.setValue(0.25)  # 设置初始值
        self.IOU5 = self.ui.IOU5
        self.IOU5.setRange(0.0, 1.0)  # 设置范围
        self.IOU5.setSingleStep(0.01)  # 设置步长
        self.IOU5.setValue(0.45)  # 设置初始值


        # tab6_graph
        self.hotmap = self.ui.hotmap
        self.chart6_2 = self.ui.chart6_2
        self.data6_1 = self.ui.data6_1
        self.data6_2 = self.ui.data6_2
        self.data6_3 = self.ui.data6_3
        self.export6_1 = self.ui.export6_1
        self.export6_2 = self.ui.export6_2
        # 侧边栏的click点击事件
        self.tab_image.clicked.connect(self.open1)
        self.tab_video.clicked.connect(self.open2)
        self.tab_track.clicked.connect(self.open3)
        self.tab_count.clicked.connect(self.open4)
        self.test2_btn.clicked.connect(self.open5)
        self.test3_btn.clicked.connect(self.open6)
        # tab1的点击事件
        self.model1.currentIndexChanged.connect(self.combo1_change)
        self.select_btn.clicked.connect(self.select_image)
        self.show_btn.clicked.connect(self.detect_objects)
        # tab2的点击事件
        self.model2.currentIndexChanged.connect(self.combo2_change)
        self.choose_video1.clicked.connect(self.chooseVideo1)
        # self.play_pause1.clicked.connect(self.playPause1)
        # self.play_pause2.clicked. xconnect(self.playPause2)
        self.show_video.clicked.connect(self.showVideo)
        # tab3的点击事件
        self.choose_btn.clicked.connect(self.chooseVideo3)
        self.play_pause3.clicked.connect(self.playPause3)
        self.track_btn.clicked.connect(self.showTrack)
        self.export3.clicked.connect(self.export_location)
        # tab4的点击事件
        self.choose_btn4.clicked.connect(self.chooseVideo4)
        # self.play_pause4.clicked.connect(self.playPause4)
        self.count_btn.clicked.connect(self.showCount)
        # tab5的点击事件
        self.select_btn_5.clicked.connect(self.selectZone)
        self.zone_btn.clicked.connect(self.showZone)
        # self.select_xlsx.clicked.connect(self.selectDataset)
        # self.draw5.clicked.connect(self.drawChart5)
        # self.export5.clicked.connect(self.export_chart5)
        # 隐藏所有的Tab widget页面
        self.tabBar = self.tabWidget.findChild(QTabBar)
        self.tabBar.hide()
        # 默认打开首页
        self.tabWidget.setCurrentIndex(0)
        # 菜单栏点击事件
        self.actionwhite.triggered.connect(self.menu_white)
        self.actionblack.triggered.connect(self.menu_black)
        self.actionblue.triggered.connect(self.menu_blue)
        self.actiondefault.triggered.connect(self.menu_default)
        self.actionintro.triggered.connect(self.menu_intro)
        self.actionversion.triggered.connect(self.menu_version)
        self.actionexit.triggered.connect(self.myexit)
        # 保存图表
        self.export6_1.clicked.connect(self.export_chart6_1)
        self.export6_2.clicked.connect(self.export_chart6_2)

    def menu_default(self):
        print('default')
        stylesheet1 = f"QMainWindow{{background-color: rgb(240,240,240)}}"
        stylesheet2 = f"QWidget{{background-color: rgb(240,240,240)}}"
        stylesheet3 = f"QLabel{{color: rgb(20, 120, 80); font: 26pt'黑体'; font-weight: bold;}}"
        stylesheet4 = f"QLabel{{background-color: rgb(230, 230, 230)}}"
        stylesheet5 = f"QLabel{{color: rgb(100, 100, 100); font-size: 16pt; font-weight: bold;}}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.tab_3.setStyleSheet(stylesheet2)
        self.ui.tab_4.setStyleSheet(stylesheet2)
        self.ui.tab_5.setStyleSheet(stylesheet2)
        self.ui.tab_6.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)
        self.ui.label_38.setStyleSheet(stylesheet4)
        self.ui.time1.setStyleSheet(stylesheet5)
        self.ui.time2.setStyleSheet(stylesheet5)
        self.ui.time3.setStyleSheet(stylesheet5)
        self.ui.time4.setStyleSheet(stylesheet5)

    def menu_white(self):
        print('light')
        stylesheet1 = f"QMainWindow{{background-color: rgb(250,250,250)}}"
        stylesheet2 = f"QWidget{{background-color: rgb(250,250,250)}}"
        stylesheet3 = f"QLabel{{color: rgb(20, 120, 80); font: 26pt'黑体'; font-weight: bold;}}"
        stylesheet4 = f"QLabel{{background-color: rgb(240, 240, 240)}}"
        stylesheet5 = f"QLabel{{color: rgb(100, 100, 100); font-size: 16pt; font-weight: bold;}}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.tab_3.setStyleSheet(stylesheet2)
        self.ui.tab_4.setStyleSheet(stylesheet2)
        self.ui.tab_5.setStyleSheet(stylesheet2)
        self.ui.tab_6.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)
        self.ui.label_38.setStyleSheet(stylesheet4)
        self.ui.time1.setStyleSheet(stylesheet5)
        self.ui.time2.setStyleSheet(stylesheet5)
        self.ui.time3.setStyleSheet(stylesheet5)
        self.ui.time4.setStyleSheet(stylesheet5)

    def menu_black(self):
        print('dark')
        stylesheet1 = f"QMainWindow{{background-color: rgb(50,50,50)}}"
        stylesheet2 = f"QWidget{{background-color: rgb(50,50,50)}}"
        stylesheet3 = f"QLabel{{color: rgb(40, 240, 160); font: 26pt'黑体'; font-weight: bold;}}"
        stylesheet4 = f"QLabel{{background-color: rgb(40, 60, 50)}}"
        stylesheet5 = f"QLabel{{color: rgb(250, 250, 250); font-size: 16pt; font-weight: bold;}}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.tab_3.setStyleSheet(stylesheet2)
        self.ui.tab_4.setStyleSheet(stylesheet2)
        self.ui.tab_5.setStyleSheet(stylesheet2)
        self.ui.tab_6.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)
        self.ui.label_38.setStyleSheet(stylesheet4)
        self.ui.time1.setStyleSheet(stylesheet5)
        self.ui.time2.setStyleSheet(stylesheet5)
        self.ui.time3.setStyleSheet(stylesheet5)
        self.ui.time4.setStyleSheet(stylesheet5)

    def menu_blue(self):
        print('blue')
        stylesheet1 = f"QMainWindow{{background-color: rgb(230,245,255)}}"
        stylesheet2 = f"QWidget{{background-color: rgb(230,245,255)}}"
        stylesheet3 = f"QLabel{{color: rgb(20, 120, 80); font: 26pt'黑体'; font-weight: bold;}}"
        stylesheet4 = f"QLabel{{background-color: rgb(210, 240, 255)}}"
        stylesheet5 = f"QLabel{{color: rgb(100, 100, 100); font-size: 16pt; font-weight: bold;}}"
        self.ui.setStyleSheet(stylesheet1)
        self.ui.centralwidget.setStyleSheet(stylesheet2)
        self.ui.tab.setStyleSheet(stylesheet2)
        self.ui.tab_2.setStyleSheet(stylesheet2)
        self.ui.tab_3.setStyleSheet(stylesheet2)
        self.ui.tab_4.setStyleSheet(stylesheet2)
        self.ui.tab_5.setStyleSheet(stylesheet2)
        self.ui.tab_6.setStyleSheet(stylesheet2)
        self.ui.label_11.setStyleSheet(stylesheet3)
        self.ui.label_38.setStyleSheet(stylesheet4)
        self.ui.time1.setStyleSheet(stylesheet5)
        self.ui.time2.setStyleSheet(stylesheet5)
        self.ui.time3.setStyleSheet(stylesheet5)
        self.ui.time4.setStyleSheet(stylesheet5)

    def menu_intro(self):
        print('intro')
        try:
            dialog = QDialog()
            dialog.setWindowTitle('introduction')
            dialog.setFixedSize(1200, 800)  # 设置对话框大小
            # 总体水平布局
            layout = QHBoxLayout(dialog)
            # 左侧的 QLabel，用于显示图片
            image_label = QLabel()
            pixmap = QPixmap('image/2.png')
            pixmap = pixmap.scaled(400, 350)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
            # 设置标题字体
            font = QFont()
            font.setPointSize(18)  # 设置字体大小
            font.setBold(True)  # 加粗
            # 设置主要文字字体
            font1 = QFont()
            font1.setPointSize(10)  # 设置字体大小
            font1.setBold(True)  # 加粗
            # 创建 QPalette 对象并设置文本颜色
            palette_title = QPalette()
            palette_text = QPalette()
            # 设置为绿色
            palette_title.setColor(QPalette.WindowText, QColor(10, 80, 50))
            palette_text.setColor(QPalette.WindowText, QColor(30, 180, 80))
            # 右侧的 QVBoxLayout，用于显示文字
            text_layout = QVBoxLayout()
            label1 = QLabel("基于YOLOv11的车辆轨迹识")
            label1.setAlignment(Qt.AlignCenter)  # 居中对齐
            label1.setFont(font)
            label1.setPalette(palette_title)
            label2 = QLabel("别与目标检测研究分析")
            label2.setAlignment(Qt.AlignCenter)  # 居中对齐
            label2.setFont(font)
            label2.setPalette(palette_title)
            label3 = QLabel("本软件致力于交通物体追踪检测，")
            label3.setAlignment(Qt.AlignCenter)  # 居中对齐
            label3.setPalette(palette_text)
            label3.setFont(font1)
            label4 = QLabel("通过YOLOv8进行轨迹识别绘制等任务，")
            label4.setAlignment(Qt.AlignCenter)  # 居中对齐
            label4.setPalette(palette_text)
            label4.setFont(font1)
            label5 = QLabel("同时对采集的数据整理成交通数据集，")
            label5.setAlignment(Qt.AlignCenter)  # 居中对齐
            label5.setPalette(palette_text)
            label5.setFont(font1)
            label6 = QLabel("方便后续对数据的处理分析和可视化。")
            label6.setAlignment(Qt.AlignCenter)  # 居中对齐
            label6.setPalette(palette_text)
            label6.setFont(font1)
            text_layout.addSpacing(100)  # 设置间距为100
            text_layout.addWidget(label1)
            text_layout.addWidget(label2)
            text_layout.addSpacing(50)  # 设置间距为50
            text_layout.addWidget(label3)
            text_layout.addWidget(label4)
            text_layout.addWidget(label5)
            text_layout.addWidget(label6)
            text_layout.addSpacing(100)  # 设置间距为100
            # 关闭按钮
            btn = QPushButton('关闭', dialog)
            btn.setFixedSize(150, 60)
            # 连接关闭信号
            btn.clicked.connect(dialog.close)
            text_layout.addWidget(btn, alignment=Qt.AlignCenter)
            layout.addLayout(text_layout)
            # 加载对话框图标
            dialog.setWindowIcon(QIcon("image/2.png"))
            # 显示对话框，而不是一闪而过
            dialog.exec()
        except Exception as e:
            print(e)

    def menu_version(self):
        print('version')
        try:
            dialog = QDialog()
            dialog.setWindowTitle('version')
            dialog.setFixedSize(1200, 800)  # 设置对话框大小
            # 总体水平布局
            layout = QHBoxLayout(dialog)
            # 左侧的 QLabel，用于显示图片
            image_label = QLabel()
            pixmap = QPixmap("image/2.png")
            pixmap = pixmap.scaled(400, 350)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
            # 设置标题字体
            font = QFont()
            font.setPointSize(18)  # 设置字体大小
            font.setBold(True)  # 加粗
            # 设置主要文字字体
            font1 = QFont()
            font1.setPointSize(14)  # 设置字体大小
            font1.setBold(True)  # 加粗
            # 创建 QPalette 对象并设置文本颜色
            palette_title = QPalette()
            palette_text = QPalette()
            # 设置为绿色
            palette_title.setColor(QPalette.WindowText, QColor(10, 80, 50))
            palette_text.setColor(QPalette.WindowText, QColor(30, 180, 80))
            # 右侧的 QVBoxLayout，用于显示文字
            text_layout = QVBoxLayout()
            label1 = QLabel("基于YOLOv11的车辆轨迹识")
            label1.setAlignment(Qt.AlignCenter)  # 居中对齐
            label1.setFont(font)
            label1.setPalette(palette_title)
            label2 = QLabel("别与目标检测研究分析")
            label2.setAlignment(Qt.AlignCenter)  # 居中对齐
            label2.setFont(font)
            label2.setPalette(palette_title)
            label3 = QLabel("版本:  V 1.0")
            label3.setAlignment(Qt.AlignCenter)  # 居中对齐
            label3.setFont(font1)
            label3.setPalette(palette_text)
            label4 = QLabel("时间:  2024年04月11日")
            label4.setAlignment(Qt.AlignCenter)  # 居中对齐
            label4.setFont(font1)
            label4.setPalette(palette_text)
            text_layout.addSpacing(100)  # 设置间距为10
            text_layout.addWidget(label1)
            text_layout.addWidget(label2)
            text_layout.addSpacing(50)  # 设置间距为10
            text_layout.addWidget(label3)
            text_layout.addWidget(label4)
            text_layout.addSpacing(100)  # 设置间距为10
            btn = QPushButton('关闭', dialog)
            btn.setFixedSize(150, 60)
            btn.clicked.connect(dialog.close)
            text_layout.addWidget(btn, alignment=Qt.AlignCenter)
            layout.addLayout(text_layout)
            # 加载对话框图标
            dialog.setWindowIcon(QIcon("image/2.png"))
            # 显示对话框，而不是一闪而过
            dialog.exec()
        except Exception as e:
            print(e)

    # tab
    def open1(self):
        self.tabWidget.setCurrentIndex(0)

    def open2(self):
        self.tabWidget.setCurrentIndex(1)

    def open3(self):
        self.tabWidget.setCurrentIndex(2)

    def open4(self):
        self.tabWidget.setCurrentIndex(3)

    def open5(self):
        self.tabWidget.setCurrentIndex(4)

    def open6(self):
        self.tabWidget.setCurrentIndex(5)
        print('绘制热力图')
        try:
            data = pd.DataFrame(columns=['x', 'y'])
            data['x'] = self.df['X']
            data['y'] = self.df['Y']
            list_x = [0, 300, 600, 900, 1200, 1500, 1800, 2100]
            list_y = [0, 200, 400, 600, 800, 1000, 1200, 1400]
            res = np.zeros((7, 7))
            for index, row in self.df.iterrows():
                # print(row['x'], row['y'])
                for i in range(7):
                    if list_x[i] <= row['X'] and row['X'] <= list_x[i + 1]:
                        for j in range(7):
                            if list_y[j] <= row['Y'] and row['Y'] <= list_y[j + 1]:
                                res[i][j] += 1
            x = ["0", "300", "600", "900", "1200", "1500", "1800"]
            y = ["0", "200", "400", "600", "800", "1000", "1200"]
            data = [(i, j, res[i][j]) for i in range(7) for j in range(7)]
            # data = [[d[1], d[0], d[2]] for d in data]
            heatmap = (
                HeatMap(init_opts=opts.InitOpts(width="650px", height="500px"))
                .add_xaxis(x)
                .add_yaxis("", y, data)
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="热力图",
                                              title_textstyle_opts=opts.TextStyleOpts(font_size=24, padding=20)),
                    visualmap_opts=opts.VisualMapOpts(
                        max_=150,
                        range_color=[
                            "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090",
                            "#fdae61", "#f46d43", "#d73027", "#a50026"
                        ],
                    )
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(font_size=24)
                )
            )
            # 获取图表的HTML内容
            self.hotmap_html = heatmap.render_embed()
            # 将图表的HTML内容加载到QWebEngineView中
            self.hotmap.setHtml(self.hotmap_html)
        except Exception as e:
            print(e)
        print('绘制柱状图')
        try:
            # 绘制chart6_2
            self.figure6_2 = Figure(figsize=(4, 2.5))
            self.myax6 = self.figure6_2.add_subplot(111)
            self.canvas = FigureCanvas(self.figure6_2)
            # 准备数据: x 物体类别  y 对应类别检测到的数量
            # 如果已经track并拿到检测到是词频信息，那就展示这些信息的柱状图
            # 初始值默认都是0
            self.data6_1.setText('0')
            self.data6_2.setText('0')
            self.data6_3.setText('0')
            if hasattr(self, 'category_nums'):
                print("category_nums 存在")
            else:
                print("category_nums 不存在")
            if hasattr(self, 'category_nums'):
                print(self.category_nums)
                # 拿到 keys values 变成列表数据，下面直接用！
                x_data = []
                for i in range(len(self.category_nums)):
                    x_data.append(i)
                # x_data = list(self.category_nums.keys())
                y_data = list(self.category_nums.values())
                # 写出检测到的车辆和行人的数量
                all_objects = 0  # 检测到的所有物体数量
                for key, value in self.category_nums.items():
                    all_objects += value
                    if key == 0:
                        print(key, value)
                        self.data6_2.setText(str(value))
                    if key == 2:
                        print(key, value)
                        self.data6_1.setText(str(value))
                    self.data6_3.setText(str(all_objects))
            # 否则展示默认数据
            else:
                x_data = ['one', 'two', 'three', 'four']
                y_data = [10, 20, 15, 25]
            # 画出词频表
            self.myax6.bar(x_data, y_data, color=['skyblue', 'lightgreen', 'lightcoral', 'lightblue'])
            # 添加标签和标题
            self.myax6.set_xlabel('categories')
            self.myax6.set_ylabel('nums')
            self.myax6.set_title('categoryies’ nums')
            # 绘制柱状图
            self.canvas.draw()
            scene = QGraphicsScene(self)
            scene.addWidget(self.canvas)
            self.chart6_2.setScene(scene)
        except Exception as e:
            print(e)

    # --- tab1 image 点击事件回调函数 ---
    def combo1_change(self, index):
        print("你选择了：" + self.model1.currentText())
        print(index)

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            self.image_path = file_path
            self.load_image()

    def load_image(self):
        try:
            # 创建 QGraphicsScene
            scene = QGraphicsScene()
            # 加载图像
            img = QPixmap(self.image_path)
            # 获取 QGraphicsView 的当前大小
            view_size = self.raw_img.size()
            view_width = view_size.width()
            view_height = view_size.height()
            # 缩放图像以适应 QGraphicsView 的大小，同时保持宽高比
            scaled_img = img.scaled(view_width, view_height, Qt.KeepAspectRatio)
            # 将缩放后的图像添加到场景中
            scene.addPixmap(scaled_img)
            self.raw_img.setScene(scene)
            # 设置 QGraphicsView 的场景矩形以适应缩放后的图像
            # 将QRect 转换为 QRectF
            # rect_f = QRectF(scaled_img.rect())
            # self.raw_img.setSceneRect(rect_f)
            self.res_img.setSceneRect(QRectF(scaled_img.rect()))  #
        except Exception as e:
            print(f"Error occurred while loading and scaling image: {e}")

    def show_image(self):
        # 创建 QGraphicsScene
        scene = QGraphicsScene()
        # 加载图像
        img = QPixmap(self.image_path)
        # 获取 QGraphicsView 的当前大小
        view_size = self.res_img.size()
        view_width = view_size.width()
        view_height = view_size.height()
        # 缩放图像以适应 QGraphicsView 的大小，同时保持宽高比
        scaled_img = img.scaled(view_width, view_height, Qt.KeepAspectRatio)
        # 将缩放后的图像添加到场景中
        scene.addPixmap(scaled_img)
        self.res_img.setScene(scene)
        # 设置 QGraphicsView 的场景矩形以适应缩放后的图像
        # 将QRect 转换为 QRectF
        rect_f = QRectF(scaled_img.rect())
        self.res_img.setSceneRect(rect_f)
        # self.res_img.setSceneRect(QRectF(scaled_img.rect()))


    def detect_objects(self):
        if not self.image_path:
            return
        try:
            #读取参数
            conf1 = float("{:.2f}".format(self.conf1.value()))
            IOU1 = float("{:.2f}".format(self.IOU1.value()))
            class1 = int(self.class1.text()) if self.class1.text() != '' else -1
            print("输入图片路径:", self.image_path)

            # 读取图像（OpenCV 格式）
            image_bgr = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if image_bgr is None:
                print("❌ 图像读取失败")
                return

            # 如果是 PNG 带透明通道（4通道）
            if image_bgr.shape[2] == 4:
                print("检测到透明通道，转换为白底")
                alpha_channel = image_bgr[:, :, 3]
                bgr_channels = image_bgr[:, :, :3]
                white_background = np.ones_like(bgr_channels, dtype=np.uint8) * 255
                alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
                image_bgr = white_background * (1 - alpha_factor) + bgr_channels * alpha_factor
                image_bgr = image_bgr.astype(np.uint8)

            #初始化yolotracker
            tracker = yoloTracker()
            tracker.conf = conf1
            tracker.iou = IOU1

            # 执行追踪并绘制目标框（返回处理后的图像）
            # output_image, _ = tracker.track(image_bgr) #生成的图像没有绘制目标框
            output_image, pred_boxes = tracker.track(image_bgr)   ###修改
            output_image, _ = draw_bboxes(output_image, pred_boxes)

            # 转成 Qt 图像格式显示
            rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qt_image)

            # 显示图像到界面上的 res_img (QGraphicsView)
            scene = QGraphicsScene()
            scaled_pixmap = pixmap.scaled(self.res_img.size(), Qt.KeepAspectRatio)
            scene.addPixmap(scaled_pixmap)
            self.res_img.setScene(scene)
            self.res_img.setSceneRect(QRectF(scaled_pixmap.rect()))
            print("✅ 图像显示完成")

        except Exception as e:
            print(f"检测结果显示失败: {e}")
            QMessageBox.critical(self, "错误", f"检测结果显示失败:\n{e}")



    # --- tab2 video 点击事件回调函数 ---
    def combo2_change(self, index):
        print("你选择了：" + self.model2.currentText())
        print(index)

    # 选择原视频
    #修改：将 PyQt5 + QMediaPlayer 播放方式替换成 OpenCV + QLabel 实时帧播放方式，这样可以完全绕过 QMediaPlayer 的兼容性限制
    def chooseVideo1(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi)')
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Error: 无法打开视频")
                return
            self.timer.start(30)  # 30ms 一帧

    def read_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            print("播放完成")
            self.timer.stop()
            self.cap.release()
            return
        frame = cv2.resize(frame, (700, 500))  # 若需要匹配 UI 大小
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video1.setPixmap(pixmap)


    # tab2 处理并播放结果视频
    def display_cv_frame2(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video2.setPixmap(QPixmap.fromImage(qt_image))
        self.video2.setScaledContents(True)


    def showVideo(self):
        try:
            # 加载 YOLO11 模型
            model = YOLO("weights/yolo11n.pt")

            cap = cv2.VideoCapture(self.video_path)

            # 循环遍历视频帧
            while cap.isOpened():
                # 从视频中读取一帧
                success, frame = cap.read()
                if success:
                    # 在帧上运行 YOLO11 跟踪，跨帧持久化跟踪
                    results = model.track(frame, persist=True)
                    # 在帧上可视化结果
                    annotated_frame = results[0].plot()
                    # 显示画面
                    self.display_cv_frame2(annotated_frame)
                    # 如果按下 'q' 键，则中断循环
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # 如果达到视频末尾，则中断循环
                    break

            # 释放视频捕获对象并关闭显示窗口
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)



    # 播放/暂停
    def playPause1(self):
        if self.media_player1.state() == 1:
            self.media_player1.pause()
        else:
            self.media_player1.play()

    # 视频总时长获取
    def getDuration1(self, d):
        # d是获取到的视频总时长（ms）
        self.ui.slider1.setRange(0, d)
        self.ui.slider1.setEnabled(True)
        self.displayTime1(d)

    # 视频实时位置获取
    def getPosition1(self, p):
        self.ui.slider1.setValue(p)
        self.displayTime1(self.ui.slider1.maximum() - p)

    # 显示剩余时间
    def displayTime1(self, ms):
        minutes = int(ms / 60000)
        seconds = int((ms - minutes * 60000) / 1000)
        self.ui.time1.setText('{}:{}'.format(minutes, seconds))

    # 用进度条更新视频位置
    def updatePosition1(self, v):
        self.media_player1.setPosition(v)
        self.displayTime1(self.ui.slider1.maximum() - v)




    # 播放/暂停
    # def playPause2(self):
    #     if self.media_player2.state() == 1:
    #         self.media_player2.pause()
    #     else:
    #         self.media_player2.play()

    # 视频总时长获取
    def getDuration2(self, d):
        # d是获取到的视频总时长（ms）
        self.ui.slider2.setRange(0, d)
        self.ui.slider2.setEnabled(True)
        self.displayTime2(d)

    # 视频实时位置获取
    def getPosition2(self, p):
        self.ui.slider2.setValue(p)
        self.displayTime2(self.ui.slider2.maximum() - p)

    # 显示剩余时间
    def displayTime2(self, ms):
        minutes = int(ms / 60000)
        seconds = int((ms - minutes * 60000) / 1000)
        self.ui.time2.setText('{}:{}'.format(minutes, seconds))

    # 用进度条更新视频位置
    def updatePosition2(self, v):
        # self.media_player2.setPosition(v)
        self.displayTime2(self.ui.slider2.maximum() - v)

    # -- tab3 track 点击事件回调函数 ---
    def chooseVideo3(self):
        try:
            # 拿到视频路径，存到track_path里，并在textBrowser中展示路径
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, 'Open Video File', '', 'Videos (*.mp4 *.avi)')
            if file_path:
                # self.track_path = file_path
                print("file_path", file_path)
                # 展示路径
                self.track_path.setText(file_path)
                # self.track_path.setText(str(file_path))
        except Exception as e:
            print(e)

    # 播放/暂停
    def playPause3(self):
        if self.media_player3.state() == 1:
            self.media_player3.pause()
        else:
            self.media_player3.play()

    # 视频总时长获取
    def getDuration3(self, d):
        # d是获取到的视频总时长（ms）
        self.ui.slider3.setRange(0, d)
        self.ui.slider3.setEnabled(True)
        self.displayTime3(d)

    # 视频实时位置获取
    def getPosition3(self, p):
        self.ui.slider3.setValue(p)
        self.displayTime3(self.ui.slider3.maximum() - p)

    # 显示剩余时间
    def displayTime3(self, ms):
        minutes = int(ms / 60000)
        seconds = int((ms - minutes * 60000) / 1000)
        self.ui.time3.setText('{}:{}'.format(minutes, seconds))

    # 用进度条更新视频位置
    def updatePosition3(self, v):
        # self.media_player3.setPosition(v)
        self.displayTime3(self.ui.slider3.maximum() - v)

    #实时显示track视频
    def display_cv_frame3(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video3.setPixmap(QPixmap.fromImage(qt_image))
        self.video3.setScaledContents(True)

    # 物体追踪，轨迹识别与绘制
    def showTrack(self):
        self.locations = {}

        if not self.track_path:
            return

        conf3 = self.conf3.value()
        conf3 = float("{:.2f}".format(conf3))
        print("Current value:", conf3)

        IOU3 = self.IOU3.value()
        IOU3 = float("{:.2f}".format(IOU3))
        if self.class3.text() == '':
            class3 = -1
        else:
            class3 = int(self.class3.text())

        try:

            # 拿到当前图片路径末尾的文件名
            file_name = os.path.basename(self.track_path.toPlainText())
            print("file_name", file_name)
            video_path = 'videos/' + file_name
            print("video_path:", video_path)
            RESULT_PATH = "./output/video_output/" + file_name

            cap = cv2.VideoCapture(video_path)
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
                    draw_trail3(output_image_frame, list(object_trails.values()))

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
                    output_image_frame = draw_bboxes4(output_image_frame, list_boxes)

                    # 存储结果
                    out.write(output_image_frame)
                    # 显示结果
                    # cv2.imshow("Demo", output_image_frame)
                    self.display_cv_frame3(output_image_frame)
                    # 按 'q' 退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            time.sleep(1)

        except Exception as e:
            print(e)



    # 导出位置坐标对话框
    def export_location(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Chart Data', '', 'Chart (*.xlsx *.csv)')
        if file_path:
            print(file_path)
            file_name = os.path.splitext(file_path)[0]
            print("文件名为:", file_name)
            file_extension = os.path.splitext(file_path)[1]
            print("文件后缀为:", file_extension)
            # 将数据导出为xlsx
            if file_extension == '.xlsx':
                # 创建一个空的DataFrame
                df = pd.DataFrame(columns=['Track ID', 'X', 'Y', 'Category'])
                # 遍历字典，并将嵌套位置信息展平后添加到DataFrame中
                for track_id, positions in self.locations.items():
                    for position in positions:
                        # 保留小数点后3位
                        X = round(position['x'], 3)
                        Y = round(position['y'], 3)
                        df = df._append({'Track ID': track_id, 'X': X, 'Y': Y, 'Category': position['c']},
                                        ignore_index=True)
                # 将DataFrame写入xlsx文件
                df.to_excel(file_path, index=False)
            # 将数据导出为csv
            else:
                # 将数据写入csv文件
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Track ID', 'X', 'Y', 'Category'])
                    for track_id, positions in self.locations.items():
                        for position in positions:
                            # 保留小数点后3位
                            X = round(position['x'], 3)
                            Y = round(position['y'], 3)
                            writer.writerow([track_id, X, Y, position['c']])
        else:
            print('用户取消了保存图片操作')
            return

    # --- tab4 count 点击事件回调函数 ---
    def chooseVideo4(self):
        try:
            # 拿到视频路径，存到track_path里，并在textBrowser中展示路径
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, 'Open Video File', '', 'Videos (*.mp4 *.avi)')
            if file_path:
                # self.track_path = file_path
                print("file_path", file_path)
                # 展示路径
                self.count_path.setText(file_path)
        except Exception as e:
            print(e)

    # 播放/暂停
    def playPause4(self):
        if self.media_player4.state() == 1:
            self.media_player4.pause()
        else:
            self.media_player4.play()

    # 视频总时长获取
    def getDuration4(self, d):
        # d是获取到的视频总时长（ms）
        self.ui.slider4.setRange(0, d)
        self.ui.slider4.setEnabled(True)
        self.displayTime4(d)

    # 视频实时位置获取
    def getPosition4(self, p):
        self.ui.slider4.setValue(p)
        self.displayTime4(self.ui.slider4.maximum() - p)

    # 显示剩余时间
    def displayTime4(self, ms):
        minutes = int(ms / 60000)
        seconds = int((ms - minutes * 60000) / 1000)
        self.ui.time4.setText('{}:{}'.format(minutes, seconds))

    # 用进度条更新视频位置
    def updatePosition4(self, v):
        # self.media_player4.setPosition(v)
        self.displayTime4(self.ui.slider4.maximum() - v)

    # # 关键步骤 - 计数
    def display_cv_frame4(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video4.setPixmap(QPixmap.fromImage(qt_image))
        self.video4.setScaledContents(True)

    def showCount(self):
        # 先处理，得到结果视频
        if not self.count_path:
            return
        # 拿到用户输入的参数
        conf4 = self.conf4.value()
        conf4 = float("{:.2f}".format(conf4))
        IOU4 = self.IOU4.value()
        IOU4 = float("{:.2f}".format(IOU4))
        if self.class4.text() == '':
            class4 = -1
        else:
            class4 = int(self.class4.text())

        try:
            # 拿到当前图片路径末尾的文件名
            file_name = os.path.basename(self.count_path.toPlainText())
            video_path = 'videos/' + file_name
            RESULT_PATH = "./output/video_output/" + file_name
            offset = 150  # 偏移量，单位为像素，可依需要调整

            cap = cv2.VideoCapture(video_path)
            # 读取第一帧，确保 frame 有值
            success, frame = cap.read()
            if not success:
                print("Error: 无法读取影片，请检查影片路径或影片格式")
                cap.release()
                exit()

            # 获取影片帧率，供后续速度计算
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 获取影像高度、宽度
            frame_height, frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            tracker = YoloTracker3('weights/yolo11n.pt')

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
                draw_trail4(output_image_frame, list(object_trails.values()), trail_colors)

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

                # print("tracker_state：", len(tracker_state.keys()))
                # print("prev_tracker_state：", len(prev_tracker_state.keys()))

                in_count, out_count, prev_tracker_state, tracker_state = trigger(
                    myDetections, pt1, pt2, prev_tracker_state, tracker_state, crossing_ids, in_count, out_count)

                text_draw = 'DOWN: ' + str(out_count) + ' , UP: ' + str(in_count)
                output_image_frame = cv2.putText(img=output_image_frame, text=text_draw, org=(10, 50),
                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                 fontScale=0.75, color=(0, 0, 255), thickness=2)

                out.write(output_image_frame)
                # cv2.imshow('Output', output_image_frame)
                self.display_cv_frame4(output_image_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)



#tab5 Zone
    def display_cv_frame5(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video5.setPixmap(QPixmap.fromImage(qt_image))
        self.video5.setScaledContents(True)

    def selectZone(self):
        try:
            # 拿到视频路径，存到zone_path里，并在textBrowser中展示路径
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, 'Open Video File', '', 'Videos (*.mp4 *.avi)')
            if file_path:
                print("file_path", file_path)
                # 展示路径
                self.zone_path.setText(file_path)  # 如果是QTextBrowser，使用setText
                print(f"self.zone_path: {self.zone_path}")
                print(f"self.zone_path type: {type(self.zone_path)}")
        except Exception as e:
            print(e)

    def showZone(self):
      # 先处理，得到结果视频
        if not self.zone_path:
            print("没有zone_path")
            return
        try:
            print("开始处理视频...")
            # 拿到当前图片路径末尾的文件名
            file_name = os.path.basename(self.zone_path.toPlainText())
            print("file_name:",file_name)
            video_path = 'videos/' + file_name
            print("video_path:", video_path)

            RESULT_PATH = "./output/video_output/" + file_name
            # SCREENSHOT_PATH = "../output/screenshots/"  # 截图保存路径

            cap = cv2.VideoCapture(video_path)
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
                            intrusion_time_dict[track_id] = {"intrusion_frames": 0, "is_in_polygon": is_in_polygon,
                                                             "screenshot_taken": False}
                        else:
                            intrusion_time_dict[track_id]["is_in_polygon"] = is_in_polygon

                        # 如果目标在区域中，增加其入侵累计帧数
                        if is_in_polygon:
                            intrusion_time_dict[track_id]["intrusion_frames"] += 1

                        # 计算入侵时间（秒）
                        intrusion_time_seconds = intrusion_time_dict[track_id]["intrusion_frames"] / fps

                        # 显示入侵时间
                        intrude_text = f'{intrusion_time_seconds:.1f}s'
                        cv2.putText(frame_with_track, intrude_text, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # 绘制检测框为红色
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
                self.display_cv_frame5(frame_with_track)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()


        except Exception as e:
            print(e)
















   # --- tab 6 ---
   # 保存热力图



    def export_chart6_1(self):
        # 保存热力图
        pixmap = QPixmap()
        # 获取图表的截图
        pixmap = self.hotmap.grab()
        # 拿到想要导出的路径
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Chart Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            # 保存为图片
            pixmap.save(file_path)  # 保存图表
            # 提示用户
            try:
                dialog = QMessageBox()
                dialog.setWindowTitle('success')
                dialog.setIcon(QMessageBox.Information)
                dialog.setText('导出成功！')
                dialog.exec_()
            except Exception as e:
                print(e)
        else:
            print('用户取消了保存图片操作')
            return

    # 保存类别词频图
    def export_chart6_2(self):
        # 拿到想要导出的路径
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Chart Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            # 保存为图片
            self.figure6_2.savefig(file_path)
            # 提示用户
            try:
                dialog = QMessageBox()
                dialog.setWindowTitle('success')
                dialog.setIcon(QMessageBox.Information)
                dialog.setText('导出成功！')
                dialog.exec_()
            except Exception as e:
                print(e)
        else:
            print('用户取消了保存图片操作')
            return

    # 退出函数
    def myexit(self):
        exit()


if __name__ == "__main__":
    # 创建应用程序对象
    app = QApplication(sys.argv)
    # 创建自定义的窗口MyWindow对象，初始化相关属性和方法
    win = MyWindow()
    # 显示图形界面
    win.ui.show()
    # 启动应用程序的事件循环，可不断见监听点击事件等
    app.exec()
