#!/usr/bin/env python3

#功能：实时宠物姿态检测 + Web 推流（狗/猫）

import os
import cv2
import numpy as np
import time
import signal
import sys
from flask import Flask, Response, jsonify
from hobot_dnn import pyeasy_dnn as dnn
from scipy.special import softmax

# ==================== 配置参数 ====================
class Config:
    # 模型配置
    MODEL_PATH = 'yolo11n_pose_bayese_640x640_nv12.bin'
    CONF_THRESH = 0.25  # 降低阈值提高检测率
    NMS_THRESH = 0.45
    
    # 摄像头配置
    CAMERA_INDEX = 1
    BACKUP_CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS_TARGET = 30
    
    # 服务配置
    HOST = '0.0.0.0'
    PORT = 5000
    
    # 显示配置
    SHOW_FPS = True
    KEYPOINT_CONF_THRESH = 0.3  # 关键点阈值（主阈值）
    KEYPOINT_CONF_MIN = 0.25   # 降级模式阈值
    SKELETON_COLOR = (255, 255, 0)  # 骨架颜色 (BGR)
    BBOX_COLOR = (0, 255, 0)  # 边框颜色
    KEYPOINT_COLOR = (0, 0, 255)  # 关键点颜色
    HEAD_POINT_COLOR = (0, 0, 255)   # 鼻/眼（红色）
    ANCHOR_POINT_COLOR = (255, 0, 0) # 肩/髋（蓝色）
    TEXT_COLOR = (255, 255, 0)  # 文本颜色
    TEXT_BG_COLOR = (0, 0, 0)  # 文本背景色
    
    # 宠物类别过滤
    PET_CLASSES = [16, 17]  # dog(16), cat(17)
    SHOW_ALL_CLASSES = False   # False=仅宠物, True=所有类别
    
    # 调试配置
    DEBUG_MODE = False
    SAVE_DEBUG_IMAGE = False

# ==================== COCO 类别映射 ====================
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# ==================== ✅ PET SKELETON FIX: 科学宠物骨架定义 ====================
# 所有宠物使用 17 点结构（兼容 Ultralytics 输出），但连接关系不同
DOG_SKELETON = [
    # 头部：鼻→眼→耳（牵引绳遮挡时可用头部中心补偿）
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 前肢：肩→肘→腕（关键运动链）
    (5, 7), (7, 9), (6, 8), (8, 10),
    # 后肢：髋→膝→踝
    (11, 13), (13, 15), (12, 14), (14, 16),
    # 躯干连接
    (5, 11), (6, 12), (5, 6), (11, 12)
]

CAT_SKELETON = DOG_SKELETON  # 猫结构相似
HORSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12), (5, 6)  # 马无明显腰椎连接
]

# 全局骨架映射表
PET_SKELETON_MAP = {
    16: DOG_SKELETON,   # dog
    17: CAT_SKELETON,   # cat
    18: HORSE_SKELETON, # horse
}

# 头部关键点ID（用于高亮和补偿）
HEAD_POINTS = [0, 1, 2]      # 鼻、左眼、右眼
ANCHOR_POINTS = [5, 6, 11, 12]  # 肩、髋（锚点）

# ==================== 姿态检测器类 ====================
class PetPoseDetector:
    def __init__(self, model_path, conf_thres=0.5, nms_thres=0.45):
        """初始化检测器"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")
        
        print(f"📦 正在加载模型: {model_path}")
        self.models = dnn.load(model_path)
        self.quantize_model = self.models[0]
        self.input_h, self.input_w = self.quantize_model.inputs[0].properties.shape[2:4]
        
        # 算法参数
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.reg = 16 
        self.strides = [8, 16, 32]
        self.conf_raw = -np.log(1 / conf_thres - 1)
        
        # 预生成 Anchor Grid
        self.grids = []
        for s in self.strides:
            gh, gw = self.input_h // s, self.input_w // s
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            self.grids.append(np.stack((xv, yv), axis=-1).reshape(-1, 2) + 0.5)
        
        self.dfl_weights = np.arange(self.reg).reshape(1, 1, self.reg)
        
        print(f"✅ 模型加载成功: 输入尺寸 {self.input_w}x{self.input_h}")

    def bgr2nv12_fast(self, bgr):
        """针对 RDK BPU 优化的 NV12 转换"""
        height, width = bgr.shape[:2]
        yuv420p = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        y = yuv420p[:height, :]
        uv_planar = yuv420p[height:, :].reshape(2, -1)
        uv_packed = uv_planar.transpose(1, 0).reshape(-1)
        return np.concatenate([y.reshape(-1), uv_packed])

    def preprocess(self, frame):
        """图像预处理：缩放 + 填充 + NV12 转换"""
        h, w = frame.shape[:2]
        r = min(self.input_h / h, self.input_w / w)
        new_w, new_h = int(w * r), int(h * r)
        
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        tx, ty = (self.input_w - new_w) // 2, (self.input_h - new_h) // 2
        canvas[ty:ty+new_h, tx:tx+new_w] = resized
        
        nv12_data = self.bgr2nv12_fast(canvas)
        return nv12_data, (tx, ty, r)

    def postprocess(self, outputs, pad_info):
        """
        核心后处理：解析 Ultralytics YOLO11n-Pose 输出
        输出顺序：Cls(0), Box(1), Kpt(2)
        """
        tx, ty, r = pad_info
        
        # 1. 提取所有输出并反量化
        raw_outputs = []
        for i, o in enumerate(outputs):
            props = self.quantize_model.outputs[i].properties
            s = props.scale_data[0] if hasattr(props, 'scale_data') and len(props.scale_data) > 0 else 1.0
            raw_outputs.append(np.frombuffer(o.buffer, dtype=np.int8).astype(np.float32) * s)
        
        all_boxes, all_scores, all_classes, all_kpts = [], [], [], []
        
        # 2. 遍历 3 个尺度
        for i in range(3):
            base_idx = i * 3
            if base_idx + 2 >= len(raw_outputs): 
                continue
            
            cls_raw = raw_outputs[base_idx]      # 类别置信度
            box_raw = raw_outputs[base_idx + 1]  # 边框
            kpt_raw = raw_outputs[base_idx + 2]  # 关键点
            
            stride = self.strides[i]
            grid = self.grids[i]
            num_grid = grid.shape[0]
            
            # 解析类别
            num_classes = cls_raw.size // num_grid
            cls_probs = cls_raw.reshape(num_grid, num_classes)
            cls_ids = np.argmax(cls_probs, axis=1)
            max_scores = np.max(cls_probs, axis=1)
            
            # 过滤低置信度
            mask = max_scores > self.conf_raw
            if not np.any(mask): 
                continue
            
            s_cls = 1 / (1 + np.exp(-max_scores[mask]))
            s_cls_ids = cls_ids[mask]
            s_grid = grid[mask]
            
            # 解析边框
            box_unit = box_raw.size // num_grid
            if box_unit == 64:  # DFL
                s_box_raw = box_raw.reshape(num_grid, 4, 16)[mask]
                s_box_softmax = softmax(s_box_raw, axis=2)
                dist = np.sum(s_box_softmax * self.dfl_weights, axis=2)
            else:  # 直接坐标
                dist = box_raw.reshape(num_grid, 4)[mask]
            
            x1y1 = (s_grid - dist[:, :2]) * stride
            x2y2 = (s_grid + dist[:, 2:]) * stride
            res_box = np.stack([x1y1[:,0], x1y1[:,1], x2y2[:,0], x2y2[:,1]], axis=1)
            
            # 解析关键点
            s_kpt = kpt_raw.reshape(num_grid, 17, 3)[mask].copy()
            s_kpt[:, :, :2] = (s_kpt[:, :, :2] * 2.0 + (s_grid[:, np.newaxis, :] - 0.5)) * stride
            
            all_boxes.append(res_box)
            all_scores.append(s_cls)
            all_classes.append(s_cls_ids)
            all_kpts.append(s_kpt)

        if not all_boxes: 
            return []

        # 3. 合并所有尺度的结果
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        kpts = np.concatenate(all_kpts)
        
        # 4. NMS 非极大值抑制
        try:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                       self.conf_thres, self.nms_thres)
        except:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 
                                       self.conf_thres, self.nms_thres)
        
        final_results = []
        if len(indices) > 0:
            idx_list = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for idx in idx_list:
                b = boxes[idx]
                rx1, ry1 = int((b[0] - tx) / r), int((b[1] - ty) / r)
                rx2, ry2 = int((b[2] - tx) / r), int((b[3] - ty) / r)
                
                curr_kpts = kpts[idx].copy()
                curr_kpts[:, 0] = (curr_kpts[:, 0] - tx) / r
                curr_kpts[:, 1] = (curr_kpts[:, 1] - ty) / r
                curr_kpts[:, 2] = 1 / (1 + np.exp(-curr_kpts[:, 2]))  # sigmoid
                
                final_results.append((scores[idx], classes[idx], (rx1, ry1, rx2, ry2), curr_kpts))
        
        return final_results

# ==================== ✅ PET SKELETON FIX: 遮挡补偿函数 ====================
def compensate_occlusion(kpts, cls_id):
    """
    牵引绳遮挡补偿：当鼻部关键点被遮挡时，用头部几何中心替代
    适用于狗/猫/马
    """
    if cls_id not in [16, 17, 18]:  # 仅宠物
        return kpts
    
    # 计算头部中心（眼+耳）
    if kpts.shape[0] >= 5:
        eye_center_x = (kpts[1, 0] + kpts[2, 0]) / 2
        eye_center_y = (kpts[1, 1] + kpts[2, 1]) / 2
        ear_center_x = (kpts[3, 0] + kpts[4, 0]) / 2
        ear_center_y = (kpts[3, 1] + kpts[4, 1]) / 2
        
        head_center_x = (eye_center_x + ear_center_x) / 2
        head_center_y = (eye_center_y + ear_center_y) / 2
        
        # 如果鼻部置信度极低，用头部中心替代
        if kpts[0, 2] < Config.KEYPOINT_CONF_MIN:
            kpts[0, 0] = head_center_x
            kpts[0, 1] = head_center_y
            kpts[0, 2] = Config.KEYPOINT_CONF_MIN * 1.2  # 提升至可显示水平
    
    return kpts

# ==================== 全局变量 ====================
detector = None
cap = None
stats = {
    'total_frames': 0,
    'total_detections': 0,
    'total_pet_detections': 0,
    'avg_fps': 0.0,
    'start_time': time.time()
}

# ==================== Flask 应用 ====================
app = Flask(__name__)

def initialize_detector():
    """初始化检测器"""
    global detector
    try:
        detector = PetPoseDetector(
            Config.MODEL_PATH,
            Config.CONF_THRESH,
            Config.NMS_THRESH
        )
        print("✅ 姿态检测器初始化成功")
        print(f"✅ 支持类别: {len(COCO_CLASSES)} 类 (COCO 数据集)")
        print(f"✅ 宠物类别: {Config.PET_CLASSES} (狗、猫、马)")
        return True
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_camera():
    """初始化摄像头（带降级机制）"""
    global cap
    
    # 尝试优先摄像头
    print(f"📹 尝试打开摄像头 {Config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    # 如果失败，尝试备用摄像头
    if not cap.isOpened():
        print(f"⚠️ 摄像头 {Config.CAMERA_INDEX} 打开失败，尝试备用摄像头 {Config.BACKUP_CAMERA_INDEX}...")
        cap = cv2.VideoCapture(Config.BACKUP_CAMERA_INDEX)
    
    # 配置摄像头参数
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
        
        # 验证设置
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✅ 摄像头初始化成功: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        return True
    else:
        print("❌ 无法打开任何摄像头")
        return False

def cleanup_resources():
    """清理资源"""
    global cap, detector
    print("\n🧹 清理资源...")
    if cap is not None and cap.isOpened():
        cap.release()
        print("✅ 摄像头已释放")
    print("✅ 资源清理完成")

def draw_text_with_bg(img, text, position, font_scale=0.7, thickness=2):
    """在图像上绘制带背景的文本（解决编码问题）"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # 绘制背景矩形
    cv2.rectangle(img, (x, y - text_height - baseline), 
                  (x + text_width, y + baseline), 
                  Config.TEXT_BG_COLOR, -1)
    # 绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, 
                Config.TEXT_COLOR, thickness, cv2.LINE_AA)

def generate_frames():
    """生成视频流帧 - 宠物骨架优化版"""
    global cap, detector, stats
    
    frame_count = 0
    fps_history = []
    last_stats_update = time.time()
    
    while True:
        start_t = time.time()
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧，尝试重新初始化摄像头...")
            if not initialize_camera():
                # 返回错误帧
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error!", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue
        
        frame_count += 1
        stats['total_frames'] += 1
        
        # 推理流程
        try:
            nv12_img, pad_info = detector.preprocess(frame)
            outputs = detector.quantize_model.forward(nv12_img)
            results = detector.postprocess(outputs, pad_info)
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"⚠️ 推理错误: {e}")
                import traceback
                traceback.print_exc()
            results = []
        
        # 绘制结果
        detection_count = len(results)
        pet_detection_count = 0
        stats['total_detections'] += detection_count
        
        for score, cls_id, (x1, y1, x2, y2), kpts in results:
            # 过滤：只显示宠物或所有类别
            if not Config.SHOW_ALL_CLASSES and cls_id not in Config.PET_CLASSES:
                continue
            
            pet_detection_count += 1
            stats['total_pet_detections'] += 1
            
            # ✅ PET SKELETON FIX: 遮挡补偿
            kpts = compensate_occlusion(kpts, cls_id)
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.BBOX_COLOR, 2)
            
            # 绘制类别标签
            class_name = COCO_CLASSES.get(int(cls_id), f'class_{cls_id}')
            label = f"{class_name} {score:.2f}"
            draw_text_with_bg(frame, label, (x1, y1 - 5), 0.5, 1)
            
            # ✅ PET SKELETON FIX: 选择对应宠物的骨架连接
            skeleton = PET_SKELETON_MAP.get(cls_id, DOG_SKELETON)
            
            # 绘制骨架：两种模式
            valid_kpts = []
            for i, kp in enumerate(kpts):
                if kp[2] > Config.KEYPOINT_CONF_THRESH:
                    valid_kpts.append((i, kp))
            
            # 模式1: 高质量绘制（≥8个关键点有效）
            if len(valid_kpts) >= 8:
                kp_dict = {idx: kp for idx, kp in valid_kpts}
                for s_idx, e_idx in skeleton:
                    if s_idx in kp_dict and e_idx in kp_dict:
                        kp1, kp2 = kp_dict[s_idx], kp_dict[e_idx]
                        cv2.line(frame, (int(kp1[0]), int(kp1[1])), 
                                (int(kp2[0]), int(kp2[1])), 
                                Config.SKELETON_COLOR, 2)
            else:
                # 模式2: 降级绘制（关键点少时）
                # 1. 绘制头部三角（鼻+双眼）- 即使鼻部被遮挡也尝试
                if kpts[1, 2] > Config.KEYPOINT_CONF_MIN and kpts[2, 2] > Config.KEYPOINT_CONF_MIN:
                    cv2.line(frame, (int(kpts[1, 0]), int(kpts[1, 1])), 
                            (int(kpts[2, 0]), int(kpts[2, 1])), 
                            (0, 255, 255), 2)  # 青色
                    # 如果鼻部可用，连接鼻-眼
                    if kpts[0, 2] > Config.KEYPOINT_CONF_MIN:
                        cv2.line(frame, (int(kpts[0, 0]), int(kpts[0, 1])), 
                                (int(kpts[1, 0]), int(kpts[1, 1])), 
                                Config.SKELETON_COLOR, 1)
                        cv2.line(frame, (int(kpts[0, 0]), int(kpts[0, 1])), 
                                (int(kpts[2, 0]), int(kpts[2, 1])), 
                                Config.SKELETON_COLOR, 1)
                
                # 2. 绘制躯干中心线（肩→髋）- 最鲁棒的连接
                if kpts[5, 2] > Config.KEYPOINT_CONF_MIN and kpts[11, 2] > Config.KEYPOINT_CONF_MIN:
                    cv2.line(frame, (int(kpts[5, 0]), int(kpts[5, 1])), 
                            (int(kpts[11, 0]), int(kpts[11, 1])), 
                            (0, 255, 255), 3)  # 青色粗线
                elif kpts[6, 2] > Config.KEYPOINT_CONF_MIN and kpts[12, 2] > Config.KEYPOINT_CONF_MIN:
                    cv2.line(frame, (int(kpts[6, 0]), int(kpts[6, 1])), 
                            (int(kpts[12, 0]), int(kpts[12, 1])), 
                            (0, 255, 255), 3)
            
            # 绘制关键点（按类型着色）
            for i, kp in enumerate(kpts):
                if kp[2] > Config.KEYPOINT_CONF_THRESH:
                    if i in HEAD_POINTS:
                        color = Config.HEAD_POINT_COLOR  # 红色
                    elif i in ANCHOR_POINTS:
                        color = Config.ANCHOR_POINT_COLOR  # 蓝色
                    else:
                        color = Config.KEYPOINT_COLOR  # 默认红色
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 
                              4, color, -1)
                # 降级模式：高置信度关键点（即使低于主阈值）
                elif kp[2] > Config.KEYPOINT_CONF_MIN and i in HEAD_POINTS + ANCHOR_POINTS:
                    color = (255, 255, 255)  # 白色（提示存在）
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 
                              3, color, -1)
        
        # 显示 FPS 和检测数量
        if Config.SHOW_FPS:
            fps = 1.0 / (time.time() - start_t + 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 10:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            stats['avg_fps'] = avg_fps
            
            # 显示检测信息
            if Config.SHOW_ALL_CLASSES:
                info_text = f"FPS: {avg_fps:.1f} | All: {detection_count}"
            else:
                info_text = f"FPS: {avg_fps:.1f} | Pets: {pet_detection_count}/{detection_count}"
            draw_text_with_bg(frame, info_text, (20, 40), 1.0, 2)
            
            # 调试模式下显示更多统计信息
            if Config.DEBUG_MODE:
                debug_text = f"Total: {stats['total_frames']} frames"
                draw_text_with_bg(frame, debug_text, (20, 80), 0.6, 1)
        
        # 保存调试图像
        if Config.SAVE_DEBUG_IMAGE and frame_count % 100 == 0:
            debug_filename = f"debug_frame_{frame_count}.jpg"
            cv2.imwrite(debug_filename, frame)
            print(f"💾 已保存调试图像: {debug_filename}")
        
        # 编码推流
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"⚠️ 编码错误: {e}")
            continue

@app.route('/')
def index():
    """主页 - 美化版"""
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RDK Pet Pose Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            text-align: center;
            padding: 40px 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 2.8em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            font-weight: 700;
        }
        
        .subtitle {
            font-size: 1.3em;
            color: #aaa;
            margin-bottom: 30px;
            font-weight: 300;
        }
        
        .stats-banner {
            background: rgba(0, 0, 0, 0.6);
            padding: 15px 30px;
            border-radius: 12px;
            display: inline-block;
            margin: 20px 0;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .stats-banner strong {
            color: #4ecdc4;
            margin-right: 5px;
        }
        
        .stats-banner span {
            margin: 0 15px;
            color: #fff;
        }
        
        .video-container {
            max-width: 950px;
            margin: 30px auto;
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.6);
            border-radius: 20px;
            overflow: hidden;
            border: 4px solid rgba(255, 255, 255, 0.1);
            background: #000;
            position: relative;
        }
        
        .video-container img {
            max-width: 100%;
            height: auto;
            display: block;
            border-radius: 16px;
        }
        
        .footer {
            margin-top: 40px;
            color: #666;
            font-size: 0.95em;
            line-height: 1.8;
        }
        
        .footer p {
            margin: 8px 0;
        }
        
        .highlight {
            color: #4ecdc4;
            font-weight: 500;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #4ecdc4;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 2.2em; }
            .subtitle { font-size: 1.1em; }
            .stats-banner { padding: 12px 20px; font-size: 0.95em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 RDK Pet Pose Detection 🐶</h1>
        <p class="subtitle">Real-time Pet Pose Estimation System based on Ultralytics YOLO11n-Pose</p>
        
        <div class="stats-banner">
            <strong>Model:</strong> YOLO11n-Pose (Ultralytics)
            <span>|</span>
            <strong>Resolution:</strong> 640x640
            <span>|</span>
            <strong>Classes:</strong> 80 (COCO)
            <span>|</span>
            <strong>Pets:</strong> Dog(16), Cat(17), Horse(18)
            <span>|</span>
            <strong>Status:</strong>
            <span class="status-indicator"></span>Running
        </div>
        
        <div class="video-container">
            <img src="/video_feed" alt="Video Stream" id="videoStream">
        </div>
        
        <div class="footer">
            <p><span class="highlight">Powered by:</span> RDK X3/X3J • BPU Acceleration • Ultralytics YOLO11n-Pose</p>
            <p><span class="highlight">Access URL:</span> http://&lt;RDK_IP&gt;:5000</p>
            <p><span class="highlight">Health Check:</span> http://&lt;RDK_IP&gt;:5000/health</p>
            <p><span class="highlight">Pet Skeleton:</span> 17-point anatomical mapping with occlusion compensation</p>
            <p style="margin-top: 20px; color: #888; font-size: 0.85em;">
                System Status: Operational • Press Ctrl+C to stop
            </p>
        </div>
    </div>
    
    <script>
        // Auto-reconnect on error
        const img = document.getElementById('videoStream');
        img.onerror = function() {
            console.log('Stream error, reconnecting...');
            setTimeout(() => {
                img.src = '/video_feed?' + new Date().getTime();
            }, 2000);
        };
    </script>
</body>
</html>'''

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    """健康检查接口"""
    global cap, detector, stats
    
    current_time = time.time()
    uptime = current_time - stats['start_time']
    
    status = {
        "status": "ok" if (cap and cap.isOpened() and detector) else "error",
        "camera": cap is not None and cap.isOpened(),
        "detector": detector is not None,
        "uptime_seconds": round(uptime, 1),
        "total_frames": stats['total_frames'],
        "total_detections": stats['total_detections'],
        "total_pet_detections": stats['total_pet_detections'],
        "avg_fps": round(stats['avg_fps'], 1),
        "model": Config.MODEL_PATH,
        "resolution": f"{Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}",
        "pet_classes": Config.PET_CLASSES,
        "show_all_classes": Config.SHOW_ALL_CLASSES
    }
    
    return jsonify(status)

@app.route('/stats')
def stats_api():
    """统计信息 API"""
    return jsonify(stats)

@app.route('/config')
def config_api():
    """配置信息 API"""
    config_info = {
        "model_path": Config.MODEL_PATH,
        "conf_thresh": Config.CONF_THRESH,
        "nms_thresh": Config.NMS_THRESH,
        "pet_classes": Config.PET_CLASSES,
        "show_all_classes": Config.SHOW_ALL_CLASSES,
        "keypoint_conf_thresh": Config.KEYPOINT_CONF_THRESH,
        "keypoint_conf_min": Config.KEYPOINT_CONF_MIN,
        "skeleton_map": {str(k): v for k, v in PET_SKELETON_MAP.items()}
    }
    return jsonify(config_info)

@app.teardown_appcontext
def teardown(exception=None):
    """应用上下文清理"""
    pass

def signal_handler(sig, frame):
    """信号处理（Ctrl+C）"""
    print("\n\n" + "=" * 60)
    print("🛑 收到终止信号，正在关闭服务...")
    print("=" * 60)
    cleanup_resources()
    print("\n👋 服务已关闭，再见！")
    sys.exit(0)

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 60)
    print("🚀 RDK Pet Pose Detection System - Pet Skeleton Optimized")
    print("=" * 60)
    
    # 初始化组件
    print("\n🔧 初始化姿态检测器...")
    if not initialize_detector():
        print("\n❌ 检测器初始化失败，程序退出")
        sys.exit(1)
    
    print("\n📹 初始化摄像头...")
    if not initialize_camera():
        print("\n❌ 摄像头初始化失败，程序退出")
        cleanup_resources()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 系统初始化完成！")
    print("=" * 60)
    print(f"\n🌐 Web Service Started!")
    print(f"📍 Access URL: http://<RDK_IP>:{Config.PORT}")
    print(f"📊 Health Check: http://<RDK_IP>:{Config.PORT}/health")
    print(f"📈 Stats API: http://<RDK_IP>:{Config.PORT}/stats")
    print(f"⚙️  Config API: http://<RDK_IP>:{Config.PORT}/config")
    print(f"\n🎯 检测模式: {'所有类别' if Config.SHOW_ALL_CLASSES else '仅宠物'}")
    print(f"🐾 宠物类别: {Config.PET_CLASSES}")
    print(f"🧠 骨架优化: 牵引绳遮挡补偿 + 降级绘制模式")
    print(f"\n⌨️  Press Ctrl+C to stop the service")
    print("=" * 60 + "\n")
    
    try:
        # 启动 Flask 服务
        app.run(host=Config.HOST, port=Config.PORT, threaded=True, debug=False)
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_resources()