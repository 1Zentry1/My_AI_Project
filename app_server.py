import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template_string
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "D:/Parking_AI/runs/segment/mixed_v3/weights/best.pt"
STATIC_DIR = "static"

CLASS_NAMES = {0: "road", 1: "sidewalk", 2: "crosswalk", 3: "car"}
SIDEWALK_CLASS = 1
CROSSWALK_CLASS = 2
CAR_CLASS = 3

DEFAULT_MASK_OVERLAP = 0.05
DEFAULT_IOU_BOX = 0.05
DEFAULT_CONF = 0.54

app = Flask(__name__)

# ── Современный HTML с CSS-анимациями ─────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ParkGuard AI – Детектор нарушений парковки</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255,255,255,0.95);
            border-radius: 24px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 900px;
            padding: 30px;
            transition: all 0.3s ease;
        }
        h1 {
            text-align: center;
            color: #1e3c72;
            margin-bottom: 20px;
            font-size: 2.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .upload-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 25px;
            background: #f8faff;
            padding: 20px;
            border-radius: 16px;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        label {
            font-weight: 600;
            color: #333;
        }
        input[type="file"], input[type="number"] {
            padding: 10px;
            border: 2px solid #e0e5f0;
            border-radius: 12px;
            font-size: 1rem;
            transition: border-color 0.2s;
            outline: none;
        }
        input[type="file"]:focus, input[type="number"]:focus {
            border-color: #1e3c72;
        }
        .threshold-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .threshold-row .form-group {
            flex: 1;
            min-width: 140px;
        }
        .deep-check-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 5px;
        }
        .deep-check-row input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        button {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 14px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            align-self: flex-start;
            box-shadow: 0 8px 20px rgba(30,60,114,0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(30,60,114,0.4);
        }
        button:active {
            transform: translateY(0);
        }
        /* Loading overlay */
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5);
            justify-content: center;
            align-items: center;
            z-index: 999;
        }
        .loading-box {
            background: white;
            padding: 30px 50px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        .spinner {
            border: 6px solid #e0e5f0;
            border-top: 6px solid #1e3c72;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        /* Result section animation */
        .result-section {
            margin-top: 30px;
            animation: fadeInUp 0.5s ease;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stats {
            background: #f0f4ff;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .image-card {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .violation-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 8px;
        }
        .violation-badge.red { background: #ffe0e0; color: #c00; }
        .violation-badge.green { background: #e0ffe0; color: #0a0; }
        .time-info {
            font-size: 0.9rem;
            color: #555;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <!-- Loading overlay -->
    <div id="loadingOverlay" class="loading-overlay">
        <div class="loading-box">
            <div class="spinner"></div>
            <p>Идет анализ изображения…</p>
            <p style="font-size:0.8rem; color:#666;">Пожалуйста, подождите</p>
        </div>
    </div>

    <div class="container">
        <h1>🅿️ ParkGuard AI</h1>
        <form id="uploadForm" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <div class="upload-section">
                <div class="form-group">
                    <label for="image">📷 Загрузите фото</label>
                    <input type="file" id="image" name="image" accept="image/*" required>
                </div>

                <div class="threshold-row">
                    <div class="form-group">
                        <label for="mask_overlap">🎯 Пересечение масок (доля)</label>
                        <input type="number" id="mask_overlap" name="mask_overlap" min="0.01" max="1.0" step="0.01" value="{{ mask_overlap | default(0.05) }}">
                    </div>
                    <div class="form-group">
                        <label for="iou_box">📦 IoU боксов</label>
                        <input type="number" id="iou_box" name="iou_box" min="0.01" max="1.0" step="0.01" value="{{ iou_box | default(0.05) }}">
                    </div>
                    <div class="form-group">
                        <label for="conf">🔍 Уверенность (conf)</label>
                        <input type="number" id="conf" name="conf" min="0.01" max="1.0" step="0.01" value="{{ conf | default(0.54) }}">
                    </div>
                </div>

                <div class="deep-check-row">
                    <input type="checkbox" id="deep_check" name="deep_check" {% if deep_check %}checked{% endif %}>
                    <label for="deep_check">🛡️ Тщательная проверка (точнее, но медленнее)</label>
                </div>

                <button type="submit">🔎 Проверить</button>
            </div>
        </form>

        {% if result_img %}
        <div class="result-section">
            <h3>Результат</h3>
            <div class="stats">
                <div>🚗 Дорог: {{ road_count }} | 🧑‍🦯 Тротуаров: {{ sidewalk_count }} | 🚸 Переходов: {{ crosswalk_count }} | 🚘 Машин: {{ car_count }}</div>
                <div style="margin-top:8px;">
                    🚨 Нарушений: <span class="violation-badge {{ 'red' if violation_count > 0 else 'green' }}">{{ violation_count }} из {{ car_count }}</span>
                </div>
                <div class="time-info">⏱️ Время обработки: {{ elapsed_time }} сек.</div>
                <div style="font-size:0.9rem; color:#666; margin-top:5px;">
                    Пороги: масок={{ mask_overlap }}, IoU боксов={{ iou_box }}, уверенность={{ conf }}
                    {% if deep_check %} | Тщательная проверка включена {% endif %}
                </div>
            </div>

            <div class="image-card">
                <img src="{{ url_for('static', filename='result.jpg') }}" alt="Результат">
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        function showLoading() {
            document.getElementById('loadingOverlay').style.display = 'flex';
        }
    </script>
</body>
</html>
"""

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0

def is_violation(car_mask, sw_mask, cw_mask, car_box, sw_boxes, mask_th, iou_th):
    forbidden = cv2.bitwise_or(sw_mask, cw_mask)
    inter = cv2.bitwise_and(car_mask, forbidden)
    car_px = np.sum(car_mask)
    if car_px > 0 and np.sum(inter) / car_px >= mask_th:
        car_cy = (car_box[1] + car_box[3]) / 2
        for sbox in sw_boxes:
            sw_cy = (sbox[1] + sbox[3]) / 2
            if car_cy < sw_cy and max(car_box[0], sbox[0]) < min(car_box[2], sbox[2]):
                return True
    for sbox in sw_boxes:
        if compute_iou(car_box, sbox) >= iou_th:
            car_cy = (car_box[1] + car_box[3]) / 2
            sw_cy = (sbox[1] + sbox[3]) / 2
            if car_cy < sw_cy and max(car_box[0], sbox[0]) < min(car_box[2], sbox[2]):
                return True
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    mask_overlap = DEFAULT_MASK_OVERLAP
    iou_box = DEFAULT_IOU_BOX
    conf = DEFAULT_CONF
    deep_check = False

    if request.method == 'POST':
        logging.info("Получен POST запрос")
        file = request.files.get('image')
        if not file:
            return "Файл не найден", 400
        try:
            mask_overlap = float(request.form.get('mask_overlap', DEFAULT_MASK_OVERLAP))
            iou_box = float(request.form.get('iou_box', DEFAULT_IOU_BOX))
            conf = float(request.form.get('conf', DEFAULT_CONF))
        except ValueError:
            mask_overlap, iou_box, conf = DEFAULT_MASK_OVERLAP, DEFAULT_IOU_BOX, DEFAULT_CONF
        deep_check = 'deep_check' in request.form

        os.makedirs(STATIC_DIR, exist_ok=True)
        img_path = os.path.join(STATIC_DIR, 'original.jpg')
        file.save(img_path)
        logging.info(f"Изображение сохранено: {img_path}")

        start_time = time.time()

        # Первый проход
        results1 = model(img_path, conf=conf, iou=0.4, imgsz=640, verbose=False)
        det1 = results1[0]

        if deep_check:
            results2 = model(img_path, conf=min(conf + 0.1, 0.95), iou=0.4, imgsz=640, verbose=False)
            det2 = results2[0]
            # Нарушения должны быть подтверждены в обоих проходах
            viol1 = find_violations(det1, mask_overlap, iou_box)
            viol2 = find_violations(det2, mask_overlap, iou_box)
            violations = viol1.intersection(viol2)
            detection = det2
        else:
            violations = find_violations(det1, mask_overlap, iou_box)
            detection = det1

        elapsed = round(time.time() - start_time, 2)

        result_img = draw_results(detection, violations)
        cv2.imwrite(os.path.join(STATIC_DIR, 'result.jpg'), result_img)

        cls_list = [int(c) for c in detection.boxes.cls] if detection.boxes is not None else []
        road = cls_list.count(0); sidewalk = cls_list.count(1); crosswalk = cls_list.count(2); car = cls_list.count(3)

        return render_template_string(HTML_TEMPLATE,
                                      result_img=True,
                                      road_count=road,
                                      sidewalk_count=sidewalk,
                                      crosswalk_count=crosswalk,
                                      car_count=car,
                                      violation_count=len(violations),
                                      mask_overlap=mask_overlap,
                                      iou_box=iou_box,
                                      conf=conf,
                                      deep_check=deep_check,
                                      elapsed_time=elapsed)

    return render_template_string(HTML_TEMPLATE, result_img=False,
                                  mask_overlap=mask_overlap,
                                  iou_box=iou_box,
                                  conf=conf,
                                  deep_check=False)

def find_violations(det, mask_overlap, iou_box):
    h, w = det.orig_img.shape[:2]
    violations = set()
    if det.masks is not None:
        masks = det.masks.data.cpu().numpy()
        boxes = det.boxes.xyxy.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)
        sw_mask = np.zeros((h, w), dtype=np.uint8)
        cw_mask = np.zeros((h, w), dtype=np.uint8)
        sw_boxes = []
        for i, c in enumerate(cls):
            m = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if c == SIDEWALK_CLASS:
                sw_mask = cv2.bitwise_or(sw_mask, m)
                sw_boxes.append(boxes[i])
            elif c == CROSSWALK_CLASS:
                cw_mask = cv2.bitwise_or(cw_mask, m)
        for i, c in enumerate(cls):
            if c == CAR_CLASS:
                car_mask = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                if is_violation(car_mask, sw_mask, cw_mask, boxes[i], sw_boxes, mask_overlap, iou_box):
                    violations.add(i)
    return violations

def draw_results(det, violations):
    img = det.orig_img.copy()
    h, w = img.shape[:2]
    if det.masks is not None:
        masks = det.masks.data.cpu().numpy()
        boxes = det.boxes.xyxy.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)
        sw_mask = np.zeros((h, w), dtype=np.uint8)
        cw_mask = np.zeros((h, w), dtype=np.uint8)
        for i, c in enumerate(cls):
            m = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if c == SIDEWALK_CLASS:
                sw_mask = cv2.bitwise_or(sw_mask, m)
            elif c == CROSSWALK_CLASS:
                cw_mask = cv2.bitwise_or(cw_mask, m)
        overlay = img.copy()
        if np.any(sw_mask):
            cv2.drawContours(overlay, cv2.findContours(sw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), -1)
        if np.any(cw_mask):
            cv2.drawContours(overlay, cv2.findContours(cw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,255,255), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        for i, c in enumerate(cls):
            if c == CAR_CLASS:
                x1, y1, x2, y2 = map(int, boxes[i])
                viol = i in violations
                color = (0,0,255) if viol else (0,255,0)
                label = "VIOLATION" if viol else "OK"
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    logging.info(f"Загрузка модели {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    logging.info("Модель загружена")
    app.run(host='127.0.0.1', port=5000, debug=False)