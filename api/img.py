import os
import json
import time
import uuid
from datetime import datetime
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
import threading
lock = threading.Lock()

# 模型相关导入
from model.model_utils import preprocess_image, load_model

# Flask 蓝图
img_api = Blueprint('img_api', __name__)

# ✅ 加载模型（只加载一次）
model = load_model()

# 🔧 初始化数据库连接（如果需要保存日志）
def init():
    cnxpool = current_app.config['dbpool']
    conn = cnxpool.get_connection()
    cursor = conn.cursor()
    return conn, cursor

# 🔧 获取 heatmap 路径（避免在模块顶层访问 current_app）
def get_heatmap_folder():
    return os.path.join(current_app.root_path, current_app.config.get('HEATMAP_FOLDER', 'static/heatmap'))

# 🔧 预测缓存路径

def get_cache_file_path():
    return os.path.join(current_app.root_path, 'static/prediction_cache.json')

def load_prediction_cache():
    path = get_cache_file_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_prediction_cache(cache):
    path = get_cache_file_path()
    with lock:
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)

@img_api.route('/upload_file', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'Invalid request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_folder = os.path.join(current_app.root_path, current_app.config.get('UPLOAD_FOLDER', 'static/img/temp_img/'))
    os.makedirs(img_folder, exist_ok=True)

    filename = secure_filename(image.filename)
    filepath = os.path.join(img_folder, filename)

    try:
        image.save(filepath)

        # === 检查缓存 ===
        cache = load_prediction_cache()
        if filename in cache:
            return jsonify({
                "status": "cached",
                "filename": filename,
                "class": cache[filename]["class"],
                "confidence": cache[filename]["confidence"]
            })

        # === 预测 ===
        processed = preprocess_image(filepath)
        prediction = model.predict(processed)
        confidence = float(prediction[0][0])
        class_name = "defected" if confidence > 0.5 else "non-defected"
        confidence_score = abs(confidence - 0.5) * 2
        confidence_str = f"{confidence_score:.1%}"

        # === 保存到缓存 ===
        cache[filename] = {
            "class": class_name,
            "confidence": confidence_str
        }
        save_prediction_cache(cache)

        return jsonify({
            "status": "success",
            "filename": filename,
            "class": class_name,
            "confidence": confidence_str
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@img_api.route('/upload_files', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    images = request.files.getlist('images')
    if len(images) == 0:
        return jsonify({'error': 'No files selected'}), 400

    img_folder = os.path.join(current_app.root_path, current_app.config.get('UPLOAD_FOLDER', 'static/img/temp_img/'))
    os.makedirs(img_folder, exist_ok=True)

    results = []
    for image in images:
        if image.filename == '':
            continue

        try:
            filename = secure_filename(image.filename)
            filepath = os.path.join(img_folder, filename)
            image.save(filepath)

            with Image.open(filepath) as img:
                results.append({
                    "status": "success",
                    "filename": filename,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format
                })
        except Exception as e:
            results.append({
                "status": "error",
                "filename": image.filename,
                "message": str(e)
            })

    return jsonify(results)

@img_api.route('/getDetected', methods=['GET'])
def get_detected_images():
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    try:
        dir_path = os.path.join(current_app.root_path, 'static/img/temp_img')
        files = []

        for f in os.listdir(dir_path):
            if f.split('.')[-1].lower() in allowed_extensions:
                file_path = os.path.join(dir_path, f)
                files.append({
                    'filename': f,
                    'mtime': os.path.getmtime(file_path)
                })

        files.sort(key=lambda x: x['mtime'], reverse=True)
        sorted_files = [f['filename'] for f in files]

        return jsonify(sorted_files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_upload_info(username, type):
    conn, cursor = init()
    query = "SELECT id FROM user WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()

    if result:
        user_id = result[0]
        current_time = datetime.now()
        query = "INSERT INTO img_log (user_id, username, datetime, type) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (user_id, username, current_time, type))
        conn.commit()

    cursor.close()
    conn.close()

# -------- Grad-CAM 相关 --------
TARGET_WIDTH = 128
TARGET_HEIGHT = 213

def preprocess_image_for_model(image):
    resized = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0), resized

def generate_grad_cam(model, img_array, target_layer_name, class_idx=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

@img_api.route('/predict_with_heatmap', methods=['POST'])
def predict_with_heatmap():
    heatmap_folder = get_heatmap_folder()
    os.makedirs(heatmap_folder, exist_ok=True)

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    base_name = os.path.splitext(filename)[0]
    uid = str(uuid.uuid4())[:8]
    unique_name = f"{base_name}_{uid}.png"

    img_path = os.path.join(current_app.root_path, 'static/img/temp_img', unique_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    image.save(img_path)

    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_input, resized_input = preprocess_image_for_model(img)

        prediction = model.predict(processed_input)
        confidence = float(prediction[0][0])
        class_name = "defected" if confidence > 0.5 else "non-defected"
        confidence_score = abs(confidence - 0.5) * 2
        confidence_str = f"{confidence_score:.1%}"

        heatmap = generate_grad_cam(model, processed_input, "inception_block")

        heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (resized_input.shape[1], resized_input.shape[0])))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        image_bgr = cv2.cvtColor(resized_input, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)

        heatmap_path = os.path.join(heatmap_folder, f"{base_name}_{uid}_heatmap.png")
        overlay_path = os.path.join(heatmap_folder, f"{base_name}_{uid}_overlay.png")
        cv2.imwrite(heatmap_path, heatmap_color)
        cv2.imwrite(overlay_path, overlay_bgr)

        return jsonify({
            "status": "success",
            "filename": unique_name,
            "class": class_name,
            "confidence": confidence_str,
            "original_img_url": f"/static/img/temp_img/{unique_name}",
            "heatmap_url": f"/static/heatmap/{base_name}_{uid}_heatmap.png",
            "overlay_url": f"/static/heatmap/{base_name}_{uid}_overlay.png"
        })

    except Exception as e:
        import traceback
        print("❌ Error during heatmap prediction:")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
