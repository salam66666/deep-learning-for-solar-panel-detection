import os
import json
import time
from datetime import datetime
from PIL import Image
from flask import Blueprint, request, jsonify, current_app
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


# ✅ 单张图片上传并预测
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
        class_name = "detected" if confidence > 0.5 else "non-detected"
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


# ✅ 多张图片上传（无模型预测）
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


# ✅ 获取所有已上传图片，按时间倒序返回
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

        # 按时间倒序
        files.sort(key=lambda x: x['mtime'], reverse=True)
        sorted_files = [f['filename'] for f in files]

        return jsonify(sorted_files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ 可选：将上传记录写入数据库日志（如启用）
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
    with lock:  # 添加线程锁，确保写入是串行的
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)

