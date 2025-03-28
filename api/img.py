import os
import random
import time
from datetime import datetime

from PIL import Image
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

img_api = Blueprint('img_api', __name__)
# 获取数据库连接池
def init():
    # 获取连接池
    cnxpool = current_app.config['dbpool']
    # 从连接池中获取连接
    conn = cnxpool.get_connection()
    cursor = conn.cursor()
    return conn, cursor



@img_api.route('/upload_files', methods=['POST'])
def upload_images():
    # 检查请求中是否包含文件
    if 'images' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    images = request.files.getlist('images')  # 获取所有上传的文件

    # 检查是否至少上传了一个文件
    if len(images) == 0:
        return jsonify({'error': 'No files selected'}), 400

    results = []
    img_folder = os.path.join(current_app.root_path, current_app.config.get('UPLOAD_FOLDER', 'static/img/temp_img/'))
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    results = []
    for image in images:
        if image.filename == '':
            results.append({'error': 'No selected file for one of the images'})
            continue

        img_count = len(os.listdir(img_folder))
        filename = secure_filename(f"{img_count + 1}.png")

        filepath = os.path.join(img_folder, filename)
        result = process_image(image, filepath, image.filename)
        if not result:
            return jsonify({'error': 'Image upload error'}), 400
        results.extend(result)

    return jsonify(results)



@img_api.route('/upload_file', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Invalid request'}), 400

    image = request.files['image']
    # 存储用户上传照片的记录

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    img_folder = os.path.join(current_app.root_path, current_app.config.get('UPLOAD_FOLDER', 'static/img/temp_img/'))
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    img_count = len(os.listdir(img_folder))
    filename = secure_filename(f"{img_count + 1}.png")

    filepath = os.path.join(img_folder, filename)
    print(filepath)
    # 直接将图像传给process_image函数进行处理和保存
    # 原版
    # img_width, img_height, img_mode, img_format = process_image(image, filepath, model_id, type, image.filename)
    # 新版
    result = process_image(image, filepath, image.filename)
    if not result:
        return jsonify({'error': 'No image'}), 400
    return jsonify(result)


# todo 在此函数中调用算法
    # 原版

# 源文件image_file, 存储位置filepath
def process_image(image_file, filepath,  filename):
    image = Image.open(image_file)
    img = image.convert('L')  # 灰度处理
    # 保存到指定路径并指定格式
    img.save(filepath, format='PNG')

    # 获取图像属性
    width, height = img.size
    mode = img.mode
    format = img.format if img.format else 'PNG'
    return width, height, mode, format

@img_api.route('/getDetected', methods=['GET'])
def get_detected_images():
    # 获取目录下所有图片文件（假设文件为 PNG、JPG、JPEG 格式）
    allowed_extensions = {'jpg', 'jpeg', 'png'}

    try:
        # 获取所有文件名
        images = [f for f in os.listdir('E:\\CDUT\\English\\7 Term\\Project\\model\\static\\img\\temp_img') if f.rsplit('.', 1)[1].lower() in allowed_extensions]
        return jsonify(images), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # 新版
#     conn, cursor = init()
#     time.sleep(random.randint(1, 5))
#     try:
#         query = "SELECT * FROM backup WHERE name = %s"
#         cursor.execute(query, (filename,))
#         result = cursor.fetchone()
#
#         if result:
#             img_type = result[1]
#             number = result[3]
#             confidence = result[4]
#             path = result[5]
#             return {
#                 "imgName": filename,
#                 "img_type": img_type,
#                 "number": number,
#                 "confidence": confidence,
#                 "path": path
#             }
#         else:
#             return False
#     except Error as e:
#         return False
#     finally:
#         cursor.close()
#         conn.close()


def save_upload_info(username, type):
    # 连接到数据库
    conn, cursor = init()
    query = "SELECT id FROM user WHERE username = %s"
    cursor.execute(query, (username, ))
    result = cursor.fetchone()

    if result:
        id = result[0]
    query = "INSERT INTO img_log (user_id, username, datetime, type) VALUES (%s, %s, %s, %s)"
    current_time = datetime.now()
    cursor.execute(query, (id, username, current_time, type))
    conn.commit()
    # 返回模型名称
    cursor.close()
    conn.close()

