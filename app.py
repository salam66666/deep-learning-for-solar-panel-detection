import os

from flask import Flask, redirect, url_for, send_file

# from db import cnxpool  # 引入连接池
# from flask_cors import CORS
#
app = Flask(__name__)
# CORS(app)
#
# # 确保连接池可在全局范围内访问
# app.config['dbpool'] = cnxpool

# 上传目录配置（供 img.py 使用）
app.config['UPLOAD_FOLDER'] = 'static/img/temp_img'
app.config['HEATMAP_FOLDER'] = 'static/heatmap'

# 确保静态目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

@app.route('/')
def root():
    return redirect(url_for('web_api.upload'))


# 注册蓝图
from api.web import web_api
from api.img import img_api
app.register_blueprint(web_api, url_prefix='/web')
app.register_blueprint(img_api, url_prefix='/img')


if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)
