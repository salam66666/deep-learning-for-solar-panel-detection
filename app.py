from flask import Flask, redirect, url_for, send_file

# from db import cnxpool  # 引入连接池
# from flask_cors import CORS
#
app = Flask(__name__)
# CORS(app)
#
# # 确保连接池可在全局范围内访问
# app.config['dbpool'] = cnxpool

# 添加根路径的重定向
@app.route('/')
def root():
    return redirect(url_for('web_api.upload'))  # 重定向到蓝图的 upload 路由


# 注册蓝图
from api.web import web_api
from api.img import img_api
app.register_blueprint(web_api, url_prefix='/web')
app.register_blueprint(img_api, url_prefix='/img')


if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)
