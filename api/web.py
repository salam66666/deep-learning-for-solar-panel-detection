from flask import Blueprint, request, render_template, jsonify, current_app
# from mysql.connector import Error
web_api = Blueprint('web_api', __name__)

@web_api.route('/upload')
def upload():
    return render_template('upload.html')

@web_api.route('/display')
def display():
    return render_template('display.html')

@web_api.route('/explain')
def explain():
    return render_template('explain.html')

@web_api.route('/result.html')
def result_page():
    return render_template('result.html')
