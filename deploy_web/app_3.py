# -*- coding:utf-8 -*-
"""
# @file name  : app_3.py
# @author     : zz0320
# @date       : 2022-5-10
# @brief      : HTML中提供上传图片功能
"""
from flask import Flask, render_template, request
import time
import os

# 初始化
BASEDIR = os.path.dirname(__file__)
app = Flask(__name__)
upload_dir = os.path.join(BASEDIR, "static", "upload_img")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)


# 定义该url接收get和post请求， 可到route的add_url_rule函数中看到默认是get请求
@app.route("/", methods=["GET", "POST"])
def func():
    # request 就是一个请求对象，用户提交的请求信息都在request中
    if request.method == "POST":
        f = request.files['imgfile']

        def save_img(file, out_dir):
            time_stamp = str(time.time())
            file_name = time_stamp + file.filename
            path_to_img = os.path.join(out_dir, file_name)
            file.save(path_to_img)
        save_img(f, upload_dir)

    return render_template("upload.html")


if __name__ == '__main__':
    app.run()

