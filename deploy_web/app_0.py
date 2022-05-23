# -*- coding:utf-8 -*-
"""
# @file name  : app_0.py
# @author     : zz0320
# @date       : 2022-05-10
# @brief      : 第一个网页：Hello World
"""
from flask import Flask


# step1：定义一个flask实例
app = Flask(__name__)


# step2：视图函数：定义url中需要执行的功能，并返回网页需显示的内容，可显示字符串、图片、表格、HTML等
def show_something():
    return "Hello World!"


# step3：设置路由：将url地址与需要执行的函数进行绑定，当进入指定的url时，自动调用对应的功能函数获取网页要显示的内容
app.add_url_rule("/", "show_something", show_something)
# app.add_url_rule("/my_path/abc/", "show_something", show_something)


if __name__ == '__main__':
    # step4：启动flask应用
    app.run(host="127.0.0.1", port=5000)
    # app.run(host='0.0.0.0', port=80, debug=True)  # local，局域网访问，此时需要找到此服务器在局域网的IP：通过ipconfig