# -*- coding:utf-8 -*-
"""
# @file name  : app_1.py
# @author     : zz0320
# @date       : 2022-05-10
# @brief      : 采用app.route()装饰器将视图函数与url绑定
"""
from flask import Flask


# step1：定义一个flask实例
app = Flask(__name__)


# step3：设置路由
# step2：视图函数
@app.route("/") # 把设置路由的功能干了
def show_something():
    return "Hello World!"


if __name__ == '__main__':
    # step4：启动flask应用
    app.run()
