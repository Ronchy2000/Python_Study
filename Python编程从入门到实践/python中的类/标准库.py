# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 13:32
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 标准库.py
# @Software: PyCharm
from collections import OrderedDict
favorite_languages = OrderedDict()

favorite_languages['jen'] = 'Python'
favorite_languages['sarah'] = 'c'
favorite_languages['edward'] = 'ruby'
favorite_languages['phil'] = 'python'

for name,language in favorite_languages.items():
    print(name.title() + "'sfavorite language is" +language.title() + ".")

