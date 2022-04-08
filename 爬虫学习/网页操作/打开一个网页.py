# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 14:40
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 打开一个网页.py
# @Software: PyCharm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
#打开谷歌浏览器
driver = webdriver.Chrome(ChromeDriverManager().install())
#打开百度搜索主页
driver.get('https://www.baidu.com')

'''
调用selenium库中的find_element_by_xpath()方法定位搜索框，
同时使用send_keys()方法在其中输入信息
'''
driver.find_element_by_xpath('//*[@id="kw"]').send_keys('中国')
driver.find_element_by_xpath('//*[@id="su"]').click()