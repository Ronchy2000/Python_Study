# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 14:54
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 登录智慧树.py
# @Software: PyCharm

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
#打开谷歌浏览器
driver = webdriver.Chrome(ChromeDriverManager().install())
#智慧树登录
url ='https://passport.zhihuishu.com/login?service=https://onlineservice.zhihuishu.com/login/gologin'
driver.get(url)

'''
调用selenium库中的find_element_by_xpath()方法定位搜索框，
同时使用send_keys()方法在其中输入信息
'''
driver.find_element_by_xpath('//*[@id="lUsername"]').send_keys('18903403141')
driver.find_element_by_xpath('//*[@id="lPassword"]').send_keys('lrq200075')

driver.find_element_by_xpath('//*[@id="f_sign_up"]/div[1]/span').click()
