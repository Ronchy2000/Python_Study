# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 14:20
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 报平安.py
# @Software: PyCharm
# import requests
# #川师报平安
url_bpa = 'http://authserver.sicnu.edu.cn/authserver/login?service=http%3A%2F%2Fehall.sicnu.edu.cn%2Fqljfw%2Fsys%2FlwReportEpidemicUndergraduate%2F*default%2Findex.do#/'
# response = requests.get(url=url_bpa, headers={'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'})
# # 获取requests请求返回的cookie
# cookie = requests.utils.dict_from_cookiejar(response.cookies)
# print(cookie)

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
#打开谷歌浏览器
driver = webdriver.Chrome(ChromeDriverManager().install())

driver.get(url_bpa)

'''
调用selenium库中的find_element_by_xpath()方法定位搜索框，
同时使用send_keys()方法在其中输入信息
'''
driver.find_element_by_xpath('//*[@id="username"]').send_keys('2019070950')
driver.find_element_by_xpath('//*[@id="password"]').send_keys('XYeeeee520!!')

driver.find_element_by_xpath('//*[@id="casLoginForm"]/p[5]/button').click()
time.sleep(200)
driver.find_element_by_class_name('mint-button geuhjrnk mt-btn-primary mint-button--normal').click()
time.sleep(200)
#选择返校
driver.find_element_by_xpath('//*[@id="app"]/div/div/div[1]/div/div/div/div/div/div/a/span').click()
driver.find_element_by_xpath('//*[@id="app"]/div/div/div[2]/div/div[2]/div/div[2]/div[2]/div/a/div[2]/div[2]/input').send_keys('36.2')
driver.find_element_by_xpath('//*[@id="app"]/div/div/div[3]/button').click()
