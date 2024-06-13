# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 15:40
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Myfirst爬虫.py
# @Software: PyCharm

import requests     #requests库
import re           #正则表达式
import time
import os

'''请求网站'''
#反反爬写法
headers = {         #Internet 中第一个 headers中找到
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'
}
#---------------------------------------------------------------------------------------------------
#url = 'https://m.tujigu.com/a/34916/'
#url = 'https://m.tujigu.com/a/3210/'
url = 'https://cn.bing.com/images/search?q=%e5%9b%be%e7%89%87&qpvt=%e5%9b%be%e7%89%87&form=IGRE&first=1&tsc=ImageBasicHover'
response = requests.get(url = url,headers = headers)  #此网站无反爬机制
print(response.request.headers) # 打印请求信息
html = response.text
# print(response.text)           #打印网页内容
'''解析网页'''
# dir_name = re.findall('<img .*? alt="(.*?)" .*?>',html)[-1]
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)



#urls = re.findall('<img src="(.*?)" .*?>',html)
urls = re.findall('<img .*? src="(.*?)" .*?>',html)
print(urls)


'''保存图片'''
# for url in urls:
#     time.sleep(1)
#     file_name = url.split('/')[-1]
#     response= requests.get(url=url,headers = headers)
#     with open(dir_name + '/' + file_name,'wb') as f:
#             f.write(response.content)
