# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 22:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 爬虫图片.py
# @Software: PyCharm
import requests
import re
import time
import os

url = 'http://pic.netbian.com/tupian/26010.html'
headers = {         #Internet 中第一个 headers中找到
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'
}
response = requests.get(url= url,headers = headers)
print(response.request.headers)
html = response.text
# print(response.text)

dir_name = re.findall('<img .*? alt="(.*?)" .*?>',html)[-1]
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

urls = re.findall('<img src="(.*?)" data-pic=".*?" .*?>',html)
print(urls)
for url in urls:
    url = 'http://pic.netbian.com' + url
    time.sleep(1)
    file_name = url.split('/')[-1]
    response= requests.get(url=url,headers = headers)
    with open(dir_name + '/' + file_name,'wb') as f:
            f.write(response.content)