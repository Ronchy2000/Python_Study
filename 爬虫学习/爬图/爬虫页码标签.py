# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 0:05
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 爬虫页码标签.py
# @Software: PyCharm
import requests
import re
import time
import os

# dir_name = '图片'
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)

headers = {  # Internet 中第一个 headers中找到
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'
}

url = 'http://pic.netbian.com/'
response = requests.get(url= url,headers= headers)
html = response.text
print (html)

urls = re.findall('<div class="page"><b>.*?</b><a href="(.*?)">.*?</a>.*?</div>',html)
# print(urls)
print('*********************************************************************')
for url in urls:
    url = 'http://pic.netbian.com' + url
    print(url)
    response = requests.get(url= url,headers= headers)
    html = response.text
    print (html)


# i = 1
# while i < 5 :
#     url = 'http://pic.netbian.com/index' + str(i) + '.html'
#     i += 1
#     response = requests.get(url= url,headers= headers)
#     print(response.request.headers)
#     html = response.text



#
# #url = 'http://pic.netbian.com/'
#     headers = {         #Internet 中第一个 headers中找到
#         'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Mobile Safari/537.36'
#     }
#     response = requests.get(url= url,headers= headers)
#     print(response.request.headers)
#     html = response.text
#
#     urls = re.findall('<a href="(.*?)" .*?><span><img .*?></span><b>.*?</b></a>',html)
#     print(urls)
#     for url in urls:
#         url = 'http://pic.netbian.com' + url
#         print(url)
#     #***** Done! href
#     #以下为爬虫图片的知识：
#         response_2 = requests.get(url=url, headers=headers)
#     #    print(response_2.request.headers)
#         html_2 = response_2.text
#         url_2  = re.findall('<img src="(.*?)" data-pic=".*?" .*?>',html_2)
#         for url_3 in url_2:
#             url_3  = 'http://pic.netbian.com' + url_3
#             time.sleep(1)
#             file_name = url_3.split('/')[-1]
#             response = requests.get(url=url_3, headers=headers)
#             with open(dir_name + '/' + file_name, 'wb') as f:
#                 f.write(response.content)


