# -*- coding: utf-8 -*-
# @Time    : 2021/1/23 13:10
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 爬经济学人.py
import requests
import re
import time
import os

url = 'https://magazinelib.com/?s=the+economist'
headers = {         #Internet 中第一个 headers中找到
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75'
}
response = requests.get(url=url,headers=headers)
print(response.request.headers)
html = response.text
print(response.text)
#
# dir_name = re.findall('<img .*? alt="(.*?)" .*?>',html)[-1]
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
#

###可匹配到！进入具体界面！
# urls= re.findall('<a href="(.*?)" rel=".*?">.*?, .*?</a>',html)
# print(urls)
###匹配next url！  没配到！
# #urls1= re.findall('<a class=".*?" href="(.*?)" one-link-mark="yes">Next</a>',html)
# #print(urls1)
urls2=re.findall('<a href="(.*?)" target="_blank" one-link-mark="yes"><img src="https://magazinelib.com/wp-includes/images/media/default.png" height="21" class="lazyloaded" data-ll-status="loaded"><noscript><img src="https://magazinelib.com/wp-includes/images/media/default.png" height="21" /></noscript>The Economist USA - January 16, 2021.pdf</a>',html)
<a href="https://vk.com/doc527052285_584513323?hash=4564574f949fc7232c&amp;dl=GQZDGNBSGQ2DGNQ:1611052426:dbe748ba96ef0cbbbe&amp;api=1&amp;no_preview=1" target="_blank" one-link-mark="yes"> <img src="https://magazinelib.com/wp-includes/images/media/default.png" height="21" class="lazyloaded" data-ll-status="loaded"><noscript><img src="https://magazinelib.com/wp-includes/images/media/default.png" height="21" /></noscript>The Economist USA - January 16, 2021.pdf</a>
print(urls2)
# for url in urls:
#     url = 'http://pic.netbian.com' + url
#     time.sleep(1)
#     file_name = url.split('/')[-1]
#     response= requests.get(url=url,headers = headers)
#     with open(dir_name + '/' + file_name,'wb') as f:
#             f.write(response.content)