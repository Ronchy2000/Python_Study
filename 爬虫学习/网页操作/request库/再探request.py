# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 15:13
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 再探request.py
# @Software: PyCharm
import requests
url ='https://passport.zhihuishu.com/login?service=https://onlineservice.zhihuishu.com/login/gologin'
#请求时，提供一个headers，现在大部分网站都增加了安全验证
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'
}

res = requests.get(url,headers=headers)  # 豆瓣首页
# print("页面状态:", res.status_code)
if res.status_code ==200:
    print('Accessed Successfully！')
else:
    print('There\'s someting goes wrong!',res.status_code,sep=':')
# print(res.text)