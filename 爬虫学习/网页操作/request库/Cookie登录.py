# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 16:16
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Cookie登录.py
# @Software: PyCharm
import urllib.request as urllib2
import http.cookiejar as cookielib
import requests

headers = {
    #'Cookies':'o_session_id=C8EA29E4C5F4C39E8A876502A1EE78A7; Hm_lvt_0a1b7151d8c580761c3aef32a3d501c6=1649413999; Hm_lpvt_0a1b7151d8c580761c3aef32a3d501c6=1649414008; CASTGC=TGT-1771466-hIZA6wpmxGWyRei5YzMGaNqxjg1MBsEGWugWQZlk7rWPW4W0NK-passport.zhihuishu.com; CASLOGC={"realName":"陆荣琦","myuniRole":0,"myinstRole":0,"userId":200454153,"headPic":"https://image.zhihuishu.com/zhs/user/weixin/201908/1eb1cc789db545da930f3e9001da2ffc_s3.jpg","uuid":"VjMQbqmQ","mycuRole":0,"username":"8ac00a3ff8d6491eace699fab8533a2b"}; jt-cas=eyJraWQiOiJFRjRGMjJDMC01Q0IwLTQzNDgtOTY3Qi0wMjY0OTVFN0VGQzgiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJjb20uemhpaHVpc2h1IiwiYXVkIjoiQ0FTIiwic3ViIjoi6ZmG6I2j55CmIiwiaWF0IjoxNjQ5NDE0MDIxLCJleHAiOjE2NDk1MDA0MjEsImp0aSI6IjZjYjhjMWM1LWQ2ODUtNGI4Yi05MTY1LTY0ODdhNzUzODA5MiIsInVpZCI6MjAwNDU0MTUzfQ.PrC30VkKJDZbyZDfYkmAzBHb5E4GnuPPaL2td2ohPWZKqiPFp_tFcpm85GIBneKYuL-ssOWfmfGoStcUx_3yUg; exitRecod_VjMQbqmQ=2',

    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'
}

post_url = 'http://authserver.sicnu.edu.cn/authserver/login?service=http%3A%2F%2Fehall.sicnu.edu.cn%2Fqljfw%2Fsys%2FlwReportEpidemicUndergraduate%2F*default%2Findex.do#/'
session = requests.session()
post_data={"username":"2019070950",
           "password":"XYeeeee520!!"
    }

session.post(post_url,data=post_data,headers=headers)
r = session.get('http://202.115.194.60/(S(i5e3gijsgldn5po2qxqxzjmu))/Index.aspx?sid=53429DA170B038CB5FF3D55EE08164B57F9C9BCC',headers=headers)

with open('zhihuishu.html','w') as f:
    f.write(r.content.decode())
print(r.text)

# print('status_code:',req.status_code) # 请求状态
# print(req.url)# 请求url
# print(req.text) # 请求结果
