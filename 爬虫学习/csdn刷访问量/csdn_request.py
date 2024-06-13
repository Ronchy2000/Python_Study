# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:47:35 2023

@author: Sword
"""
import requests
import time

i=1

proxies = {
    'http': 'socks5://127.0.0.1:8443',
    'https': 'socks5://127.0.0.1:8443'
}


for _ in range(313):
    print(f"第{i}轮")
    file=open('CSDN.txt', 'r',encoding='utf-8',errors='ignore') #打开文件读取访问文章的url
    while True:
        url=file.readline().rstrip()  #读取文件中的一行，删除该行右端的所有空白字符
        header={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
            } #请求头
        try:
            data=requests.get(url=url,headers=header,proxies = proxies) #利用requests发送请求
        except ValueError:
            break
        else:
            print(data.status_code,end='')
            if(data.status_code == 200): #判断返回状态
                print(f"访问{url}成功")
            else:
                print(f"访问{url}失败")
            time.sleep(4) #每访问一个url等待时间
    file.close()
    time.sleep(3) #访问完所有url等待时间进入下一轮
    i+=1 #访问完一轮次数加1
