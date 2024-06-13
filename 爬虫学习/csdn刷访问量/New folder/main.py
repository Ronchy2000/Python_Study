#!/usr/bin/env python
'''
1. 手动输入某个新闻链接，然后模拟人工访问网页
2. 可定义：N 个网页、刷新数量、刷新速度、访问浏览器
3. 数量要求：[700 ~ 100000]
'''
from ast import Try
import requests
import warnings
import urllib3
from bs4 import BeautifulSoup 
import os,time,random,datetime
import _thread,queue
import user_set as user
from selenium import webdriver
from selenium.webdriver.chrome.options import Options 
from requests.packages.urllib3.exceptions import InsecureRequestWarning
 
 
warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
urllib3.disable_warnings()
available_ip = ''
test_url = 'https://www.baidu.com'
 
# 读取有效 ip 文件
def ip_file ():
    available_ips=[]
    try:
        with open('./ipporxy.txt','r',encoding='UTF-8') as f:
            for l in f.readlines():
                available_ips.append(l.rstrip('\n'))
        return available_ips
    except Exception:
        print("Error 没有ipporxy.txt文件")
        exit()
 
def ip_random ():
    ips = ip_file ()
    if ips:
        proxie = random.choice(ips)
        requests.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        proxies = { "http": proxie, "https": proxie}
        try:
            res = requests.get(test_url,proxies=proxies,verify=False,timeout=2)
        except Exception:
            print('代理 ip 连接出错，更换ip中...')
            ip_random ()
        else:
            if res.status_code == 200:
                print('ip 通道正常：[%s]，可以使用....'%proxie)
                available_ip = proxies
    else:
        print('Error 没有可用的代理 ip')
    
 
# 设置主功能，传入线程
def pv_pool(name,url,count):
    for i in range(count):
        headers = random.choice(user.UserAgent1)
        chrome_options = Options() 
        webdriver.DesiredCapabilities.CHROME['acceptSslCerts']=True
        chrome_options.add_argument('headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--user-agent=%s'%headers)
        try:
            chrome_options.add_argument('--proxy-server=%s'%available_ip) 
            driver = webdriver.Chrome(options = chrome_options) 
            driver.get(url) # 访问页面
        except Exception:
            print('Error 访问出现错误，更换 ip')
            ip_random ()
            continue
        else:
            time.sleep(random.uniform(user.speed_low,user.speed_high))
            driver.close()
            print("第 %d 次访问网页【%s】"%(i+1,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('~~~~{}线程结束时间：{}~~~~~\n'.format(name,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
 
 
if __name__ == "__main__":
    # 获取有效 ip 地址
    ip_random ()
 
    # 开始刷点击量
    for site in user.webSites:
        # url = site['url']
        url = site['https://www.jianshu.com/p/c898d931291f']
        times = int(site['times'])//3
        print(times)
        try:
            # print(url,times)
            # 启动多线程
            print('※※※※※线程启动时间：{}※※※※\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            _thread.start_new_thread(pv_pool, (1,url,times))
            _thread.start_new_thread(pv_pool, (2,url,times))
            _thread.start_new_thread(pv_pool, (3,url,times))
        except:
            print ("Error: 无法启动线程")
    while 1:
        pass