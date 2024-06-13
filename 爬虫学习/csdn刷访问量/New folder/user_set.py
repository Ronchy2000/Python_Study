#!/usr/bin/env python
import proxiesIp
import requests
from lxml import etree
 
# 设置多个需要刷量网址，及其次数，默认值为num
num = 30
webSites = [{'url':'www.baidu.com','times':num}]
 
# 多个网页
# webSites = [{'url':'xxx','times':num},{'url':'xxx','times':num}]
 
# 设置随机的刷新速度
speed_low = 2
speed_high = 3
 
# 设置微信浏览器：安卓，ios
UserAgent1 =[
    {'User-Agent':'mozilla/5.0 (linux; u; android 4.1.2; zh-cn; mi-one plus build/jzo54k) applewebkit/534.30 (khtml, like gecko) version/4.0 mobile safari/534.30 MicroMessenger/5.0.1.352'},
    {'User-Agent':'mozilla/5.0 (iphone; cpu iphone os 5\_1\_1 like mac os x) applewebkit/534.46 (khtml, like gecko) mobile/9b206 MicroMessenger/5.0'},
    {'User-Agent':'Mozilla/5.0 (Linux; Android 5.0; SM-N9100 Build/LRX21V) > AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 > Chrome/37.0.0.0 Mobile Safari/537.36 > MicroMessenger/6.0.2.56\_r958800.520 NetType/WIFI'},
    {'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 7\_1\_2 like Mac OS X) > AppleWebKit/537.51.2 (KHTML, like Gecko) Mobile/11D257 > MicroMessenger/6.0.1 NetType/WIFI'},
    {'User-Agent':'Mozilla/5.0 (Linux; U; Android 2.3.6; zh-cn; GT-S5660 Build/GINGERBREAD) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1 MicroMessenger/4.5.255'},
    {'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) AppleWebKit/537.51.2 (KHTML, like Gecko) Mobile/11D257 MicroMessenger/6.1 NetType/WIFI'}]
 
# 设置pc浏览器：chrome、safari、火狐 等
UserAgent2 = [
    {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'},
    {'User-Agent':'Mozilla/5.0 (Windows NT6.1; WOW64) Apple\WebKit/537.36 (KHTML, likeGecko) Chrome/69.0.3497.92 Safari/537.36'},
    {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'},
    {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'},
    {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'},
    {'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'},
    {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362'},
    {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'}]
 
# 测试伪装 ip 地址，后期设置 ip 接口获取
def turn_symbol(c1):
    new_c1 = c1.replace('\n','').replace('\t','')
    return  new_c1
 
#获取免费代理网站ip(ip+空格+port形式)
def getfreeIps():
    f=open('ipporxy.txt', "r+")
    f.truncate()
    for i in range(1,100):
        url = 'http://www.89ip.cn/index_{}.html'.format(i)
        res = requests.get(url)
        res.encoding = 'utf-8'
        html=etree.HTML(res.text)
        ipdress=html.xpath('//table[@class="layui-table"]/tbody/tr/td[1]/text()')
        port=html.xpath('//table[@class="layui-table"]/tbody/tr/td[2]/text()')
        ipdress=list(map(turn_symbol,ipdress))
        port=list(map(turn_symbol,port))
        data=list(zip(ipdress,port))
        
        for j in range(len(data)):
            # 测试并存储有效 ip 
            proxiesIp.ip_is_alive([data[j][0],data[j][1]])
 
if __name__ == "__main__":
    # 测试可用的 ip 地址，并进行本地存储
    getfreeIps()