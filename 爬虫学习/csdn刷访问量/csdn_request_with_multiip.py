# coding: utf-8

import urllib.request  
import urllib.parse 
import time  
from multiprocessing import Pool#多进程
import random
from lxml import etree  #解析

def GetUserAgent():
    '''
    功能：随机获取HTTP_User_Agent
    '''
    user_agents=[
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1"
    ]
    user_agent = random.choice(user_agents)
    return user_agent

def getProxies():
    '''
    功能：爬取西刺高匿IP构造原始代理IP池
    '''
    init_proxies = []
    ##爬取前五页
    for i in range(1,6):
        print("####")
        print("####爬取第"+str(i)+"页####")
        print("####")        
        print("IP地址\t\t\t端口\t存活时间\t\t验证时间")
        url = "http://www.xicidaili.com/nn/"+str(i)
        user_agent = GetUserAgent()
        headers=("User-Agent",user_agent)
        opener = urllib.request.build_opener() 
        opener.addheaders = [headers] 
        try:
            data = opener.open(url,timeout=5).read()
        except Exception as er:
            print("爬取的时候发生错误，具体如下：")
            print(er)
        selector=etree.HTML(data)
        '''
        ip_addrs = selector.xpath('//tr[@class="odd"]/td[2]/text()')  #IP地址
        port = selector.xpath('//tr[@class="odd"]/td[3]/text()')  #端口
        sur_time = selector.xpath('//tr[@class="odd"]/td[9]/text()')  #存活时间
        ver_time = selector.xpath('//tr[@class="odd"]/td[10]/text()')  #验证时间
        '''
        #原代码只提取了奇数行部分，修改下
        ip_addrs = selector.xpath('//tr/td[2]/text()')  #IP地址
        port = selector.xpath('//tr/td[3]/text()')  #端口
        sur_time = selector.xpath('//tr/td[9]/text()')  #存活时间
        ver_time = selector.xpath('//tr/td[10]/text()')  #验证时间
        for j in range(len(ip_addrs)):
            ip = ip_addrs[j]+":"+port[j] 
            init_proxies.append(ip)
            #输出爬取数据 
            print(ip_addrs[j]+"\t\t"+port[j]+"\t\t"+sur_time[j]+"\t"+ver_time[j])
    return init_proxies


def testProxy(curr_ip):
    '''
    功能：验证IP有效性
    @curr_ip：当前被验证的IP
    '''
    tmp_proxies = []
    #socket.setdefaulttimeout(10)  #设置全局超时时间
    #tarURL = "http://www.baidu.com/" 
    #tarURL = "http://blog.csdn.net/deserts_x/article/details/76726885" #CSDN-老树画画
    tarURL = "https://blog.csdn.net/Ronchy_2000/article/details/139610981?spm=1001.2014.3001.5501" #CSDN-Ronchy
    user_agent = GetUserAgent()
    proxy_support = urllib.request.ProxyHandler({"http":curr_ip})
    opener = urllib.request.build_opener(proxy_support)
    opener.addheaders=[("User-Agent",user_agent)]
    urllib.request.install_opener(opener)
    try:
        res = urllib.request.urlopen(tarURL,timeout=5).read()
        if len(res)!=0:
            tmp_proxies.append(curr_ip)
    except urllib.error.URLError as er2: 
        if hasattr(er2,"code"):
            print("验证代理IP（"+curr_ip+"）时发生错误（错误代码）："+str(er2.code))
        if hasattr(er2,"reason"):
            print("验证代理IP（"+curr_ip+"）时发生错误（错误原因）："+str(er2.reason))
    except Exception as er:
        print("验证代理IP（"+curr_ip+"）时发生如下错误）：")
        print(er)
    time.sleep(2)
    return tmp_proxies


def mulTestProxies(init_proxies):
    '''
    功能：多进程验证IP有效性
    @init_proxies：原始未验证代理IP池
    '''
    pool = Pool(processes=7)
    fl_proxies = pool.map(testProxy,init_proxies)
    pool.close()
    pool.join()  #等待进程池中的worker进程执行完毕
    return fl_proxies


if __name__ == '__main__':
    #---（1）获取代理IP池
    init_proxies = getProxies()  #获取原始代理IP
    tmp_proxies = mulTestProxies(init_proxies) #多进程测试原始代理IP  
    proxy_addrs = []
    for tmp_proxy in tmp_proxies:
        if len(tmp_proxy)!=0:
            #print(tmp_proxy)
            proxy_addrs.append(tmp_proxy)

    print(len(proxy_addrs))
    print(proxy_addrs)