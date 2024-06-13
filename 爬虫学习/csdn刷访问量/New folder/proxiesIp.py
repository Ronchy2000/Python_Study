import telnetlib
import requests
 
test_url = 'https://www.baidu.com'
 
# ip 检测，存储有效 ip地址
def ip_is_alive(ip_port):
    ip,port=ip_port[0],ip_port[1]
    try:
        tn = telnetlib.Telnet(ip, port=port,timeout=1)
    except:
        print('[-]无效ip:{}:{}'.format(ip,port))
    else:
        proxies = ip+':'+port
        try:
            res = requests.get(test_url, proxies={"http": proxies, "https": proxies},timeout=1)
        except:
            print('[-]无效ip:{}:{}'.format(ip,port))
        else:   
            if res.status_code == 200:
                print('[+]有效ip:{}:{}'.format(ip,port))
                # 将有效 ip 写入文件中
                with open('ipporxy.txt','a+') as f:
                    f.write(ip+':'+port+'\n')
        
        
 
 
 