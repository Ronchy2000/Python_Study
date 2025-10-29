import requests
import random
import time
from datetime import datetime
import hashlib

# ============ 配置区域 ============
TARGET_URL = "https://github.com/Ronchy2000/Multi-agent-RL"  # 修改为你要访问的网址
MAX_INTERVAL = 26  # 最大访问间隔（秒）
MIN_INTERVAL = 5  # 最小访问间隔（秒），避免间隔过短

# 更丰富的User-Agent列表
USER_AGENTS = [
    # Windows - Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # Windows - Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0",
    # Windows - Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/119.0.0.0",
    
    # macOS - Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # macOS - Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    # macOS - Firefox
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    
    # Linux - Chrome
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Linux - Firefox
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
    
    # Android - Chrome
    "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
    
    # iOS - Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

# 生成随机IP（X-Forwarded-For）
def generate_random_ip():
    """生成随机IP地址"""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

def get_random_headers():
    """生成随机的请求头，模拟不同设备和网络"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': random.choice(['zh-CN,zh;q=0.9,en;q=0.8', 'en-US,en;q=0.9', 'ja-JP,ja;q=0.8', 'zh-TW,zh;q=0.9']),
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': random.choice(['no-cache', 'max-age=0']),
        'X-Forwarded-For': generate_random_ip(),  # 模拟不同IP
        'DNT': random.choice(['1', '0']),  # Do Not Track
    }
    
    # 随机添加Referer
    if random.random() > 0.3:
        referers = [
            'https://www.google.com/',
            'https://www.baidu.com/',
            'https://www.bing.com/',
            'https://twitter.com/',
            'https://www.facebook.com/',
            ''
        ]
        headers['Referer'] = random.choice(referers)
    
    return headers

def get_device_fingerprint(headers):
    """生成设备指纹用于追踪"""
    ua = headers.get('User-Agent', '')
    # 简单的设备识别
    if 'iPhone' in ua or 'iPad' in ua:
        device_type = 'iOS'
    elif 'Android' in ua:
        device_type = 'Android'
    elif 'Windows' in ua:
        device_type = 'Windows'
    elif 'Macintosh' in ua:
        device_type = 'macOS'
    elif 'Linux' in ua:
        device_type = 'Linux'
    else:
        device_type = 'Unknown'
    
    # 生成设备ID（基于UA的hash）
    device_id = hashlib.md5(ua.encode()).hexdigest()[:8]
    
    return device_type, device_id

def visit_url(url, visit_num):
    """访问指定的URL"""
    try:
        headers = get_random_headers()
        device_type, device_id = get_device_fingerprint(headers)
        
        # 发送请求
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*70}")
        print(f"[{timestamp}] 第 {visit_num} 次访问")
        print(f"{'='*70}")
        print(f"✓ 访问成功")
        print(f"  状态码: {response.status_code}")
        print(f"  设备类型: {device_type}")
        print(f"  设备ID: {device_id}")
        print(f"  模拟IP: {headers.get('X-Forwarded-For')}")
        print(f"  User-Agent: {headers['User-Agent'][:80]}...")
        print(f"  Referer: {headers.get('Referer', '(无)')}")
        print(f"  响应大小: {len(response.content)} bytes")
        print(f"  最终URL: {response.url}")
        
        # 记录到文件
        log_to_file(timestamp, visit_num, device_type, device_id, headers, response)
        
        return True
    except requests.exceptions.RequestException as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] ✗ 访问失败: {str(e)}")
        return False

def log_to_file(timestamp, visit_num, device_type, device_id, headers, response):
    """记录访问日志到文件"""
    try:
        with open('visit_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | Visit#{visit_num} | {device_type} | DeviceID:{device_id} | ")
            f.write(f"IP:{headers.get('X-Forwarded-For')} | Status:{response.status_code}\n")
    except Exception as e:
        print(f"  警告: 无法写入日志文件 - {e}")

def get_random_interval():
    """生成随机访问间隔（秒）"""
    # 使用对数正态分布，使间隔更符合实际攻击模式
    # 大部分间隔较短，少数间隔很长
    mu = (MIN_INTERVAL + MAX_INTERVAL) / 4  # 平均值偏向较小
    sigma = MAX_INTERVAL / 6  # 标准差
    
    interval = random.lognormvariate(0, 1) * sigma
    interval = max(MIN_INTERVAL, min(interval, MAX_INTERVAL))
    
    return interval

def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*20 + "随机访问模拟脚本")
    print("="*70)
    print(f"目标URL: {TARGET_URL}")
    print(f"访问间隔: {MIN_INTERVAL}秒 ~ {MAX_INTERVAL}秒")
    print(f"设备池大小: {len(USER_AGENTS)} 种不同设备")
    print(f"日志文件: visit_log.txt")
    print("="*70)
    print("\n提示: 按 Ctrl+C 停止程序\n")
    
    visit_count = 0
    success_count = 0
    
    try:
        while True:
            visit_count += 1
            
            # 访问URL
            success = visit_url(TARGET_URL, visit_count)
            if success:
                success_count += 1
            
            # 计算下次访问的间隔
            interval = get_random_interval()
            next_visit_time = datetime.now().timestamp() + interval
            next_visit_str = datetime.fromtimestamp(next_visit_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n⏱  等待 {interval:.1f}秒 ({interval/60:.1f}分钟)")
            print(f"📅 下次访问: {next_visit_str}")
            print(f"📊 成功率: {success_count}/{visit_count} ({success_count/visit_count*100:.1f}%)")
            print("-"*70)
            
            # 等待
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(" "*25 + "程序终止")
        print("="*70)
        print(f"总访问次数: {visit_count}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_count/visit_count*100:.1f}%" if visit_count > 0 else "N/A")
        print("="*70)
        print("\n日志已保存到 visit_log.txt\n")

if __name__ == "__main__":
    main()
    