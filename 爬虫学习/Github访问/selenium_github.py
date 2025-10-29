# -*- coding: utf-8 -*-
"""
selenium_github.py

用途：用真实浏览器自动打开页面、执行 JS、模拟真实浏览器加载，
伪装成不同设备访问，使用泊松分布控制访问间隔。

依赖安装：
pip install selenium webdriver-manager numpy

配置：
- 修改 CONFIG 字典中的参数
"""
import time
import random
import csv
import os
from datetime import datetime
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ============= 配置区域（在这里修改参数） =================
CONFIG = {
    "URL": "https://github.com/Ronchy2000/Multi-agent-RL",   # 目标 URL
    "MAX_VISITS": 5000,           # 最大访问次数（设置为 0 表示无限次）
    "INTERVAL_MEAN": 10,        # 平均访问间隔（秒），泊松分布的 lambda 参数
    "HEADLESS": True,           # True: 无头模式, False: 显示浏览器窗口（调试用）
    "WAIT_AFTER_LOAD": 3.0,     # 页面加载后等待的秒数
    "CSV_FILE": "visits_log_selenium.csv",
}
# ========================================================

# 常见设备的 User-Agent 列表（模拟不同设备）
USER_AGENTS = [
    # Desktop - Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    
    # Desktop - Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    
    # Desktop - Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    
    # Mobile - iPhone
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
    
    # Mobile - Android
    "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    
    # Tablet - iPad
    "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    
    # Tablet - Android
    "Mozilla/5.0 (Linux; Android 13; SM-X906C) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Safari/537.36",
]

# 不同设备的屏幕分辨率
SCREEN_SIZES = [
    {"width": 1920, "height": 1080},  # Desktop FHD
    {"width": 1366, "height": 768},   # Desktop HD
    {"width": 2560, "height": 1440},  # Desktop 2K
    {"width": 1536, "height": 864},   # Desktop
    {"width": 414, "height": 896},    # iPhone 11 Pro Max
    {"width": 390, "height": 844},    # iPhone 13
    {"width": 375, "height": 667},    # iPhone SE
    {"width": 412, "height": 915},    # Android Phone
    {"width": 360, "height": 740},    # Android Phone
    {"width": 820, "height": 1180},   # iPad Air
    {"width": 1024, "height": 1366},  # iPad Pro
    {"width": 800, "height": 1280},   # Android Tablet
]

def get_log_file_path():
    """
    获取日志文件路径，确保 logs 目录存在
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建 logs 目录路径
    logs_dir = os.path.join(script_dir, "logs")
    # 如果 logs 目录不存在，创建它
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # 返回完整的日志文件路径
    return os.path.join(logs_dir, CONFIG["CSV_FILE"])

def now_iso():
    """返回当前 UTC 时间的 ISO 格式字符串"""
    return datetime.utcnow().isoformat() + "Z"

def get_poisson_interval(mean):
    """
    使用泊松分布生成访问间隔时间
    mean: 平均间隔时间（秒）
    返回: 下一次访问的等待时间（秒）
    """
    # 泊松分布生成事件数，我们用指数分布生成时间间隔更合适
    # 指数分布的 lambda = 1/mean
    interval = np.random.exponential(mean)
    # 确保间隔不会太小（至少 2 秒）
    return max(2.0, interval)

def get_random_device():
    """
    随机选择一个设备配置（User-Agent 和屏幕尺寸）
    """
    ua = random.choice(USER_AGENTS)
    screen = random.choice(SCREEN_SIZES)
    return ua, screen

def create_driver(user_agent, screen_size):
    """
    创建一个配置好的 Chrome WebDriver
    """
    options = Options()
    
    if CONFIG["HEADLESS"]:
        options.add_argument("--headless=new")
    
    # 基本参数
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # 设置 User-Agent
    options.add_argument(f"user-agent={user_agent}")
    
    # 设置窗口大小
    options.add_argument(f"--window-size={screen_size['width']},{screen_size['height']}")
    
    # 隐藏 webdriver 标识
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # 创建 driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # 修改 navigator.webdriver 属性
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver

def write_csv_header(csvfile):
    """写入 CSV 文件头"""
    with open(csvfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "timestamp_utc", 
                "visit_number", 
                "url", 
                "user_agent", 
                "screen_width",
                "screen_height",
                "status", 
                "note"
            ])

def log_visit(visit_num, url, user_agent, screen_size, status, note=""):
    """记录访问日志到 CSV"""
    log_file = get_log_file_path()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_iso(),
            visit_num,
            url,
            user_agent,
            screen_size["width"],
            screen_size["height"],
            status,
            note
        ])

def visit_page(driver, url):
    """
    访问页面并执行一些随机操作
    """
    try:
        driver.get(url)
        
        # 等待页面加载
        time.sleep(CONFIG["WAIT_AFTER_LOAD"])
        
        # 随机滚动页面（模拟真实用户行为）
        scroll_times = random.randint(1, 3)
        for _ in range(scroll_times):
            scroll_amount = random.randint(300, 800)
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.5, 1.5))
        
        # 偶尔滚动回顶部
        if random.random() < 0.3:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(random.uniform(0.5, 1.0))
        
        return "SUCCESS", ""
    
    except Exception as e:
        return "ERROR", str(e)

def main():
    """主函数"""
    url = CONFIG["URL"]
    max_visits = CONFIG["MAX_VISITS"]
    log_file = get_log_file_path()
    
    print(f"🚀 开始访问任务")
    print(f"📍 目标 URL: {url}")
    print(f"🔢 最大访问次数: {max_visits if max_visits > 0 else '无限'}")
    print(f"⏱️  平均间隔: {CONFIG['INTERVAL_MEAN']} 秒（泊松分布）")
    print(f"💾 日志文件: {log_file}")
    print("-" * 60)
    
    write_csv_header(log_file)
    
    visit_count = 0
    driver = None
    
    try:
        while True:
            # 检查是否达到最大访问次数
            if max_visits > 0 and visit_count >= max_visits:
                print(f"\n✅ 已完成 {visit_count} 次访问，达到最大访问次数")
                break
            
            visit_count += 1
            
            # 获取随机设备配置
            user_agent, screen_size = get_random_device()
            
            # 每次访问都创建新的 driver（更好地模拟不同设备）
            try:
                if driver:
                    driver.quit()
            except:
                pass
            
            print(f"\n[访问 #{visit_count}]")
            print(f"  🖥️  设备: {screen_size['width']}x{screen_size['height']}")
            print(f"  🌐 UA: {user_agent[:80]}...")
            
            try:
                driver = create_driver(user_agent, screen_size)
                status, note = visit_page(driver, url)
                
                if status == "SUCCESS":
                    print(f"  ✅ 访问成功")
                else:
                    print(f"  ❌ 访问失败: {note}")
                
                log_visit(visit_count, url, user_agent, screen_size, status, note)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ 发生异常: {error_msg}")
                log_visit(visit_count, url, user_agent, screen_size, "EXCEPTION", error_msg)
            
            # 如果还没达到最大次数，等待下一次访问
            if max_visits == 0 or visit_count < max_visits:
                interval = get_poisson_interval(CONFIG["INTERVAL_MEAN"])
                print(f"  ⏳ 等待 {interval:.1f} 秒后进行下一次访问...")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    
    finally:
        if driver:
            try:
                driver.quit()
                print("🔒 已关闭浏览器")
            except:
                pass
        
        print(f"\n📊 总访问次数: {visit_count}")
        print(f"💾 日志已保存到: {log_file}")

if __name__ == "__main__":
    main()
    main()