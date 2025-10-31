# -*- coding: utf-8 -*-
"""
点赞功能测试脚本 - 测试切换 IP 后是否能重复点赞
"""
import os
import time
import random
import csv
from datetime import datetime, timezone

import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 从主脚本导入配置
from selenium_with_proxy import (
    MihomoProxyPool, CONFIG, USER_AGENTS, SCREEN_SIZES,
    get_random_device, now_iso
)

# ========== 点赞测试配置 ==========
LIKE_CONFIG = {
    "URL": "https://faculty.xidian.edu.cn/DANIEL/zh_CN/index.htm"
    ,"MAX_ATTEMPTS": 200  # 最大尝试次数（防止无限循环）
    ,"TARGET_LIKES": 50   # 目标有效点赞数（0=以最大尝试次数为准，>0=达到目标后停止）
    ,"INTERVAL_MEAN": 15  # 平均间隔（秒）
    ,"HEADLESS": True  # 建议先用 False 调试，看清楚点击过程
    
    # 点赞元素定位（需要根据实际页面调整）
    ,"LIKE_BUTTON_ID": "_parise_imgobj_u6"  # 点赞按钮 ID
    ,"LIKE_COUNT_ID": "_parise_obj_u6"      # 点赞数显示元素 ID
    
    # 等待时间
    ,"WAIT_AFTER_LOAD": 3     # 页面加载后等待
    ,"WAIT_AFTER_CLICK": 2    # 点击后等待
}

def create_driver_for_like(user_agent, screen_size, use_proxy=True, headless=False):
    """创建用于点赞的 Chrome 驱动"""
    options = webdriver.ChromeOptions()
    
    # 无头模式
    if headless:
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
    
    # User-Agent
    options.add_argument(f'user-agent={user_agent}')
    
    # 窗口大小
    options.add_argument(f'--window-size={screen_size["width"]},{screen_size["height"]}')
    
    # 代理设置
    if use_proxy:
        proxy_address = CONFIG["MIHOMO_PROXY"]
        options.add_argument(f'--proxy-server={proxy_address}')
    
    # 反检测设置
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # 清除所有浏览器数据（关键：清除 cookie 和 localStorage）
    options.add_argument('--disable-application-cache')
    options.add_argument('--disk-cache-size=0')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # 注入反检测脚本
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver

def clear_browser_storage(driver):
    """清除浏览器存储（关键步骤）"""
    try:
        # 清除 localStorage
        driver.execute_script("window.localStorage.clear();")
        
        # 清除 sessionStorage
        driver.execute_script("window.sessionStorage.clear();")
        
        # 删除所有 cookies
        driver.delete_all_cookies()
        
        print("  已清除浏览器存储（cookies + localStorage）")
        return True
    except Exception as e:
        print(f"  ⚠️ 清除存储失败: {e}")
        return False

def get_like_count(driver):
    """获取当前点赞数"""
    try:
        count_element = driver.find_element(By.ID, LIKE_CONFIG["LIKE_COUNT_ID"])
        count_text = count_element.text.strip()
        return int(count_text) if count_text.isdigit() else None
    except Exception as e:
        print(f"  ⚠️ 无法获取点赞数: {e}")
        return None

def click_like_button(driver):
    """点击点赞按钮"""
    try:
        # 等待按钮出现
        wait = WebDriverWait(driver, 10)
        like_button = wait.until(
            EC.element_to_be_clickable((By.ID, LIKE_CONFIG["LIKE_BUTTON_ID"]))
        )
        
        # 滚动到按钮位置
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", like_button)
        time.sleep(0.5)
        
        # 记录点击前的点赞数
        count_before = get_like_count(driver)
        
        # 点击按钮
        like_button.click()
        print(f"  已点击点赞按钮")
        
        # 等待更新
        time.sleep(LIKE_CONFIG["WAIT_AFTER_CLICK"])
        
        # 记录点击后的点赞数
        count_after = get_like_count(driver)
        
        # 判断是否成功
        if count_before is not None and count_after is not None:
            if count_after > count_before:
                print(f"  ✅ 点赞成功！{count_before} → {count_after}")
                return True, count_before, count_after
            else:
                print(f"  点赞失败，数量未变化: {count_before}")
                return False, count_before, count_after
        else:
            print(f"  ⚠️ 无法确认点赞结果")
            return None, count_before, count_after
            
    except Exception as e:
        print(f"  ❌ 点击失败: {e}")
        return False, None, None

def test_like_with_ip_rotation():
    """测试切换 IP 后点赞"""
    
    print("=" * 60)
    print("点赞功能测试 - 动态 IP 切换")
    print("=" * 60)
    print()
    print(f"📍 目标页面: {LIKE_CONFIG['URL']}")
    print(f"目标有效点赞数: {LIKE_CONFIG['TARGET_LIKES'] if LIKE_CONFIG['TARGET_LIKES'] > 0 else '不限制（以最大尝试次数为准）'}")
    print(f"最大尝试次数: {LIKE_CONFIG['MAX_ATTEMPTS']}")
    print(f"无头模式: {'开启' if LIKE_CONFIG['HEADLESS'] else '关闭（建议先关闭观察）'}")
    print()
    print("=" * 60)
    print()
    
    # 初始化代理池
    proxy_pool = MihomoProxyPool()
    if len(proxy_pool) == 0:
        print("❌ 代理池为空，无法进行测试")
        return
    
    total_nodes = len(proxy_pool)
    print(f"✅ 加载 {total_nodes} 个可用节点")
    print()
    
    success_count = 0
    fail_count = 0
    attempt_count = 0
    used_ips = set()  # 记录已使用的 IP
    available_nodes = proxy_pool.available_nodes.copy()  # 可用节点列表（用于顺序轮换）
    current_node_index = 0  # 当前节点索引
    
    # 判断停止条件
    def should_continue():
        if attempt_count >= LIKE_CONFIG["MAX_ATTEMPTS"]:
            return False
        if LIKE_CONFIG["TARGET_LIKES"] > 0 and success_count >= LIKE_CONFIG["TARGET_LIKES"]:
            return False
        if len(used_ips) >= total_nodes:
            # 所有 IP 已用完
            return False
        return True
    
    while should_continue():
        attempt_count += 1
        print(f"\n{'=' * 60}")
        print(f"[尝试 #{attempt_count}] (成功: {success_count}, 已用IP: {len(used_ips)}/{total_nodes})")
        print(f"{'=' * 60}")
        
        # 1. 顺序选择节点（避免重复）
        if current_node_index >= len(available_nodes):
            print("⚠️ 所有节点已遍历完毕，无更多可用 IP")
            break
        
        node = available_nodes[current_node_index]
        current_node_index += 1
        
        node_name = node.get("name")
        latency = node.get("latency_ms", "N/A")
        
        print(f"切换节点: {node_name} (延迟: {latency}ms)")
        
        success, msg = proxy_pool.switch_node(node_name)
        if not success:
            print(f"节点切换失败: {msg}")
            fail_count += 1
            continue
        
        print(f"节点切换成功")
        
        # 2. 查询当前 IP
        print(f"查询出口 IP...")
        exit_ip = proxy_pool.get_current_ip()
        if exit_ip:
            print(f"🌍 当前 IP: {exit_ip}")
            
            # 检查 IP 是否已使用过
            if exit_ip in used_ips:
                print(f"⚠️ 该 IP 已使用过，跳过")
                continue
            
            used_ips.add(exit_ip)
        else:
            print(f"⚠️ 无法获取 IP，跳过该节点")
            continue
        
        # 3. 创建新的浏览器实例（关键：每次都是全新的浏览器）
        user_agent, screen_size = get_random_device()
        print(f"设备: {screen_size['width']}x{screen_size['height']}")
        
        driver = None
        try:
            driver = create_driver_for_like(user_agent, screen_size, True, LIKE_CONFIG["HEADLESS"])
            
            # 4. 访问页面
            print(f"加载页面...")
            driver.get(LIKE_CONFIG["URL"])
            time.sleep(LIKE_CONFIG["WAIT_AFTER_LOAD"])
            
            # 5. 清除浏览器存储（关键步骤）
            clear_browser_storage(driver)
            
            # 6. 重新加载页面（让清除生效）
            print(f"重新加载页面...")
            driver.refresh()
            time.sleep(LIKE_CONFIG["WAIT_AFTER_LOAD"])
            
            # 7. 尝试点赞
            like_result, count_before, count_after = click_like_button(driver)
            
            if like_result:
                success_count += 1
            else:
                fail_count += 1
            
            # 8. 可选：查看 Network 请求（调试用）
            if not LIKE_CONFIG["HEADLESS"]:
                print(f"\n💡 提示：打开 DevTools (F12) 查看 Network 面板")
                print(f"   观察点击时是否发送了点赞请求")
                time.sleep(3)
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            fail_count += 1
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
        
        # 9. 检查是否达到目标
        if LIKE_CONFIG["TARGET_LIKES"] > 0 and success_count >= LIKE_CONFIG["TARGET_LIKES"]:
            print(f"\n✅ 已达到目标有效点赞数: {success_count}/{LIKE_CONFIG['TARGET_LIKES']}")
            break
        
        # 10. 检查 IP 是否用完
        if len(used_ips) >= total_nodes:
            print(f"\n⚠️ 所有 {total_nodes} 个节点的 IP 已全部使用完毕")
            break
        
        # 11. 等待下一次测试
        if should_continue():
            interval = np.random.exponential(LIKE_CONFIG["INTERVAL_MEAN"])
            interval = max(5.0, interval)
            print(f"\n等待 {interval:.1f} 秒后进行下一次测试...")
            time.sleep(interval)
    
    # 统计结果
    print()
    print("=" * 60)
    print("📊 测试结果统计")
    print("=" * 60)
    print(f"总尝试次数: {attempt_count}")
    print(f"成功点赞: {success_count} 次")
    print(f"点赞失败: {fail_count} 次")
    print(f"使用 IP 数: {len(used_ips)}/{total_nodes}")
    if attempt_count > 0:
        print(f"成功率: {success_count / attempt_count * 100:.1f}%")
    print()
    
    # 判断停止原因
    if LIKE_CONFIG["TARGET_LIKES"] > 0 and success_count >= LIKE_CONFIG["TARGET_LIKES"]:
        print(f"✅ 完成原因: 达到目标有效点赞数 ({success_count}/{LIKE_CONFIG['TARGET_LIKES']})")
    elif len(used_ips) >= total_nodes:
        print(f"⚠️ 完成原因: 所有节点 IP 已用完 ({len(used_ips)}/{total_nodes})")
        if LIKE_CONFIG["TARGET_LIKES"] > 0 and success_count < LIKE_CONFIG["TARGET_LIKES"]:
            print(f"   未达到目标点赞数 ({success_count}/{LIKE_CONFIG['TARGET_LIKES']})")
            print(f"   建议: 增加更多代理节点或降低 TARGET_LIKES")
    elif attempt_count >= LIKE_CONFIG["MAX_ATTEMPTS"]:
        print(f"⏹️ 完成原因: 达到最大尝试次数 ({attempt_count}/{LIKE_CONFIG['MAX_ATTEMPTS']})")
    print()
    
    if success_count > 1:
        print("🎉 太好了！切换 IP 后可以重复点赞！")
        if LIKE_CONFIG["TARGET_LIKES"] > 0:
            print(f"提示: 你可以设置 TARGET_LIKES 参数来控制目标点赞数")
    elif success_count == 1:
        print("⚠️ 只有第一次成功，后续失败。可能原因：")
        print("   1. 后端按其他维度去重（账号、设备指纹等）")
        print("   2. 需要清除更多浏览器数据")
        print("   3. 网站有更严格的反作弊机制")
    else:
        print("❌ 所有测试都失败，建议：")
        print("   1. 手动访问页面，确认点赞按钮是否正常")
        print("   2. 检查元素 ID 是否正确")
        print("   3. 查看 DevTools Network 面板，确认请求")
    print()

if __name__ == "__main__":
    test_like_with_ip_rotation()

