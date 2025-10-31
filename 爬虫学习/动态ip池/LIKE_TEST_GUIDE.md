# 点赞功能测试指南

## 目标

测试切换 IP 后是否能够重复点赞（绕过"每个用户只能点一次"的限制）。

---

## 测试流程

### 步骤 1：调试元素定位

运行调试脚本，确认页面元素：

```bash
python debug_like_element.py
```

这个脚本会：
- 自动查找点赞按钮和计数元素
- 显示当前的 localStorage 和 Cookies
- 打开浏览器让你手动观察

**注意观察：**
1. 点赞按钮的 ID 是否正确
2. 点赞计数的 ID 是否正确
3. 点击后 Network 面板是否有请求
4. localStorage 或 Cookie 中是否有"已赞"标记

### 步骤 2：调整配置

根据调试结果，编辑 `selenium_like_test.py` 中的配置：

```python
LIKE_CONFIG = {
    "URL": "你的目标页面",
    "MAX_LIKES": 10,
    "HEADLESS": False,  # 建议先用 False 观察
    
    # 根据调试结果调整这两个 ID
    "LIKE_BUTTON_ID": "_parise_imgobj_u6",
    "LIKE_COUNT_ID": "_parise_obj_u6",
}
```

### 步骤 3：测试点赞

运行测试脚本：

```bash
python selenium_like_test.py
```

脚本会：
1. 切换到新的代理节点
2. 创建全新的浏览器实例
3. 清除所有 cookies 和 localStorage
4. 访问页面并点击点赞
5. 记录结果并统计成功率

---

## 可能的结果

### 结果 1：点赞成功，数字增加 ✅

```
[测试 #1]
🔄 切换节点: 日本节点
🌍 当前 IP: 156.246.92.93
👆 已点击点赞按钮
✅ 点赞成功！54 → 55

[测试 #2]
🔄 切换节点: 新加坡节点
🌍 当前 IP: 185.234.56.78
👆 已点击点赞按钮
✅ 点赞成功！55 → 56
```

**说明：** 网站只按 IP 或前端存储去重，切换 IP + 清除存储即可绕过

**下一步：** 可以批量运行，有效刷赞！

---

### 结果 2：第一次成功，后续失败 ⚠️

```
[测试 #1]
✅ 点赞成功！54 → 55

[测试 #2]
❌ 点赞失败，数量未变化: 55

[测试 #3]
❌ 点赞失败，数量未变化: 55
```

**可能原因：**
1. 后端按其他维度去重（账号、设备指纹、Canvas 指纹等）
2. 需要登录账号才能点赞
3. 清除存储不完整

**调试方法：**
- 设置 `HEADLESS: False`
- 打开 DevTools → Network
- 观察第二次点击是否发送请求
- 查看请求响应内容

---

### 结果 3：所有测试都失败 ❌

```
[测试 #1]
❌ 点击失败: Message: no such element

[测试 #2]
❌ 点击失败: Message: element not interactable
```

**可能原因：**
1. 元素 ID 不正确
2. 页面结构发生变化
3. 元素被 iframe 包裹
4. 需要先登录

**解决方法：**
- 运行 `debug_like_element.py` 重新确认元素
- 检查是否在 iframe 中
- 尝试用 XPath 定位

---

## 高级技巧

### 技巧 1：更彻底地清除浏览器数据

如果清除 localStorage 和 Cookie 还不够，可以：

```python
def clear_all_storage(driver):
    """更彻底的清除"""
    # 清除所有存储
    driver.execute_script("window.localStorage.clear();")
    driver.execute_script("window.sessionStorage.clear();")
    driver.delete_all_cookies()
    
    # 清除 IndexedDB
    driver.execute_script("""
        indexedDB.databases().then(dbs => {
            dbs.forEach(db => indexedDB.deleteDatabase(db.name));
        });
    """)
    
    # 清除缓存
    driver.execute_cdp_cmd('Network.clearBrowserCache', {})
    driver.execute_cdp_cmd('Network.clearBrowserCookies', {})
```

### 技巧 2：使用隐私模式

```python
options.add_argument('--incognito')  # 隐私模式
```

### 技巧 3：监控 Network 请求

```python
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# 启用性能日志
caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'performance': 'ALL'}

driver = webdriver.Chrome(desired_capabilities=caps, options=options)

# 获取所有网络请求
logs = driver.get_log('performance')
for log in logs:
    if 'praise' in str(log).lower() or 'like' in str(log).lower():
        print(log)
```

### 技巧 4：等待 AJAX 完成

```python
from selenium.webdriver.support.ui import WebDriverWait

def wait_for_ajax(driver, timeout=10):
    """等待 jQuery AJAX 完成"""
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return jQuery.active == 0")
    )
```

---

## 常见去重机制

### 前端去重（容易绕过）

1. **localStorage/sessionStorage**
   - 存储键：`iscanclick_parise_obj_u6`
   - 绕过方法：清除存储 + 刷新页面

2. **Cookie**
   - 存储键：`liked_xxx`
   - 绕过方法：删除 Cookie

### 后端去重（较难绕过）

1. **IP 地址**
   - 绕过方法：切换代理节点 ✅

2. **登录账号**
   - 绕过方法：使用多个账号

3. **设备指纹**
   - 包括：Canvas 指纹、WebGL 指纹、字体列表等
   - 绕过方法：使用指纹伪装插件或修改浏览器配置

4. **User-Agent + IP 组合**
   - 绕过方法：随机 UA + 切换 IP ✅

---

## 调试清单

运行测试前，先检查：

- [ ] Mihomo 已启动且在 global 模式
- [ ] 代理池已加载可用节点
- [ ] 元素 ID 正确（运行 debug_like_element.py 确认）
- [ ] 页面可以正常访问
- [ ] 手动点赞功能正常

运行测试时：

- [ ] 第一次是否成功？
- [ ] 切换 IP 后是否成功？
- [ ] Network 面板是否有请求？
- [ ] 请求响应是什么？

---

## 推荐测试步骤

### 第 1 轮：手动测试（确认基础功能）

1. 设置 `HEADLESS: False`
2. 设置 `MAX_LIKES: 2`
3. 运行脚本，观察整个过程
4. 确认元素定位、点击、清除存储都正常

### 第 2 轮：小规模测试（验证切换 IP）

1. 设置 `MAX_LIKES: 5`
2. 观察是否每次都成功
3. 检查日志中的 IP 是否不同

### 第 3 轮：正式运行（批量点赞）

1. 设置 `HEADLESS: True`（节省资源）
2. 设置 `MAX_LIKES: 50` 或更多
3. 设置合理的间隔（如 15 秒）
4. 后台运行

---

## 注意事项

1. **合理控制频率**：避免过于频繁导致服务器压力
2. **遵守网站规则**：某些网站明确禁止刷赞行为
3. **监控成功率**：如果成功率下降，可能被检测到
4. **备份日志**：记录每次测试的结果，便于分析

---

**最后更新**: 2025-10-30

