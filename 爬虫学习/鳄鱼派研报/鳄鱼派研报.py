# 请求连接：http://h5.2025eyp.com/

import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import urljoin

# 创建保存文件的目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# 清理文件名，移除非法字符
def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)


# 获取网页内容
def fetch_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'http://h5.2025eyp.com/'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 如果状态码不是200，将引发HTTPError异常

        # 打印状态码和部分HTML内容以进行调试
        print(f"请求成功，状态码: {response.status_code}")
        print("HTML内容预览 (前500字符):")
        print(response.text[:500])

        # 可选：将完整HTML保存到文件以便更详细地检查
        with open('鳄鱼派研报内容/data/webpage_debug.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("完整HTML已保存至 webpage_debug.html")

        return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


# 解析帖子列表页面，获取所有帖子的链接
def parse_post_list(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    post_links = []

    # 需要根据实际网站结构调整选择器
    # 这里假设帖子链接在 <a> 标签中，并且有特定的类或属性
    for link in soup.select('a.post-link'):  # 替换为实际的CSS选择器
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)
            title = link.get_text().strip()
            post_links.append((full_url, title))

    return post_links


# 解析帖子内容
def parse_post_content(html):
    soup = BeautifulSoup(html, 'html.parser')

    # 需要根据实际网站结构调整选择器
    # 假设内容在一个带有特定类的div中
    content_div = soup.select_one('div.post-content')  # 替换为实际的CSS选择器

    if content_div:
        return content_div.get_text(separator='\n', strip=True)
    return "内容解析失败"


# 保存内容到文件
def save_to_file(content, filename, directory):
    filepath = os.path.join(directory, f"{clean_filename(filename)}.txt")
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"成功保存文件: {filepath}")
        return True
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def main():
    base_url = "http://h5.2025eyp.com/"
    output_dir = create_directory("鳄鱼派研报内容")

    # 首先测试是否能成功获取主页
    print("尝试获取网站主页...")
    list_html = fetch_page(base_url)

    if not list_html:
        print("获取网站内容失败")
        return

    # 打印HTML后暂停，让用户检查HTML结构
    input("按Enter键继续解析帖子列表...")

    # 接下来进行解析处理，但先打印解析前的HTML结构提示
    print("\n请根据保存的HTML分析并更新以下选择器：")
    print("1. 在parse_post_list函数中将'a.post-link'改为正确的选择器")
    print("2. 在parse_post_content函数中将'div.post-content'改为正确的选择器")

    choice = input("是否继续执行帖子列表解析? (y/n): ")
    if choice.lower() != 'y':
        return

    # 解析帖子列表，获取所有帖子链接
    post_links = parse_post_list(list_html, base_url)

    if not post_links:
        print("未找到任何帖子链接，请检查选择器是否正确")
        return

    print(f"找到 {len(post_links)} 个帖子")
    print("找到的帖子链接:")
    for i, (url, title) in enumerate(post_links[:5], 1):
        print(f"{i}. {title}: {url}")

    if len(post_links) > 5:
        print(f"...以及其他 {len(post_links) - 5} 个帖子")

    choice = input("是否继续处理帖子内容? (y/n): ")
    if choice.lower() != 'y':
        return

    # 遍历每个帖子链接，获取内容并保存
    for i, (post_url, post_title) in enumerate(post_links, 1):
        print(f"正在处理第 {i}/{len(post_links)} 个帖子: {post_title}")

        post_html = fetch_page(post_url)
        if post_html:
            content = parse_post_content(post_html)
            # 添加标题和URL作为内容的一部分
            full_content = f"标题: {post_title}\n链接: {post_url}\n\n{content}"
            save_to_file(full_content, f"{i:03d}_{post_title}", output_dir)
        else:
            print(f"获取帖子内容失败: {post_url}")

        # 间隔一段时间，避免请求过于频繁
        time.sleep(2)

    print("所有帖子处理完成")

if __name__ == "__main__":
    main()