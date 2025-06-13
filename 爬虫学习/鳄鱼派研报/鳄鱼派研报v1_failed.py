# 请求连接：http://h5.2025eyp.com/
# 请求连接：http://h5.2025eyp.com/

'''
爬不下来，他用的技术手段是：
I see the issue. The website at h5.2025eyp.com appears to be using a client-side rendering approach (likely Vue.js or React), and there's no direct API endpoint we can access.  The responses you're receiving are just HTML shells that don't contain the actual content. The content is loaded later by JavaScript after the page loads.


'''
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

def fetch_article(article_number):
    url = f"http://h5.2025eyp.com/articles/{article_number}"
    print(f"尝试获取文章: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'http://h5.2025eyp.com/'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 保存HTML到文件
        debug_dir = create_directory("鳄鱼派研报内容/data")
        with open(f'{debug_dir}/article_{article_number}.html', 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"状态码: {response.status_code}")
        print(f"文章HTML已保存至 article_{article_number}.html")

        return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

def check_api_endpoints(article_number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
    }

    # 尝试获取API数据
    api_url = f"http://h5.2025eyp.com/api/articles/{article_number}"
    print(f"尝试获取API数据: {api_url}")

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        print(f"API状态码: {response.status_code}")

        if response.status_code == 200:
            # 保存JSON响应
            debug_dir = create_directory("鳄鱼派研报内容/data")
            with open(f'{debug_dir}/article_{article_number}_api.json', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"API响应已保存至 article_{article_number}_api.json")

            # 打印部分响应内容
            print("API响应预览:")
            print(response.text[:300])
            return response.json()
    except Exception as e:
        print(f"API请求错误: {e}")

    return None

def main_test_article():
    output_dir = create_directory("鳄鱼派研报内容")
    article_number = 94

    # 尝试直接获取HTML
    article_html = fetch_article(article_number)

    # 尝试获取API数据
    article_data = check_api_endpoints(article_number)

    if article_data:
        print("\n成功通过API获取文章数据")
        print(f"文章标题: {article_data.get('title', '无标题')}")
    else:
        print("\n无法通过API获取文章数据")

if __name__ == "__main__":
    main_test_article()