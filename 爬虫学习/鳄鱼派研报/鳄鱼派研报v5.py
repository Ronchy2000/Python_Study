from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import re

"""
该版本只能保存多个文件的title 和 内容，
+ 保存的文件加入日期命名。
+ fix bug: 修改文件命名日期重复
"""

# 创建保存文件的目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def fetch_article_with_selenium(article_number):
    url = f"http://h5.2025eyp.com/articles/{article_number}"
    print(f"[{article_number}] 尝试获取文章: {url}")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    # Windows Chrome user-agent
    # options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

    # iPhone Safari user-agent
    # options.add_argument("--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1")

    # Android Chrome user-agent
    # options.add_argument("--user-agent=Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36")

    # Edge on Windows user-agent
    # options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0")

    # Firefox on macOS user-agent
    # options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        # 等待页面加载
        print(f"[{article_number}] 等待页面加载...")
        time.sleep(5)  # 增加等待时间确保页面完全加载

        # 检查是否有"找不到页面"的提示
        page_text = driver.page_source
        if "找不到页面" in page_text or "404" in page_text:
            print(f"[{article_number}] 文章不存在，跳过")
            return None

        # 保存HTML用于调试
        debug_dir = create_directory("鳄鱼派研报内容/data")
        with open(f'{debug_dir}/article_{article_number}_selenium_raw.html', 'w', encoding='utf-8') as f:
            f.write(page_text)

        # 尝试寻找标题和内容
        title = ""
        content = ""

        # 从v2中采用的选择器方法
        selectors = [
            {"title": ".article-title", "content": ".article-content"},
            {"title": "h1", "content": ".content"},
            {"title": ".title", "content": ".body"}
        ]

        for selector in selectors:
            try:
                # 尝试找到标题
                title_elements = driver.find_elements(By.CSS_SELECTOR, selector["title"])
                if title_elements and title_elements[0].text.strip():
                    title = title_elements[0].text.strip()
                    print(f"[{article_number}] 找到标题: {title}")

                # 尝试找到内容
                content_elements = driver.find_elements(By.CSS_SELECTOR, selector["content"])
                if content_elements and content_elements[0].text.strip():
                    content = content_elements[0].text.strip()
                    print(f"[{article_number}] 找到内容 ({len(content)} 字符)")
                    if title and content:
                        break
            except Exception as e:
                print(f"[{article_number}] 尝试选择器 {selector} 失败: {e}")
                continue

        # 如果还是没找到内容，尝试获取所有文本
        if not content:
            print(f"[{article_number}] 未能通过选择器找到内容，尝试获取页面所有文本...")
            try:
                content = driver.find_element(By.TAG_NAME, "body").text
                if len(content) > 100:  # 确保有足够的文本
                    print(f"[{article_number}] 使用body标签提取到内容: {len(content)} 字符")
                else:
                    content = ""
            except Exception as e:
                print(f"[{article_number}] 获取body文本失败: {e}")

        # 如果找不到内容和标题，尝试另一种方式
        if not content and not title:
            body_text = driver.find_element(By.TAG_NAME, "body").text
            if len(body_text) > 100:  # 确保有足够的文本
                content = body_text
                # 尝试从内容中提取标题（假设第一行是标题）
                title = content.split('\n')[0]
                print(f"[{article_number}] 使用备用方法提取内容: {len(content)} 字符")

        # After successfully finding content, extract the date
        date_str = ""
        if content:
            # Look for date pattern in the content (YYYY.MM.DD HH:MM format)
            date_match = re.search(r'(\d{4}\.\d{2}\.\d{2})', content)
            if date_match:
                date_str = date_match.group(1).replace(' ', '_').replace(':', '')
                print(f"[{article_number}] 提取到日期: {date_str}")

        if title:  # 如果至少找到标题
            if not content:
                content = "内容未能提取"
                print(f"[{article_number}] 警告: 找到标题但未找到内容")

            return {
                "title": title,
                "content": content,
                "article_number": article_number,
                "date": date_str  # Add the extracted date to the returned data
            }
        else:
            print(f"[{article_number}] 无法提取标题和内容")
            return None

    except Exception as e:
        print(f"[{article_number}] 抓取错误: {type(e).__name__}: {str(e)[:100]}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
                print(f"[{article_number}] WebDriver已关闭")
            except:
                pass


def sanitize_filename(filename):
    # 移除文件名中的非法字符
    return re.sub(r'[\\/*?:"<>|]', '', filename).strip()

def save_article(article_data, output_dir):
    if not article_data or not article_data.get("title"):
        return False

    title = article_data["title"]
    content = article_data.get("content", "")
    article_number = article_data["article_number"]
    date_str = article_data.get("date", "")  # Get the extracted date

    # Check if title already starts with the date to avoid repetition
    if date_str and title.startswith(date_str):
        filename = sanitize_filename(title)
    # Create filename with date prefix if available and not already in title
    elif date_str:
        filename = sanitize_filename(f"{date_str}-{title}")
    else:
        filename = sanitize_filename(title)

    if not filename:
        filename = f"article_{article_number}"

    # Save the file
    with open(f'{output_dir}/{filename}.txt', 'w', encoding='utf-8') as f:
        f.write(f"标题: {title}\n\n")
        f.write(content)

    print(f"[{article_number}] 文章已保存: {filename}.txt")
    return True


def main():
    output_dir = create_directory("鳄鱼派研报内容/文章")

    # 文章编号范围
    start_number = 1
    end_number = 149

    success_count = 0
    fail_count = 0

    for article_number in range(start_number, end_number + 1):
        try:
            # 获取并保存文章
            article_data = fetch_article_with_selenium(article_number)

            if article_data and save_article(article_data, output_dir):
                success_count += 1
            else:
                fail_count += 1

            # 休息一下，避免请求过于频繁
            time.sleep(2)

        except Exception as e:
            print(f"[{article_number}] 处理失败: {str(e)[:100]}")
            fail_count += 1

        print(f"当前进度: {article_number}/{end_number}, 成功: {success_count}, 失败: {fail_count}")

    print(f"\n全部完成! 总共尝试: {end_number - start_number + 1}, 成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()