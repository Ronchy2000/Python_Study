from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os


# 创建保存文件的目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def fetch_with_selenium(article_number):
    url = f"http://h5.2025eyp.com/articles/{article_number}"
    print(f"使用Selenium获取文章: {url}")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

    try:
        # 使用内置的ChromeDriver
        print("初始化WebDriver...")
        driver = webdriver.Chrome(options=options)

        print(f"访问URL: {url}")
        driver.get(url)

        # 先等待页面加载完成
        print("等待页面加载...")
        time.sleep(5)  # 给JavaScript足够的时间加载内容

        # 获取加载后的HTML
        html_content = driver.page_source
        print("已获取HTML内容")

        # 保存原始HTML以便检查
        debug_dir = create_directory("鳄鱼派研报内容/data")
        with open(f'{debug_dir}/article_{article_number}_selenium_raw.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"原始HTML已保存至 article_{article_number}_selenium_raw.html")

        # 尝试不同的选择器
        title = ""
        content = ""

        try:
            # 尝试寻找标题和内容
            selectors = [
                {"title": ".article-title", "content": ".article-content"},
                {"title": "h1", "content": ".content"},
                {"title": ".title", "content": ".body"}
            ]

            for selector in selectors:
                try:
                    # 尝试找到标题
                    title_elements = driver.find_elements(By.CSS_SELECTOR, selector["title"])
                    if title_elements:
                        title = title_elements[0].text
                        print(f"找到标题: {title}")

                    # 尝试找到内容
                    content_elements = driver.find_elements(By.CSS_SELECTOR, selector["content"])
                    if content_elements:
                        content = content_elements[0].text
                        print(f"找到内容 ({len(content)} 字符)")
                        break
                except Exception as e:
                    print(f"尝试选择器 {selector} 失败: {e}")
                    continue

            # 如果还是没找到，尝试获取所有文本
            if not content:
                print("未能通过选择器找到内容，尝试获取页面所有文本...")
                content = driver.find_element(By.TAG_NAME, "body").text

        except Exception as e:
            print(f"内容提取错误: {e}")

        return {
            "title": title,
            "content": content,
            "html": html_content
        }
    except Exception as e:
        print(f"Selenium错误: {type(e).__name__}: {str(e)}")
    finally:
        try:
            driver.quit()
            print("WebDriver已关闭")
        except:
            pass

    return None


def main():
    article_number = 94

    # 使用Selenium获取文章
    data = fetch_with_selenium(article_number)

    if data and (data["title"] or data["content"]):
        # 保存文章内容
        output_dir = create_directory("鳄鱼派研报内容")
        with open(f'{output_dir}/article_{article_number}.txt', 'w', encoding='utf-8') as f:
            f.write(f"标题: {data['title']}\n\n")
            f.write(data['content'])
        print(f"文章已保存至 article_{article_number}.txt")
    else:
        print("无法获取文章内容")


if __name__ == "__main__":
    main()