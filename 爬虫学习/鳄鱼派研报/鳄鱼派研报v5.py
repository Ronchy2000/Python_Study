import os
import re
import json
import time
import pathlib
from typing import Optional, Dict, Any, Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

"""
鳄鱼派研报 v5

更新要点：
- 必须使用 Selenium：直接请求返回的是 SPA 壳，本版全程用无头 Chrome 渲染页面。
- 文章保存为 Markdown，并按分类归档到"文章md/全部研报｜宏观分析｜行业分析"。
- 保存排版更易读:保留标题层级、列表、粗体、链接、图片等常见格式。
- 跳过已下载文章：使用 index.json 记录 article_id，重复运行时自动跳过。
- 继续同时输出调试 HTML（方便排查结构变动）。
- 🆕 增量探测：在 index.json 中记录 last_probed_id，下次从该位置继续探测。
- 🆕 增量下载：只下载新发现的文章，避免重复工作。
- 🆕 智能恢复：即使中断，下次运行也能自动继续。
"""


BASE_URL = "http://h5.2025eyp.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def create_directory(directory: str) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r"[\\/*?:\"<>|]", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def extract_date_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})", text)
    if not m:
        return None
    y, mo, d = re.split(r"[.\-/]", m.group(1))
    return f"{int(y):04d}.{int(mo):02d}.{int(d):02d}"


def html_to_markdown(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    for br in soup.find_all(["br", "hr"]):
        br.replace_with("\n")

    def convert(node) -> str:
        if isinstance(node, str):
            return node
        name = node.name.lower() if node.name else ""

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            prefix = "#" * level
            inner = "".join(convert(child) for child in node.children).strip()
            return f"\n{prefix} {inner}\n\n"

        if name == "p":
            inner = "".join(convert(child) for child in node.children).strip()
            return f"{inner}\n\n" if inner else ""

        if name in {"ul", "ol"}:
            items = []
            for li in node.find_all("li", recursive=False):
                content = "".join(convert(child) for child in li.children).strip()
                if content:
                    items.append(f"- {content}")
            return ("\n".join(items) + "\n\n") if items else ""

        if name == "li":
            inner = "".join(convert(child) for child in node.children).strip()
            return f"- {inner}\n"

        if name in {"strong", "b"}:
            inner = "".join(convert(child) for child in node.children)
            return f"**{inner}**"
        if name in {"em", "i"}:
            inner = "".join(convert(child) for child in node.children)
            return f"*{inner}*"

        if name == "a":
            text = "".join(convert(child) for child in node.children) or node.get("href", "")
            href = node.get("href", "")
            return f"[{text}]({href})" if href else text

        if name == "img":
            alt = node.get("alt", "")
            src = node.get("src", "")
            return f"![{alt}]({src})" if src else ""

        inner = "".join(convert(child) for child in node.children)
        return inner

    md = "".join(convert(child) for child in soup.body.children) if soup.body else convert(soup)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def detect_category(title: str, text_preview: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit in {"全部研报", "宏观分析", "行业分析"}:
            return explicit

    candidates = "\n".join(filter(None, [explicit or "", title or "", text_preview or ""]))
    for key in ["宏观分析", "行业分析", "全部研报"]:
        if key and key in candidates:
            return key
    if re.search(r"宏观|大势|总量", candidates):
        return "宏观分析"
    if re.search(r"行业|产业|板块", candidates):
        return "行业分析"
    return "全部研报"


def normalize_content_html(html: str, base_url: str) -> str:
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            img["src"] = urljoin(base_url, src)
        if img.has_attr("data-src"):
            del img["data-src"]

    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            a["href"] = urljoin(base_url, href)

    return str(soup)


def build_filename(info: Dict[str, Any]) -> str:
    title = info.get("title") or f"article_{info.get('id')}"
    date = info.get("date")
    base = f"{date}-{title}" if date else title
    safe = sanitize_filename(base)
    if not safe:
        safe = f"article_{info.get('id')}"
    return f"{safe}.md"


def ensure_category_dir(root: str, category: str) -> str:
    if category not in {"全部研报", "宏观分析", "行业分析"}:
        category = "全部研报"
    return create_directory(os.path.join(root, category))


def read_saved_index(index_path: str) -> Dict[str, Any]:
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                # 确保包含必要的字段
                if "saved_ids" not in data:
                    data["saved_ids"] = []
                if "last_probed_id" not in data:
                    data["last_probed_id"] = 0
                return data
        except Exception:
            return {"saved_ids": [], "last_probed_id": 0}
    return {"saved_ids": [], "last_probed_id": 0}


def write_saved_index(index_path: str, info: Dict[str, Any]) -> None:
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(info, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, index_path)


def article_already_saved(article_id: int, saved_index: Dict[str, Any]) -> bool:
    return int(article_id) in set(saved_index.get("saved_ids", []))


def add_saved_id(article_id: int, saved_index: Dict[str, Any]) -> None:
    ids = set(saved_index.get("saved_ids", []))
    ids.add(int(article_id))
    saved_index["saved_ids"] = sorted(ids)


def update_last_probed_id(probed_id: int, saved_index: Dict[str, Any]) -> None:
    """更新最后探测的 ID"""
    saved_index["last_probed_id"] = int(probed_id)


def save_markdown(info: Dict[str, Any], root_md: str) -> str:
    category_dir = ensure_category_dir(root_md, info.get("category") or "全部研报")
    filename = build_filename(info)
    path = os.path.join(category_dir, filename)

    lines = [f"# {info.get('title')}"]
    meta = [
        f"- 分类: {info.get('category')}",
        f"- 日期: {info.get('date') or '未知'}",
        f"- 文章ID: {info.get('id')}",
        f"- 来源: {info.get('source_url')}",
    ]
    lines.extend(meta)
    lines.append("")
    lines.append("---")
    lines.append("")

    brief = info.get("brief")
    if brief:
        lines.append(f"> {brief.strip()}")
        lines.append("")

    body = info.get("content_md") or ""
    lines.append(body.rstrip())

    content = "\n".join(lines).rstrip() + "\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)
    return path


def _find_first_text(driver, selectors: Iterable[str]) -> str:
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
        except Exception:
            continue
        for element in elements:
            text = element.text.strip()
            if text:
                return text
    return ""


def _find_first_html(driver, selectors: Iterable[str]) -> str:
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
        except Exception:
            continue
        for element in elements:
            html = element.get_attribute("innerHTML") or ""
            html = html.strip()
            if html:
                return html
    return ""


class SeleniumArticleFetcher:
    def __init__(self, base_url: str, debug_dir: Optional[str] = None, wait_timeout: int = 20):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={USER_AGENT}")

        self.driver = webdriver.Chrome(options=options)
        self.base_url = base_url.rstrip("/")
        self.wait_timeout = wait_timeout
        self.debug_dir = debug_dir
        if debug_dir:
            create_directory(debug_dir)

    def close(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass

    def _wait_for_content(self) -> None:
        try:
            WebDriverWait(self.driver, self.wait_timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            WebDriverWait(self.driver, self.wait_timeout).until(
                lambda d: self._has_meaningful_text(d)
            )
        except TimeoutException:
            pass

    @staticmethod
    def _has_meaningful_text(driver) -> bool:
        try:
            article = driver.find_element(By.CSS_SELECTOR, ".article")
            if article.text and len(article.text.strip()) > 40:
                return True
        except NoSuchElementException:
            pass
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            return len(body.text.strip()) > 200
        except NoSuchElementException:
            return False

    def fetch(self, article_id: int) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/articles/{article_id}"
        try:
            self.driver.get(url)
            self._wait_for_content()
        except Exception:
            return None

        page_source = self.driver.page_source
        if self.debug_dir:
            debug_path = os.path.join(self.debug_dir, f"article_{article_id}_selenium_raw.html")
            with open(debug_path, "w", encoding="utf-8") as fp:
                fp.write(page_source)

        if "找不到页面" in page_source or "404" in page_source:
            return None

        title = _find_first_text(
            self.driver,
            [".article .title", ".article-title", "h1.title", "h1"],
        )

        category_raw = _find_first_text(
            self.driver,
            [".article .tags", ".article .category", ".article .cate"],
        )

        date_text = _find_first_text(
            self.driver,
            [".article .time", ".article .date", "time"],
        )

        brief = _find_first_text(
            self.driver,
            [".article .brief", ".article .summary"],
        )

        content_html = _find_first_html(
            self.driver,
            [
                ".article .md-editor-preview",
                ".article .content",
                ".article-content",
                "article",
            ],
        )

        if not content_html:
            content_text = _find_first_text(
                self.driver,
                [
                    ".article",
                    "body",
                ],
            )
        else:
            content_text = ""

        content_html = normalize_content_html(content_html, self.base_url)

        if content_html:
            content_md = html_to_markdown(content_html)
        else:
            content_md = content_text.strip()

        if not title and len(content_md) < 80:
            return None

        date_fmt = extract_date_from_text(date_text) or extract_date_from_text(content_md)
        category = detect_category(title, content_md[:200], category_raw)

        return {
            "id": article_id,
            "title": title or f"article_{article_id}",
            "category": category,
            "date": date_fmt,
            "brief": brief,
            "content_md": content_md,
            "source_url": url,
        }


def find_latest_article_id(fetcher, start_from: int = 1, initial_step: int = 50, max_consecutive_fails: int = 5) -> int:
    """
    自动发现最新文章的 ID - 改进版（两阶段探测）
    
    策略：
    1. 粗略探测：用大步长快速找到大致边界
    2. 精确定位：用小步长精确找到最后一篇文章
    """
    print(f"\n🔍 自动探测最新文章 ID...")
    print(f"   起始 ID: {start_from}")
    print(f"   阶段1: 粗略探测（步长 {initial_step}）")
    
    # 阶段1：粗略探测
    current_id = start_from
    last_valid_id = start_from - 1  # 初始化为起始ID的前一个
    consecutive_fails = 0
    
    # 先检查起始ID是否有效
    try:
        info = fetcher.fetch(start_from)
        if info:
            last_valid_id = start_from
            print(f"  ✅ ID {start_from}: {info['title'][:40]}... ({info['date']})")
        else:
            print(f"  ❌ ID {start_from}: 未找到")
    except Exception:
        print(f"  ❌ ID {start_from}: 访问失败")
    
    # 继续大步长探测
    current_id = start_from + initial_step
    while consecutive_fails < max_consecutive_fails:
        try:
            info = fetcher.fetch(current_id)
            if info:
                last_valid_id = current_id
                consecutive_fails = 0
                print(f"  ✅ ID {current_id}: {info['title'][:40]}... ({info['date']})")
            else:
                consecutive_fails += 1
                print(f"  ❌ ID {current_id}: 未找到 ({consecutive_fails}/{max_consecutive_fails})")
        except Exception:
            consecutive_fails += 1
            print(f"  ❌ ID {current_id}: 访问失败 ({consecutive_fails}/{max_consecutive_fails})")
        
        current_id += initial_step
    
    # 阶段2：精确定位（从最后一个有效ID开始，小步长向前）
    print(f"\n   阶段2: 精确定位（从 ID {last_valid_id + 1} 开始，步长 5）")
    precise_step = 5
    consecutive_fails = 0
    
    current_id = last_valid_id + 1
    while consecutive_fails < 10:  # 连续10个不存在就停止
        try:
            info = fetcher.fetch(current_id)
            if info:
                last_valid_id = current_id
                consecutive_fails = 0
                print(f"  ✅ ID {current_id}: {info['title'][:40]}... ({info['date']})")
            else:
                consecutive_fails += 1
        except Exception:
            consecutive_fails += 1
        
        current_id += precise_step
    
    print(f"\n✅ 探测完成！最新文章 ID: {last_valid_id}")
    
    return last_valid_id


def crawl_range(start_id: int, end_id: int, out_root: str, pause_sec: float = 1.2) -> None:
    root_dir = create_directory(out_root)
    index_path = os.path.join(root_dir, "index.json")
    saved_index = read_saved_index(index_path)

    debug_dir = os.path.join(os.path.dirname(root_dir), "data")

    fetcher = SeleniumArticleFetcher(BASE_URL, debug_dir=debug_dir)

    total = end_id - start_id + 1
    success = 0
    skipped = 0
    failed = 0

    try:
        for article_id in range(start_id, end_id + 1):
            try:
                if article_already_saved(article_id, saved_index):
                    print(f"[{article_id}] 已在索引中，跳过")
                    skipped += 1
                    continue

                info = fetcher.fetch(article_id)
                if info is None:
                    print(f"[{article_id}] 未获取到内容，跳过")
                    failed += 1
                    continue

                md_path = save_markdown(info, root_dir)
                add_saved_id(article_id, saved_index)
                write_saved_index(index_path, saved_index)
                print(f"[{article_id}] 已保存: {md_path}")
                success += 1

                time.sleep(pause_sec)

            except Exception as exc:
                print(f"[{article_id}] 处理失败: {type(exc).__name__}: {str(exc)[:160]}")
                failed += 1

            done = article_id - start_id + 1
            print(f"进度: {done}/{total} | 成功: {success}, 跳过: {skipped}, 失败: {failed}")

    finally:
        fetcher.close()

    print(f"完成: 尝试 {total}, 成功 {success}, 跳过 {skipped}, 失败 {failed}")


def main():
    project_root = pathlib.Path(__file__).resolve().parent
    out_root = project_root / "鳄鱼派研报内容" / "文章md"
    index_path = out_root / "index.json"
    
    # 确保目录存在
    create_directory(str(out_root))
    
    # 读取已保存的索引
    saved_index = read_saved_index(str(index_path))
    saved_ids = saved_index.get("saved_ids", [])
    max_saved_id = max(saved_ids) if saved_ids else 0
    last_probed_id = saved_index.get("last_probed_id", 0)
    
    print(f"📊 当前状态:")
    print(f"   已保存文章: {len(saved_ids)} 篇")
    print(f"   最大已保存 ID: {max_saved_id}")
    print(f"   上次探测到的最新 ID: {last_probed_id}")
    
    # 决定从哪里开始探测
    if last_probed_id > 0:
        # 从上次探测的位置继续
        start_probe = last_probed_id + 1
        print(f"\n🔄 增量模式：从 ID {start_probe} 开始探测新文章")
    else:
        # 首次运行，从最大已保存ID开始
        start_probe = max(max_saved_id + 1, 1)
        print(f"\n🆕 首次探测：从 ID {start_probe} 开始")
    
    # 自动发现最新文章 ID
    print("\n🚀 启动自动探测...")
    fetcher = SeleniumArticleFetcher(BASE_URL)
    try:
        latest_id = find_latest_article_id(fetcher, start_from=start_probe, initial_step=50)
        
        # 更新探测记录
        if latest_id >= start_probe:
            update_last_probed_id(latest_id, saved_index)
            write_saved_index(str(index_path), saved_index)
            print(f"\n💾 已更新探测记录：最新 ID = {latest_id}")
        
        end_id = latest_id + 10  # 留一些余量
    finally:
        fetcher.close()
    
    # 增量下载：只下载未保存的文章
    if max_saved_id > 0:
        # 优先下载新发现的文章
        start_id = max_saved_id + 1
        print(f"\n📥 增量下载模式：下载 ID {start_id} - {end_id} 的新文章")
    else:
        # 首次运行，从头开始
        start_id = 1
        print(f"\n📥 完整下载模式：下载 ID {start_id} - {end_id} 的所有文章")
    
    if start_id <= end_id:
        crawl_range(start_id, end_id, str(out_root), pause_sec=1.0)
    else:
        print(f"\n✅ 没有新文章需要下载！")
    
    # 显示最终统计
    final_index = read_saved_index(str(index_path))
    final_count = len(final_index.get("saved_ids", []))
    print(f"\n🎉 任务完成！")
    print(f"   当前共有 {final_count} 篇文章")
    print(f"   最新探测 ID: {final_index.get('last_probed_id', 0)}")


if __name__ == "__main__":
    main()

