import os
import re
import json
import time
import pathlib
from typing import Optional, Dict, Any, Iterable, List, Tuple
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

"""
鳄鱼派研报 v6

更新要点：
- 在 v5 的基础上，探测与下载全量改为增量式。
- 探测进度写回 index.json（复用 saved_ids / last_probed_id 等字段）。
- 记录缺失 ID，后续自动跳过，避免重复探测。
- 每次运行仅向前探测指定数量的 ID，速度更快。
- 探测阶段直接返回新文章内容，下载阶段无需重复请求。
- 🆕 并发探测：使用多线程并发请求，大幅提升探测速度。
- 🆕 下载状态跟踪：区分已探测和已下载，自动检测并修复未完成的下载。
"""


BASE_URL = "http://h5.2025eyp.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# 探测时的控制参数
PROBE_MAX_FETCHES = 80            # 单次运行最多实际请求多少篇
PROBE_CONSECUTIVE_MISS = 25       # 连续缺失多少个 ID 后认为没有新文章
MISSING_BUCKET_LIMIT = 800        # index.json 中最多保留多少个缺失 ID
PROBE_HISTORY_LIMIT = 20          # 历史探测记录条数上限
CONCURRENT_WORKERS = 5            # 并发探测的线程数


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


def ensure_index_defaults(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    data.setdefault("saved_ids", [])          # 已探测到的文章 ID
    data.setdefault("downloaded_ids", [])     # 已成功下载的文章 ID
    data.setdefault("missing_ids", [])
    data.setdefault("last_probed_id", 0)
    data.setdefault("next_probe_id", 1)
    data.setdefault("probe_history", [])
    return data


def read_saved_index(index_path: str) -> Dict[str, Any]:
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return ensure_index_defaults(data)
        except Exception:
            return ensure_index_defaults({})
    return ensure_index_defaults({})


def write_saved_index(index_path: str, info: Dict[str, Any]) -> None:
    # 过滤临时字段（以 "_" 开头的不写入磁盘）
    dump_ready = {k: v for k, v in info.items() if not str(k).startswith("_")}
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(dump_ready, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, index_path)


def article_already_saved(article_id: int, saved_index: Dict[str, Any]) -> bool:
    return int(article_id) in set(saved_index.get("saved_ids", []))


def add_saved_id(article_id: int, saved_index: Dict[str, Any]) -> None:
    ids = set(saved_index.get("saved_ids", []))
    ids.add(int(article_id))
    saved_index["saved_ids"] = sorted(ids)


def add_downloaded_id(article_id: int, saved_index: Dict[str, Any]) -> None:
    """标记文章已成功下载"""
    ids = set(saved_index.get("downloaded_ids", []))
    ids.add(int(article_id))
    saved_index["downloaded_ids"] = sorted(ids)


def article_downloaded(article_id: int, saved_index: Dict[str, Any]) -> bool:
    """检查文章是否已下载"""
    return int(article_id) in set(saved_index.get("downloaded_ids", []))


def record_missing_id(article_id: int, saved_index: Dict[str, Any], limit: int = MISSING_BUCKET_LIMIT) -> None:
    missing_ids = set(saved_index.get("missing_ids", []))
    missing_ids.add(int(article_id))
    saved_index["missing_ids"] = sorted(missing_ids)[-limit:]


def clear_missing_id(article_id: int, saved_index: Dict[str, Any]) -> None:
    missing_ids = set(saved_index.get("missing_ids", []))
    if int(article_id) in missing_ids:
        missing_ids.discard(int(article_id))
        saved_index["missing_ids"] = sorted(missing_ids)


def update_probe_history(saved_index: Dict[str, Any], start_id: int, stop_id: int, found_id: int) -> None:
    history = list(saved_index.get("probe_history", []))
    history.append({
        "start": int(start_id),
        "stop": int(stop_id),
        "found": int(found_id),
        "ts": int(time.time())
    })
    saved_index["probe_history"] = history[-PROBE_HISTORY_LIMIT:]


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


def _resolve_probe_start(saved_index: Dict[str, Any]) -> int:
    saved_ids = saved_index.get("saved_ids", [])
    max_saved_id = max(saved_ids) if saved_ids else 0
    next_cursor = int(saved_index.get("next_probe_id", 1))
    last_probed = int(saved_index.get("last_probed_id", 0))
    return max(1, max(max_saved_id, last_probed, next_cursor))


def _fetch_single_article(article_id: int, base_url: str, debug_dir: Optional[str] = None) -> Tuple[int, Optional[Dict[str, Any]]]:
    """单个文章的抓取任务，每个线程独立创建 driver"""
    fetcher = SeleniumArticleFetcher(base_url, debug_dir=debug_dir, wait_timeout=15)
    try:
        info = fetcher.fetch(article_id)
        return (article_id, info)
    finally:
        fetcher.close()


def probe_new_articles(saved_index: Dict[str, Any],
                       base_url: str = BASE_URL,
                       debug_dir: Optional[str] = None,
                       max_fetches: int = PROBE_MAX_FETCHES,
                       max_consecutive_missing: int = PROBE_CONSECUTIVE_MISS,
                       workers: int = CONCURRENT_WORKERS
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    并发增量探测，返回新的文章列表，并更新 saved_index。
    """
    saved_ids = set(saved_index.get("saved_ids", []))
    known_missing = set(saved_index.get("missing_ids", []))

    start_id = _resolve_probe_start(saved_index)
    
    # 生成待探测的 ID 列表
    probe_candidates = []
    current_id = start_id
    while len(probe_candidates) < max_fetches:
        if current_id not in known_missing or current_id >= saved_index.get("next_probe_id", current_id):
            probe_candidates.append(current_id)
        current_id += 1
    
    print(f"\n🚀 开始并发探测 ID {start_id} - {current_id - 1}（共 {len(probe_candidates)} 个请求，{workers} 线程）...")
    
    new_articles: List[Dict[str, Any]] = []
    results: Dict[int, Optional[Dict[str, Any]]] = {}
    last_found_id = int(saved_index.get("last_probed_id", 0))
    
    # 并发执行
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(_fetch_single_article, aid, base_url, debug_dir): aid
            for aid in probe_candidates
        }
        
        completed = 0
        for future in as_completed(future_to_id):
            article_id, info = future.result()
            results[article_id] = info
            completed += 1
            
            if info:
                clear_missing_id(article_id, saved_index)
                last_found_id = max(last_found_id, article_id)
                if article_id not in saved_ids:
                    new_articles.append(info)
                    print(f"  ✅ [{completed}/{len(probe_candidates)}] ID {article_id}: {info['title'][:40]}... ({info.get('date', 'N/A')})")
            else:
                record_missing_id(article_id, saved_index)
                print(f"  ❌ [{completed}/{len(probe_candidates)}] ID {article_id}: 未找到")
    
    # 检查连续缺失，决定下一次探测起点
    consecutive_missing = 0
    final_probe_id = start_id
    for check_id in range(start_id, current_id):
        if check_id in results:
            if results[check_id] is None:
                consecutive_missing += 1
                if consecutive_missing >= max_consecutive_missing:
                    final_probe_id = check_id + 1
                    break
            else:
                consecutive_missing = 0
                final_probe_id = check_id + 1
    
    saved_index["next_probe_id"] = final_probe_id
    saved_index["last_probed_id"] = last_found_id
    update_probe_history(saved_index, start_id, current_id - 1, last_found_id)
    saved_index["_last_probe_ids"] = probe_candidates  # 仅供调试展示

    return new_articles, saved_index


def download_new_articles(new_articles: List[Dict[str, Any]], saved_index: Dict[str, Any],
                          out_root: str, index_path: str) -> None:
    success = 0
    skipped = 0
    for info in new_articles:
        article_id = int(info.get("id"))
        if article_downloaded(article_id, saved_index):
            skipped += 1
            continue

        md_path = save_markdown(info, out_root)
        add_saved_id(article_id, saved_index)
        add_downloaded_id(article_id, saved_index)  # 标记为已下载
        write_saved_index(index_path, saved_index)
        print(f"[{article_id}] 已保存: {md_path}")
        success += 1

    if success == 0 and skipped == 0:
        print("没有新文章需要下载。")
    else:
        print(f"保存完成：新增 {success} 篇，跳过 {skipped} 篇已下载文章。")


def check_and_repair_downloads(saved_index: Dict[str, Any], out_root: str) -> List[int]:
    """
    检查已探测但未下载的文章，返回需要重新下载的 ID 列表
    """
    saved_ids = set(saved_index.get("saved_ids", []))
    downloaded_ids = set(saved_index.get("downloaded_ids", []))
    
    # 找出已探测但未下载的 ID
    pending_ids = saved_ids - downloaded_ids
    
    if pending_ids:
        print(f"\n⚠️  发现 {len(pending_ids)} 篇已探测但未下载的文章")
        print(f"   ID 列表: {sorted(list(pending_ids))[:10]}{'...' if len(pending_ids) > 10 else ''}")
        return sorted(list(pending_ids))
    
    return []


def main():
    project_root = pathlib.Path(__file__).resolve().parent
    out_root = project_root / "鳄鱼派研报内容" / "文章md"
    index_path = out_root / "index.json"

    create_directory(str(out_root))

    saved_index = read_saved_index(str(index_path))
    saved_ids = saved_index.get("saved_ids", [])
    downloaded_ids = saved_index.get("downloaded_ids", [])
    max_saved_id = max(saved_ids) if saved_ids else 0
    last_probed_id = saved_index.get("last_probed_id", 0)
    
    print("📊 当前状态:")
    print(f"   已探测文章: {len(saved_ids)} 篇")
    print(f"   已下载文章: {len(downloaded_ids)} 篇")
    print(f"   最大已保存 ID: {max_saved_id}")
    print(f"   上次探测到的最新 ID: {last_probed_id}")
    print(f"   下一次计划探测 ID: {saved_index.get('next_probe_id', 1)}")
    
    # 检查并修复未完成的下载
    pending_downloads = check_and_repair_downloads(saved_index, str(out_root))

    debug_dir = project_root / "鳄鱼派研报内容" / "data"

    # 如果有未完成的下载，优先处理
    if pending_downloads:
        print(f"\n🔧 开始修复未完成的下载（共 {len(pending_downloads)} 篇）...")
        repaired = 0
        failed = 0
        
        for idx, article_id in enumerate(pending_downloads, 1):
            try:
                # 重新获取文章内容
                fetcher = SeleniumArticleFetcher(BASE_URL, debug_dir=str(debug_dir))
                info = fetcher.fetch(article_id)
                fetcher.close()
                
                if info:
                    md_path = save_markdown(info, str(out_root))
                    add_downloaded_id(article_id, saved_index)
                    write_saved_index(str(index_path), saved_index)
                    print(f"  ✅ [{idx}/{len(pending_downloads)}] 修复 ID {article_id}: {info['title'][:30]}...")
                    repaired += 1
                else:
                    print(f"  ⚠️  [{idx}/{len(pending_downloads)}] ID {article_id}: 无法重新获取内容")
                    failed += 1
            except Exception as e:
                print(f"  ❌ [{idx}/{len(pending_downloads)}] ID {article_id}: 修复失败 - {e}")
                failed += 1
        
        print(f"\n✅ 修复完成：成功 {repaired} 篇，失败 {failed} 篇")
        print("=" * 60)

    # 继续正常的探测和下载流程
    try:
        new_articles, saved_index = probe_new_articles(
            saved_index, 
            base_url=BASE_URL,
            debug_dir=str(debug_dir)
        )
        write_saved_index(str(index_path), saved_index)
    except Exception as e:
        print(f"❌ 探测过程出错: {e}")
        return

    probed_ids = saved_index.get("_last_probe_ids", [])
    if probed_ids:
        print(f"\n🔍 本次探测共检查 {len(probed_ids)} 个 ID，范围 {probed_ids[0]} - {probed_ids[-1]}")
    else:
        print("\n🔍 本次探测未执行新的请求（可能全部命中已知缺失 ID）。")

    if new_articles:
        print(f"🎯 新发现 {len(new_articles)} 篇文章，开始保存...")
    else:
        print("🎯 本次未发现新文章。")

    download_new_articles(new_articles, saved_index, str(out_root), str(index_path))

    final_index = read_saved_index(str(index_path))
    final_saved = len(final_index.get("saved_ids", []))
    final_downloaded = len(final_index.get("downloaded_ids", []))
    print("\n🎉 任务完成！")
    print(f"   已探测文章: {final_saved} 篇")
    print(f"   已下载文章: {final_downloaded} 篇")
    print(f"   最新探测 ID: {final_index.get('last_probed_id', 0)}")
    print(f"   下一次探测将从 ID: {final_index.get('next_probe_id', 1)} 开始")
    
    # 最终检查
    if final_saved != final_downloaded:
        print(f"\n⚠️  注意：还有 {final_saved - final_downloaded} 篇文章未完成下载")


if __name__ == "__main__":
    main()
