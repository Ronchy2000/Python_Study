"""
独立的并发探测脚本 - 完整探测最新文章 ID

本脚本完全独立，不依赖 v5/v6，自己实现所有功能。
从 index.json 读取上次探测进度，使用并发方式快速探测新文章。
"""
import os
import re
import json
import pathlib
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# ============= 配置参数 =============
BASE_URL = "http://h5.2025eyp.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
PROBE_STEP = 5                    # 探测步长
MAX_PROBE_RANGE = 200             # 单次最多探测范围
CONCURRENT_WORKERS = 8            # 并发线程数
CONSECUTIVE_MISS_LIMIT = 50       # 连续缺失多少个后停止


# ============= 工具函数 =============
def read_index(index_path: str) -> Dict[str, Any]:
    """读取 index.json"""
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data.setdefault("saved_ids", [])
                data.setdefault("last_probed_id", 0)
                data.setdefault("missing_ids", [])
                return data
        except Exception:
            pass
    return {"saved_ids": [], "last_probed_id": 0, "missing_ids": []}


def write_index(index_path: str, data: Dict[str, Any]) -> None:
    """写入 index.json"""
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, index_path)


def extract_date_from_text(text: str) -> Optional[str]:
    """提取日期"""
    if not text:
        return None
    m = re.search(r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})", text)
    if not m:
        return None
    y, mo, d = re.split(r"[.\-/]", m.group(1))
    return f"{int(y):04d}.{int(mo):02d}.{int(d):02d}"


def _find_first_text(driver, selectors: List[str]) -> str:
    """查找第一个匹配的文本"""
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip()
                if text:
                    return text
        except Exception:
            continue
    return ""


# ============= 探测类 =============
class QuickArticleProbe:
    """快速文章探测器"""
    
    def __init__(self, base_url: str):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={USER_AGENT}")
        
        self.driver = webdriver.Chrome(options=options)
        self.base_url = base_url.rstrip("/")
    
    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass
    
    def probe(self, article_id: int) -> Optional[Dict[str, Any]]:
        """探测单篇文章是否存在，返回简要信息"""
        url = f"{self.base_url}/articles/{article_id}"
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
        except Exception:
            return None
        
        page_source = self.driver.page_source
        if "找不到页面" in page_source or "404" in page_source:
            return None
        
        # 快速提取关键信息
        title = _find_first_text(
            self.driver,
            [".article .title", ".article-title", "h1.title", "h1"]
        )
        
        date_text = _find_first_text(
            self.driver,
            [".article .time", ".article .date", "time"]
        )
        
        if not title:
            return None
        
        date_fmt = extract_date_from_text(date_text)
        
        return {
            "id": article_id,
            "title": title,
            "date": date_fmt,
            "url": url
        }


def _probe_single(article_id: int, base_url: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    """单个探测任务"""
    probe = QuickArticleProbe(base_url)
    try:
        result = probe.probe(article_id)
        return (article_id, result)
    finally:
        probe.close()


def concurrent_probe(start_id: int, end_id: int, step: int, workers: int) -> Dict[int, Optional[Dict[str, Any]]]:
    """并发探测一批 ID"""
    probe_ids = list(range(start_id, end_id + 1, step))
    results = {}
    
    print(f"🚀 并发探测 {len(probe_ids)} 个 ID（{workers} 线程）...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(_probe_single, aid, BASE_URL): aid
            for aid in probe_ids
        }
        
        completed = 0
        for future in as_completed(future_to_id):
            article_id, info = future.result()
            results[article_id] = info
            completed += 1
            
            if info:
                print(f"  ✅ [{completed}/{len(probe_ids)}] ID {article_id}: {info['title'][:40]}... ({info.get('date', 'N/A')})")
            else:
                print(f"  ❌ [{completed}/{len(probe_ids)}] ID {article_id}: 未找到")
    
    return results


def find_latest_article_id(start_from: int, saved_ids: set, missing_ids: set) -> Tuple[int, List[Dict[str, Any]]]:
    """探测最新的文章 ID"""
    current_start = start_from
    found_articles = []
    last_found_id = start_from - 1
    
    while current_start < start_from + MAX_PROBE_RANGE:
        probe_end = min(current_start + MAX_PROBE_RANGE - 1, start_from + MAX_PROBE_RANGE)
        
        # 并发探测
        results = concurrent_probe(current_start, probe_end, PROBE_STEP, CONCURRENT_WORKERS)
        
        # 分析结果
        consecutive_miss = 0
        for check_id in sorted(results.keys()):
            if results[check_id]:
                last_found_id = check_id
                consecutive_miss = 0
                if check_id not in saved_ids:
                    found_articles.append(results[check_id])
            else:
                missing_ids.add(check_id)
                consecutive_miss += 1
                if consecutive_miss >= CONSECUTIVE_MISS_LIMIT:
                    print(f"\n⚠️  连续 {consecutive_miss} 个 ID 未找到，停止探测")
                    return last_found_id, found_articles
        
        # 继续下一批
        current_start = probe_end + PROBE_STEP
        
        if consecutive_miss > 0:
            break
    
    return last_found_id, found_articles


def main():
    project_root = pathlib.Path(__file__).resolve().parent
    index_path = project_root / "鳄鱼派研报内容" / "文章md" / "index.json"
    
    # 读取索引
    index_data = read_index(str(index_path))
    saved_ids = set(index_data.get("saved_ids", []))
    missing_ids = set(index_data.get("missing_ids", []))
    last_probed_id = index_data.get("last_probed_id", 0)
    
    max_saved_id = max(saved_ids) if saved_ids else 0
    start_probe = max(last_probed_id, max_saved_id) + 1
    
    print("=" * 60)
    print("📊 独立并发探测脚本")
    print("=" * 60)
    print(f"已保存文章: {len(saved_ids)} 篇 (最大 ID: {max_saved_id})")
    print(f"已记录缺失: {len(missing_ids)} 个")
    print(f"上次探测到: ID {last_probed_id}")
    print(f"本次探测从: ID {start_probe}")
    print("=" * 60)
    
    # 开始探测
    latest_id, new_articles = find_latest_article_id(start_probe, saved_ids, missing_ids)
    
    # 更新索引
    if latest_id >= start_probe:
        index_data["last_probed_id"] = latest_id
        index_data["missing_ids"] = sorted(missing_ids)[-800:]  # 只保留最近 800 个
        write_index(str(index_path), index_data)
        print(f"\n✅ 已更新探测记录: 最新 ID = {latest_id}")
    
    # 显示结果
    print("\n" + "=" * 60)
    print(f"🎯 探测完成！")
    print("=" * 60)
    print(f"最新文章 ID: {latest_id}")
    print(f"新发现文章: {len(new_articles)} 篇")
    
    if new_articles:
        print("\n新文章列表:")
        for article in new_articles[:10]:
            print(f"  • ID {article['id']}: {article['title']} ({article.get('date', 'N/A')})")
        if len(new_articles) > 10:
            print(f"  ... 还有 {len(new_articles) - 10} 篇")
        print("\n💡 运行 v6 主程序进行增量下载")
    else:
        print("未发现新文章")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
