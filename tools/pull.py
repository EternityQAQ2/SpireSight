import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# ================= 配置区域 =================
WIKI_DOMAIN = "sts2.huijiwiki.com"
API_URL = f"https://{WIKI_DOMAIN}/api.php"
OUTPUT_DIR = Path(__file__).resolve().parent / "sts_raw_data"

# 需要下载的 Data 页面列表
DATA_PAGES = [
    "Card.tabx",
    "Relic.tabx",
    "Power.tabx",
    "Potion.tabx",
    "Monster.tabx",
    "Event.tabx",
    "Keyword.tabx",
    "Interaction.tabx",
    "Affliction.tabx",
    "Intent.tabx",
    "Modifier.tabx",
    "Badge.tabx",
    "Enchantment.tabx",
    "Trivia.tabx",
    "Version.tabx",
]

HEADERS = {
    'User-Agent': 'LINXAURA'
}

def download_page_content(page_title):
    """获取单个 Data 页面的原始 wikitext 内容"""
    params = {
        'action': 'query',
        'format': 'json',
        'titles': f'Data:{page_title}',
        'prop': 'revisions',
        'rvprop': 'ids|timestamp|content',
        'rvslots': 'main'
    }
    
    try:
        request = Request(f"{API_URL}?{urlencode(params)}", headers=HEADERS)
        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        pages = data['query']['pages']
        for page_id, page_data in pages.items():
            if int(page_id) < 0:
                print(f"[错误] 页面 Data:{page_title} 不存在 (ID: {page_id})")
                return None
            
            if 'revisions' in page_data:
                revision = page_data['revisions'][0]
                return {
                    "content": revision['slots']['main']['*'],
                    "page_id": page_data.get("pageid"),
                    "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                }
        return None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
        print(f"[请求异常] Data:{page_title} - {e}")
        return None

def save_as_jsonl(filename, snapshot):
    """每个文件保存一个完整 TabX 文档及其来源元数据。"""
    filepath = OUTPUT_DIR / filename.replace('.tabx', '.jsonl')
    record = {
        "raw_line": snapshot["content"],
        "source_file": filename,
        "source_url": f"https://{WIKI_DOMAIN}/wiki/Data:{filename}",
        "page_id": snapshot["page_id"],
        "revision_id": snapshot["revision_id"],
        "revision_timestamp": snapshot["revision_timestamp"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    filepath.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"[保存成功] {filepath}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"开始从 {WIKI_DOMAIN} 离线下载数据...")
    
    for page_title in DATA_PAGES:
        print(f"正在下载: Data:{page_title} ...")
        snapshot = download_page_content(page_title)
        
        if snapshot:
            save_as_jsonl(page_title, snapshot)
        else:
            print(f"[跳过] Data:{page_title} 未获取到内容")
        
        # 避免触发频率限制
        time.sleep(0.5)

    print("\n离线下载任务完成！数据已保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
