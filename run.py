"""CLI entry point: python run.py crawl | index | serve"""

import logging
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

USAGE = """\
Usage: python run.py <command>

Commands:
  crawl   Crawl all sheets (no embedding)
  index   Crawl + embed + index into ChromaDB
  serve   Launch the Streamlit chat UI
"""


def cmd_crawl() -> None:
    from ingestion.crawler import crawl

    result = crawl()
    print(f"\nCrawled {len(result.tabs)} tabs, {len(result.errors)} errors.")
    for tab in result.tabs:
        print(f"  {tab.spreadsheet_title} / {tab.tab_name} — {len(tab.rows)} rows")
    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  {err}")


def cmd_index() -> None:
    from ingestion.crawler import crawl
    from vectorstore.index_manager import index
    from vectorstore.store import count

    print("Crawling...")
    crawl_result = crawl()
    print(f"Crawled {len(crawl_result.tabs)} tabs. Indexing...")

    stats = index(crawl_result)
    print(f"\nIndexing complete:")
    print(f"  Added:   {stats['added']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Deleted: {stats['deleted']}")
    print(f"  Total chunks in DB: {count()}")


def cmd_serve() -> None:
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("crawl", "index", "serve"):
        print(USAGE)
        sys.exit(1)

    command = sys.argv[1]
    {"crawl": cmd_crawl, "index": cmd_index, "serve": cmd_serve}[command]()


if __name__ == "__main__":
    main()
