# ingest.py
"""
Step 1: Crawls all chapters of 'মেজদিদি' from Bengali Wikisource, cleans the Bangla text,
splits it into chunks with metadata (Book, Chapter, Section, URL) and stores them in ChromaDB.

Run from the 'src' folder:  python ingest.py

Why the MediaWiki API instead of scraping the HTML page?
  Wikisource book pages are assembled from transcluded pages with many header/footer/navigation
  templates, so plain HTML scraping with BeautifulSoup is unreliable. The official API is stable:
    1) list=allpages  -> lists all chapter subpages of the book (এক, দুই, ... আট)
    2) action=parse   -> returns the rendered HTML (with transcluded text) of each chapter
"""
import os
import re
import time
import shutil
import unicodedata
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ----------------------------------------------------------------------------
# Book configuration
# ----------------------------------------------------------------------------
BASE_DOMAIN = "https://bn.wikisource.org"
API_URL = f"{BASE_DOMAIN}/w/api.php"
INDEX_URL = "https://bn.wikisource.org/wiki/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF_(%E0%A6%B6%E0%A6%B0%E0%A7%8E%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC,_%E0%A7%A7%E0%A7%AF%E0%A7%AC%E0%A7%A9)/%E0%A6%AE%E0%A7%87%E0%A6%9C%E0%A6%A6%E0%A6%BF%E0%A6%A6%E0%A6%BF"

BOOK_NAME = "মেজদিদি"
AUTHOR_NAME = "শরৎচন্দ্র চট্টোপাধ্যায়"
SECTION_NAME = "মূল গল্প (মেজদিদি)"

# ----------------------------------------------------------------------------
# Paths (resolved relative to this file, so the script works from any directory)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_store")
CHAPTER_TEXT_DIR = BASE_DIR / "data" / "chapters"  # cleaned chapter text is saved here for inspection
COLLECTION_NAME = "bangla_book_rag"

# Multilingual embedding model that supports Bengali.
# BGE-M3 is trained for question -> passage retrieval in 100+ languages (including Bengali)
# and accepts up to 8192 tokens, so chunks are never truncated.
EMBED_MODEL = "BAAI/bge-m3"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 40

# If a chapter yields fewer characters than this, we assume its text was not extracted properly
MIN_CHAPTER_CHARS = 200

HEADERS = {
    "User-Agent": "BanglaBookRAGBot/1.0 (student RAG assignment; python-requests)"
}


def normalize_bn(text: str) -> str:
    """Normalizes a Bengali Unicode string (NFC) for reliable comparison."""
    return unicodedata.normalize("NFC", text.strip())


# Chapter name -> position (used to sort chapters in the correct reading order)
_NUMBER_WORDS = [
    "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "এগার", "বার", "তের", "চৌদ্দ", "পনের", "ষোল",
]
NUMBER_ORDER = {normalize_bn(w): i for i, w in enumerate(_NUMBER_WORDS, start=1)}


# ----------------------------------------------------------------------------
# Wikisource API helpers
# ----------------------------------------------------------------------------
def api_get(params: dict) -> dict:
    """Sends a request to the Wikisource API (retries up to 3 times on failure)."""
    params = {**params, "format": "json", "formatversion": "2"}
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 429:  # rate limited, wait and retry
                time.sleep(5 * attempt)
                continue
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
            time.sleep(2 * attempt)
            continue
        if "error" in data:
            raise RuntimeError(f"Wikisource API error: {data['error']}")
        return data
    raise RuntimeError(f"Could not reach the Wikisource API: {last_error}")


def title_from_url(url: str) -> str:
    """Extracts the Wikisource page title from a URL."""
    path = urllib.parse.urlparse(url).path
    raw_title = path.split("/wiki/", 1)[1]
    return normalize_bn(urllib.parse.unquote(raw_title).replace("_", " "))


def url_from_title(title: str) -> str:
    """Builds a full, clickable URL from a Wikisource page title."""
    return f"{BASE_DOMAIN}/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="/(),_")


def chapter_sort_key(title: str):
    name = normalize_bn(title.rsplit("/", 1)[-1])
    return (NUMBER_ORDER.get(name, 999), name)


def get_chapter_pages(index_url: str) -> list[dict]:
    """Finds all chapter pages (subpages) of the book."""
    print("1. Looking up the chapter list on Wikisource...")
    index_title = title_from_url(index_url)
    print(f"   Book index page: {index_title}")

    # Method 1: every page whose title starts with "<index title>/" (এক, দুই, ... আট)
    titles = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": "0",
        "apprefix": index_title + "/",
        "aplimit": "500",
    }
    while True:
        data = api_get(params)
        titles.extend(p["title"] for p in data["query"]["allpages"])
        if "continue" in data:
            params.update(data["continue"])
        else:
            break

    # Keep only direct chapter pages (not deeper subpages)
    prefix_len = len(index_title) + 1
    titles = [t for t in titles if "/" not in t[prefix_len:]]

    # Method 2 (fallback): read chapter links from the index page itself
    if not titles:
        print("   Note: allpages returned nothing, falling back to links on the index page...")
        data = api_get({"action": "parse", "page": index_title, "prop": "links"})
        parent_title = index_title.rsplit("/", 1)[0]
        for link in data["parse"].get("links", []):
            t = normalize_bn(link["title"])
            last = t.rsplit("/", 1)[-1]
            if t.startswith(parent_title + "/") and last in NUMBER_ORDER:
                titles.append(t)
        titles = list(dict.fromkeys(titles))  # remove duplicates, keep order

    titles.sort(key=chapter_sort_key)

    return [
        {
            "chapter_name": f"পরিচ্ছেদ: {t.rsplit('/', 1)[-1]}",
            "page_title": t,
            "url": url_from_title(t),
        }
        for t in titles
    ]


def fetch_chapter_html(page_title: str) -> str:
    """Fetches the rendered HTML (including transcluded text) of one chapter page."""
    data = api_get({
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "redirects": "1",
        "disableeditsection": "1",
        "disablelimitreport": "1",
    })
    return data["parse"]["text"]


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_bangla_text(raw_text: str) -> str:
    """Cleans raw Bengali text: removes unwanted symbols, footnote markers and extra whitespace."""
    text = raw_text

    # Remove zero-width space and non-breaking space characters
    text = text.replace("\u200b", "").replace("\xa0", " ")

    # Remove Wikisource footnote markers like [১], [২], [1], [2], [শিরোনাম]
    text = re.sub(r"\[[^\]]{1,15}\]", "", text)

    # Remove standalone page numbers (Bengali and English digits)
    text = re.sub(r"^[ \t]*[০-৯0-9]{1,4}[ \t]*$", "", text, flags=re.MULTILINE)

    # Remove trailing spaces at the end of each line
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3 or more consecutive newlines with a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Strict mode: drops navigation, header/footer templates and page numbers
_STRICT_REMOVE = re.compile(
    r"(ws-noexport|headertemplate|footertemplate|navbox|noprint|mw-editsection|"
    r"pagenum|ws-pagenum|ws-header|ws-footer)"
)
# Loose mode: drops only elements that are certainly not prose
_LOOSE_REMOVE = re.compile(r"(navbox|mw-editsection|pagenum|ws-pagenum)")


def _extract_with(html: str, remove_pattern: re.Pattern) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["style", "script", "noscript"]):
        tag.decompose()

    for tag in soup.find_all(class_=remove_pattern):
        if not getattr(tag, "decomposed", False):
            tag.decompose()

    # Turn <br> into real line breaks
    for br in soup.find_all("br"):
        br.replace_with("\n")

    root = soup.find("div", class_="prp-pages-output") or soup

    paragraphs = []
    for p in root.find_all("p"):
        p_text = re.sub(r"\s+", " ", p.get_text()).strip()
        if len(p_text) > 1:
            paragraphs.append(p_text)

    # Fallback: prose is not wrapped in <p> tags, so take the whole text
    if len("\n\n".join(paragraphs)) < MIN_CHAPTER_CHARS:
        return clean_bangla_text(root.get_text(separator="\n"))

    return clean_bangla_text("\n\n".join(paragraphs))


def extract_chapter_text(html: str) -> str:
    """Tries strict cleaning first; if too little text is found, retries with loose cleaning."""
    text = _extract_with(html, _STRICT_REMOVE)
    if len(text) >= MIN_CHAPTER_CHARS:
        return text
    print("      (strict cleaning found too little text, retrying with loose cleaning...)")
    return _extract_with(html, _LOOSE_REMOVE)


def save_chapter_text(idx: int, chapter_name: str, text: str) -> None:
    """Saves the cleaned text to disk so you can inspect it and write test questions from it."""
    CHAPTER_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = chapter_name.replace(":", "").replace(" ", "_")
    (CHAPTER_TEXT_DIR / f"{idx:02d}_{safe_name}.txt").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    chapters = get_chapter_pages(INDEX_URL)
    if not chapters:
        print("Error: No chapters found. Please check your internet connection and INDEX_URL.")
        return

    print(f"   Found {len(chapters)} chapters in '{BOOK_NAME}':")
    for c in chapters:
        print(f"     - {c['chapter_name']}  ->  {urllib.parse.unquote(c['url'])}")

    print("2. Downloading and cleaning all chapters...")
    chapter_docs = []
    failed = []

    for idx, chap in enumerate(chapters, start=1):
        chap_name = chap["chapter_name"]
        print(f"   [{idx}/{len(chapters)}] {chap_name}")
        try:
            html = fetch_chapter_html(chap["page_title"])
            cleaned_text = extract_chapter_text(html)
        except Exception as e:
            print(f"      WARNING: Failed to fetch '{chap_name}': {e}")
            failed.append(chap_name)
            continue

        if len(cleaned_text) < MIN_CHAPTER_CHARS:
            print(f"      WARNING: Not enough text in '{chap_name}' "
                  f"(HTML: {len(html)} chars, extracted text: {len(cleaned_text)} chars).")
            failed.append(chap_name)
            continue

        preview = cleaned_text[:60].replace("\n", " ")
        print(f"      OK: {len(cleaned_text)} chars | starts with: {preview}...")
        save_chapter_text(idx, chap_name, cleaned_text)

        chapter_docs.append(Document(
            page_content=cleaned_text,
            metadata={
                "book_name": BOOK_NAME,
                "author": AUTHOR_NAME,
                "chapter_name": chap_name,
                "section_name": f"{SECTION_NAME} - {chap_name}",
                "source_url": chap["url"],
            },
        ))
        time.sleep(0.5)  # polite crawling delay

    if not chapter_docs:
        print("No readable text could be crawled from the book.")
        return

    if failed:
        print(f"\n   WARNING: These chapters were NOT ingested, so the book is incomplete: "
              f"{', '.join(failed)}\n")

    total_chars = sum(len(d.page_content) for d in chapter_docs)
    print(f"   Total: {len(chapter_docs)} chapters, {total_chars} characters.")

    print(f"3. Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    # Split order: paragraph -> line -> Bengali danda (।) -> question/exclamation mark -> word
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "। ", "।", "? ", "! ", " ", ""],
        add_start_index=True,
    )
    raw_chunks = splitter.split_documents(chapter_docs)

    chunks = []
    for chunk in raw_chunks:
        if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        chunk.metadata["chunk_index"] = len(chunks)
        chunks.append(chunk)

    # Add a small "book info" chunk so questions about the book's name/author can be answered.
    # (The chapter text itself never mentions the author.)
    info_text = (
        f"বইয়ের নাম: {BOOK_NAME}। লেখক: {AUTHOR_NAME}। "
        f"এই বইয়ে মোট {len(chapter_docs)}টি পরিচ্ছেদ আছে।"
    )
    chunks.append(Document(
        page_content=info_text,
        metadata={
            "book_name": BOOK_NAME,
            "author": AUTHOR_NAME,
            "chapter_name": "বইয়ের পরিচিতি",
            "section_name": "বইয়ের পরিচিতি",
            "source_url": INDEX_URL,
            "chunk_index": len(chunks),
        },
    ))

    print(f"   Produced {len(chunks)} chunks with metadata.")

    print(f"4. Loading multilingual embedding model ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Remove the old database so old and new vectors do not get mixed
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print(f"5. Creating the vector store and saving it in '{CHROMA_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done! Stored {vectorstore._collection.count()} vectors in ChromaDB.")


if __name__ == "__main__":
    main()