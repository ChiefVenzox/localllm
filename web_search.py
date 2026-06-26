"""
web_search.py
=============
Small HTML web search helper.

This intentionally does not use a search/weather API. It fetches DuckDuckGo Lite
HTML like a lightweight browser and extracts result titles, snippets and URLs.
It is best-effort: search pages change, so callers should be ready for None.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_duck_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return qs["uddg"][0]
    return href


def search(query: str, max_results: int = 5, timeout: int = 10) -> list[dict]:
    q = query.strip()
    if not q:
        return []
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q) + "&kl=tr-tr"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")

    pattern = re.compile(
        r"<a[^>]+href=\"(?P<href>[^\"]+)\"[^>]+class='result-link'>(?P<title>.*?)</a>"
        r".*?<td class='result-snippet'>\s*(?P<snippet>.*?)\s*</td>"
        r".*?<span class='link-text'>(?P<display>.*?)</span>",
        re.S,
    )
    results = []
    for match in pattern.finditer(page):
        url = _resolve_duck_url(match.group("href"))
        title = _clean_html(match.group("title"))
        snippet = _clean_html(match.group("snippet"))
        display = _clean_html(match.group("display"))
        if title and (snippet or display):
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "display_url": display,
            })
        if len(results) >= max_results:
            break
    return results


def best_snippet(query: str, required: tuple[str, ...] = (), max_results: int = 6) -> dict | None:
    results = search(query, max_results=max_results)
    if not required:
        return results[0] if results else None
    required_l = tuple(r.lower() for r in required)
    for item in results:
        hay = f"{item['title']} {item['snippet']} {item['display_url']}".lower()
        if any(r in hay for r in required_l):
            return item
    return results[0] if results else None


if __name__ == "__main__":
    for result in search("Istanbul hava durumu bugun kac derece", max_results=3):
        print(result["title"])
        print(result["snippet"])
        print(result["url"])
        print()
