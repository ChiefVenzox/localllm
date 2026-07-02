"""
web_lookup.py
=============
CANLI Wikipedia arama (retrieval): bir soruyu o an Wikipedia'da arar ve TEMIZ
ozeti dondurur. Modeli egitmeye gerek yok -> her konuda "tam nokta atisi" dogru
cevap. Tamamen stdlib (urllib + json); Wikipedia'nin resmi temiz API'lerini kullanir:

  * opensearch  -> soruya en uygun makale basligini bulur
  * REST summary -> o makalenin temiz, kisa ozetini (extract) verir (HTML kazima yok)

Bilge sunucusu bunu, Tsetlin hakemi "kapsam disi / emin degil" dediginde devreye
sokar: model uydurmasin, gercek kaynaktan cevap gelsin.
"""
from __future__ import annotations
import json
import re
import urllib.parse
import urllib.request

USER_AGENT = "yerelLLM-Bilge/0.1 (yerel asistan; kisisel kullanim)"

# "kuantum fiziği nedir?" -> "kuantum fiziği"
_STRIP = re.compile(
    r"\b(nedir|ne demek(tir)?|nedir ki|kimdir|ne ise yarar|hakk[iı]nda"
    r"( bilgi)?( verir misin| ver)?|anlat([iı]r m[iı]s[iı]n)?|a[cç][iı]kla(r m[iı]s[iı]n)?|"
    r"k[iı]saca|bana)\b", re.I)
_CAPITAL_RE = re.compile(
    r"^\s*(?P<subject>.+?)\s*(?:['’`´]?[nm]?[ıiuü]n)?\s+ba[şs]kenti\s+"
    r"(?:neresi|nedir|ne|nerededir)?\s*[?.!]*\s*$",
    re.I,
)


def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def clean_query(q: str) -> str:
    q = q.strip().rstrip("?.!").strip()
    q = _STRIP.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _search_title(query, lang="tr"):
    url = (f"https://{lang}.wikipedia.org/w/api.php?action=opensearch"
           f"&search={urllib.parse.quote(query)}&limit=1&namespace=0&format=json")
    data = _get_json(url)
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    return titles[0] if titles else None


def _summary(title, lang="tr"):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    data = _get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t}")
    extract = (data.get("extract") or "").strip()
    url = (data.get("content_urls", {}).get("desktop", {}).get("page")
           or f"https://{lang}.wikipedia.org/wiki/{t}")
    return extract, url


def _wikidata_item(title: str, lang="tr") -> str | None:
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=pageprops"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )
    data = _get_json(url)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        item = page.get("pageprops", {}).get("wikibase_item")
        if item:
            return item
    return None


def _wikidata_label(qid: str, lang="tr") -> str | None:
    data = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{urllib.parse.quote(qid)}.json")
    entity = data.get("entities", {}).get(qid, {})
    labels = entity.get("labels", {})
    label = labels.get(lang) or labels.get("en")
    return label.get("value") if label else None


def _wikidata_capital(qid: str, lang="tr") -> str | None:
    data = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{urllib.parse.quote(qid)}.json")
    entity = data.get("entities", {}).get(qid, {})
    claims = entity.get("claims", {})
    capital_claims = claims.get("P36") or []
    for claim in capital_claims:
        value = (
            claim.get("mainsnak", {})
            .get("datavalue", {})
            .get("value", {})
        )
        capital_id = value.get("id") if isinstance(value, dict) else None
        if capital_id:
            return _wikidata_label(capital_id, lang)
    return None


def _capital_lookup(question: str, lang="tr") -> dict | None:
    normalized = question.replace("’", "'").replace("`", "'").replace("´", "'")
    match = _CAPITAL_RE.match(normalized)
    if not match:
        return None
    subject = match.group("subject").strip(" ,?.!")
    if len(subject) < 2:
        return None
    title = _search_title(subject, lang)
    if not title:
        return None
    item = _wikidata_item(title, lang)
    if not item:
        return None
    capital = _wikidata_capital(item, lang)
    if not capital:
        return None
    _, url = _summary(title, lang)
    return {
        "title": title,
        "summary": f"{title} icin Wikidata'da kayitli baskent: {capital}.",
        "url": url,
    }


def _shorten(text, max_chars=400):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    dot = cut.rfind(". ")
    return (cut[:dot + 1] if dot > 80 else cut).strip()


def lookup(question, lang="tr", max_chars=400):
    """Soruyu Wikipedia'da arar. Bulursa {title, summary, url}, yoksa None."""
    capital = _capital_lookup(question, lang)
    if capital:
        return capital
    query = clean_query(question)
    if len(query) < 2:
        return None
    try:
        title = _search_title(query, lang)
        if not title:
            return None
        extract, url = _summary(title, lang)
        if not extract or len(extract) < 20:
            return None
        return {"title": title, "summary": _shorten(extract, max_chars), "url": url}
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for q in ["kuantum fiziği nedir", "Atatürk kimdir", "fotosentez ne demek",
              "Mona Lisa'yı kim yaptı", "asdf zırt pırt"]:
        r = lookup(q)
        if r:
            print(f"'{q}'\n  -> [{r['title']}] {r['summary'][:160]}\n")
        else:
            print(f"'{q}'\n  -> (bulunamadi)\n")
