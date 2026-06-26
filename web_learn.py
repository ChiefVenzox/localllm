"""
web_learn.py
============
"Belirli siteleri gez -> temiz metni cikar -> yerelde kendini egit."

Tamamen STANDART KUTUPHANE ile (urllib + html.parser). Hicbir dis bagimlilik yok.

Guvenlik / nezaket:
  * Yalnizca WHITELIST'teki alan adlarina (domain) gidilir.
  * robots.txt'e saygi gosterilir (urllib.robotparser).
  * Istekler arasi gecikme + sayfa boyutu + sayfa sayisi siniri.
  * Sadece http(s) ve text/html; ikili/dosya indirilmez.

Cikti:
  * to_docs(page)  -> ham metin dokumanlari ("okuyarak" ogrenme icin)
  * to_qa(page)    -> [{"messages":[user, assistant]}] (chat formatinda ogrenme icin)
  * data/web/ altina ham metin + uretilen S/C kaydedilir (yeniden egitim icin).

CLI (sadece gez+goster, egitim yok):
    python web_learn.py --seed https://tr.wikipedia.org/wiki/Yapay_zeka --max-pages 1
"""
from __future__ import annotations
import argparse
import hashlib
import html as _html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

USER_AGENT = "yerelLLM-Bilge/0.1 (yerel ogrenme botu; kisisel kullanim)"
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "header",
              "footer", "aside", "form", "button", "figure", "figcaption",
              "select", "label"}
_VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "area",
              "base", "col", "embed", "param", "track", "wbr"}
_BLOCK_TAGS = {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2",
               "h3", "h4", "h5", "h6", "blockquote", "pre"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
# HTML'de bitis etiketi opsiyonel olan elemanlar: yeni X acilinca onceki kapanir
_AUTO_CLOSE = {
    "li": {"li"}, "option": {"option"}, "tr": {"tr"},
    "td": {"td", "th", "tr"}, "th": {"td", "th", "tr"},
    "dt": {"dt", "dd"}, "dd": {"dt", "dd"},
    "thead": {"tbody", "tfoot", "tr"}, "tbody": {"tbody", "tfoot"},
    "p": {"p", "div", "ul", "ol", "table", "section", "article", "blockquote",
          "pre", "h1", "h2", "h3", "h4", "h5", "h6"},
}
# class/id/role icinde gecerse o blogu (ve altini) at: menu, sidebar, dil listesi vb.
# DIKKAT: genis substring'ler (orn. "vector-") icerigi de atlayabilir -> spesifik tut.
_BOILER = ("navbox", "mw-jump", "mw-portlet", "catlinks", "mw-editsection",
           "interlanguage", "interwiki", "p-lang", "langlist", "language-list",
           "sidebar", "breadcrumb", "noprint", "hatnote", "shortdescription",
           "metadata", "ambox", "cookie", "mw-indicators", "sister", "reflist",
           "mw-references", "printfooter", "site-notice", "navbar",
           "thumbcaption", "gallery", "vector-menu", "vector-toc",
           "vector-header", "vector-sticky", "vector-page-tool",
           "vector-main-menu", "vector-column", "vector-user", "vector-dropdown",
           "vector-pinnable", "mw-footer", "toclevel", "vector-body-before",
           "mw-redirectedfrom", "flaggedrevs", "mw-fr-", "contentsub",
           "sitesub", "mw-revision", "jump-to-nav", "mw-empty-elt",
           "mw-jump-link", "mw-list-item")
_BOILER_ROLE = {"navigation", "banner", "complementary", "contentinfo",
                "search", "menu", "menubar", "dialog"}


def _is_boiler(attrs) -> bool:
    s, role = "", ""
    for k, v in attrs:
        if not v:
            continue
        if k in ("class", "id"):
            s += " " + v.lower()
        elif k == "role":
            role = v.lower()
    if role in _BOILER_ROLE:
        return True
    return any(pat in s for pat in _BOILER)


# ---------------------------------------------------------------------------
#  HTML -> baslik + bolumler (heading, metin) + baglantilar
#  Etiket YIGINI ile dogru atlama: <nav>/<header>/... ve boilerplate class'lar
#  (ve altlari) komple atlanir; ana metin kalir.
# ---------------------------------------------------------------------------
class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, bool]] = []   # (etiket, atlaniyor mu)
        self.skip = 0                  # atlanan acik eleman sayisi
        self.in_title = False
        self.title = ""
        self.cur_heading = ""
        self.cur_text: List[str] = []
        self.sections: List[Tuple[str, str]] = []
        self.links: List[str] = []
        self._heading_buf: Optional[List[str]] = None

    def _flush_section(self):
        text = _clean(" ".join(self.cur_text))
        if text:
            self.sections.append((self.cur_heading.strip(), text))
        self.cur_text = []

    def _pop_one(self):
        t, sk = self.stack.pop()
        if sk:
            self.skip -= 1
        elif t in _HEADING_TAGS and self._heading_buf is not None:
            self.cur_heading = _clean(" ".join(self._heading_buf))
            self._heading_buf = None
        elif t == "title":
            self.in_title = False
        return t

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_TAGS:
            if tag == "br" and not self.skip:
                self.cur_text.append("\n")
            return
        # opsiyonel-kapanis: yeni etiket onceki ayni-tur elemani kapatir
        closers = _AUTO_CLOSE.get(tag)
        while self.stack and closers and self.stack[-1][0] in closers:
            self._pop_one()

        # html/body/main: feature-flag class'lari (orn. 'vector-toc-not-available')
        # icerik degildir -> bu kok etiketlerde boilerplate kontrolu yapma.
        boiler = tag not in ("html", "body", "main") and _is_boiler(attrs)
        skip_this = bool(self.skip) or (tag in _SKIP_TAGS) or boiler
        self.stack.append((tag, skip_this))
        if skip_this:
            self.skip += 1
            return
        if tag == "title":
            self.in_title = True
        elif tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)
        elif tag in _HEADING_TAGS:
            self._flush_section()
            self._heading_buf = []
        elif tag in _BLOCK_TAGS:
            self.cur_text.append("\n")

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        # eslesen etikete kadar her seyi kapat (kapanmamis cocuklar dahil)
        depth = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                depth = i
                break
        if depth is None:
            return
        while len(self.stack) > depth:
            self._pop_one()

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title += data
        elif self._heading_buf is not None:
            self._heading_buf.append(data)
        else:
            self.cur_text.append(data)

    def finish(self):
        self._flush_section()


_WS = re.compile(r"[ \t\r\f\v]+")
_MULTINL = re.compile(r"\n\s*\n+")
_BRACKET_REF = re.compile(r"\[\d+\]|\[kaynak belirtilmeli\]", re.I)


def _clean(text: str) -> str:
    text = _html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")  # nbsp / zero-width
    text = _BRACKET_REF.sub("", text)         # [1] gibi dipnot isaretleri
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # " ," -> ","
    text = re.sub(r"\(\s+", "(", text)            # "( x" -> "(x"
    text = re.sub(r"\s+\)", ")", text)            # "x )" -> "x)"
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
#  Cekme (fetch) + robots
# ---------------------------------------------------------------------------
class Fetcher:
    def __init__(self, delay: float = 2.0, max_bytes: int = 2_000_000,
                 timeout: float = 12.0, respect_robots: bool = True):
        self.delay = delay
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.respect_robots = respect_robots
        self._robots: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last = 0.0

    def _can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            self._robots[base] = self._read_robots(base)
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def _read_robots(self, base: str):
        """robots.txt'i KENDI User-Agent'imizla ceker (varsayilan UA bircok sitede
        403 alir, bu da yanlislikla 'her seyi yasakla'ya yol acardi)."""
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(base + "/robots.txt",
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = r.read(500_000).decode("utf-8", errors="replace")
            rp.parse(txt.splitlines())
            return rp
        except Exception:
            # robots okunamadi (404/403/aglik) -> kullanici whitelist'ledigi icin izinli say
            return None

    def get(self, url: str) -> Optional[str]:
        if not url.lower().startswith(("http://", "https://")):
            return None
        if not self._can_fetch(url):
            print(f"[web] robots.txt engelledi: {url}")
            return None
        dt = time.monotonic() - self._last
        if dt < self.delay:
            time.sleep(self.delay - dt)
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "tr,en;q=0.8",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                ctype = r.headers.get("Content-Type", "")
                if "text/html" not in ctype and "text/plain" not in ctype:
                    return None
                raw = r.read(self.max_bytes)
            enc = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype)
            if m:
                enc = m.group(1)
            return raw.decode(enc, errors="replace")
        except Exception as e:
            print(f"[web] cekilemedi {url}: {repr(e)[:80]}")
            return None
        finally:
            self._last = time.monotonic()


def extract(html_text: str, base_url: str) -> Dict:
    p = _Extractor()
    try:
        p.feed(html_text)
        p.finish()
    except Exception:
        pass
    title = _clean(p.title)
    for suf in (" - Vikipedi", " - Wikipedia", " | "):
        i = title.find(suf)
        if i > 0:
            title = title[:i].strip()
    # baglantilari mutlak url'ye cevir
    links = []
    for href in p.links:
        try:
            u = urllib.parse.urljoin(base_url, href)
            u, _ = urllib.parse.urldefrag(u)
            if u.lower().startswith(("http://", "https://")):
                links.append(u)
        except Exception:
            continue
    # cok kisa/menu bolumlerini ele
    sections = [(h, t) for (h, t) in p.sections if len(t) >= 80]
    return {"url": base_url, "title": title, "sections": sections, "links": links}


# ---------------------------------------------------------------------------
#  Gezinme (crawl)
# ---------------------------------------------------------------------------
def crawl(seeds: List[str], allow_domains: List[str], *, max_pages: int = 5,
          max_depth: int = 0, fetcher: Optional[Fetcher] = None) -> List[Dict]:
    fetcher = fetcher or Fetcher()
    allow = set(d.lower().lstrip("www.") for d in allow_domains)

    def ok_domain(u: str) -> bool:
        net = urllib.parse.urlsplit(u).netloc.lower().lstrip("www.")
        return any(net == d or net.endswith("." + d) for d in allow)

    seen, pages = set(), []
    q = deque((s, 0) for s in seeds)
    while q and len(pages) < max_pages:
        url, depth = q.popleft()
        if url in seen or not ok_domain(url):
            continue
        seen.add(url)
        html_text = fetcher.get(url)
        if not html_text:
            continue
        page = extract(html_text, url)
        if page["sections"]:
            pages.append(page)
            print(f"[web] +{page['title'][:50]!r} ({len(page['sections'])} bolum) <- {url}")
        if depth < max_depth:
            for link in page["links"]:
                if link not in seen and ok_domain(link):
                    q.append((link, depth + 1))
    return pages


# ---------------------------------------------------------------------------
#  Metin -> egitim verisi
# ---------------------------------------------------------------------------
_Q_TEMPLATES = [
    "{k} nedir?",
    "{k} hakkinda bilgi verir misin?",
    "{k} ne demek?",
    "Bana {k} konusunu anlatir misin?",
]

# Kullanici sapka/diakritik kullanmayabilir -> sorulari katlanmis bicimleriyle de uret
_CIRCUM = str.maketrans({"â": "a", "î": "i", "û": "u", "Â": "A", "Î": "I", "Û": "U"})
_TR2ASCII = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "İ": "I", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "C", "Ğ": "G", "Ö": "O", "Ş": "S", "Ü": "U",
})


def _fold_variants(s: str) -> List[str]:
    """Orijinal + sapkasiz + tam-ASCII-Turkce bicimleri (tekil)."""
    out = [s]
    c = s.translate(_CIRCUM)
    if c not in out:
        out.append(c)
    a = c.translate(_TR2ASCII)
    if a not in out:
        out.append(a)
    return out


def _first_sentences(text: str, max_chars: int = 150) -> str:
    """Kucuk model temiz ezberlesin diye KISA tut: tercihen ilk cumle, <= ~150 char."""
    text = text.strip()
    m = re.search(r"[.!?](\s|$)", text)        # ilk cumle sonu
    if m and 40 <= m.start() + 1 <= max_chars + 30:
        return text[:m.start() + 1].strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 60 else cut).strip()


def _lead(sections: List[Tuple[str, str]]) -> str:
    """Ilk 'duzgun' (prose) paragraf: yeterince uzun ve parantez/bildirimle baslamayan."""
    for _, t in sections:
        s = t.lstrip()
        if len(s) >= 120 and not s.startswith(("(", "[", "Bu kararli", "2 be")):
            return s
    return sections[0][1] if sections else ""


def to_qa(page: Dict, max_qa: int = 10) -> List[Dict]:
    """Baslik/bolumlerden otomatik Soru-Cevap (chat) ornekleri uretir."""
    out = []
    title = page["title"]
    sections = page["sections"]
    # 1) Ana konu: ilk DUZGUN paragraf -> baslik sorulari (sapkali/sapkasiz/ASCII)
    if title and sections:
        ans = _first_sentences(_lead(sections))
        if len(ans) > 40:
            for kv in _fold_variants(title):
                for tmpl in _Q_TEMPLATES[:2]:
                    out.append({"messages": [
                        {"role": "user", "content": tmpl.format(k=kv)},
                        {"role": "assistant", "content": ans},
                    ]})
    # 2) Her bolum basligi -> bolum metni (basligin sapkasiz bicimiyle)
    tfold = title.translate(_CIRCUM)
    for heading, text in sections:
        if not heading or len(heading) < 3 or len(heading) > 60:
            continue
        ans = _first_sentences(text)
        if len(ans) < 40:
            continue
        hk = heading.translate(_CIRCUM).lower()
        out.append({"messages": [
            {"role": "user", "content": f"{tfold} konusunda {hk} nedir?"},
            {"role": "assistant", "content": ans},
        ]})
        if len(out) >= max_qa:
            break
    return out[:max_qa]


def to_docs(page: Dict, max_chars: int = 1200) -> List[str]:
    """Sayfayi 'okuma' icin ham metin dokumanlarina boler."""
    docs, buf = [], []
    size = 0
    head = page["title"]
    for heading, text in page["sections"]:
        chunk = (f"{heading}. " if heading else "") + text
        if size + len(chunk) > max_chars and buf:
            docs.append((head + "\n" + "\n".join(buf)).strip())
            buf, size = [], 0
        buf.append(chunk)
        size += len(chunk)
    if buf:
        docs.append((head + "\n" + "\n".join(buf)).strip())
    return [d for d in docs if len(d) > 80]


# ---------------------------------------------------------------------------
#  Kayit (yeniden egitim / iz surme icin)
# ---------------------------------------------------------------------------
def save_harvest(pages: List[Dict], out_dir: str = "data/web") -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    qa_path = os.path.join(out_dir, "web_qa.jsonl")
    txt_path = os.path.join(out_dir, "web_docs.txt")
    n_qa = n_doc = 0
    with open(qa_path, "a", encoding="utf-8") as fq, \
         open(txt_path, "a", encoding="utf-8") as ft:
        for page in pages:
            for qa in to_qa(page):
                fq.write(json.dumps(qa, ensure_ascii=False) + "\n")
                n_qa += 1
            for doc in to_docs(page):
                ft.write(doc + "\n\n")
                n_doc += 1
    return {"pages": len(pages), "qa": n_qa, "docs": n_doc,
            "qa_path": qa_path, "txt_path": txt_path}


def harvest(seeds, allow_domains, **kw) -> Tuple[List[Dict], Dict]:
    pages = crawl(seeds, allow_domains, **kw)
    stats = save_harvest(pages) if pages else {"pages": 0, "qa": 0, "docs": 0}
    return pages, stats


def study_pages(learner, pages: List[Dict], *, do_docs: bool = True,
                do_qa: bool = True, qa_steps: int = 6, doc_passes: int = 1,
                max_qa_per_page: int = 10) -> Dict:
    """Cekilen sayfalari CANLI modele ogretir: dokumanlari 'okur' (study_text),
    Soru-Cevaplari 'ogretir' (teach). learner: OnlineLearner."""
    n_doc = n_qa = 0
    loss_first = loss_last = None
    for page in pages:
        if do_docs:
            for doc in to_docs(page):
                r = learner.study_text(doc, passes=doc_passes)
                if r:
                    n_doc += 1
                    loss_first = r["loss_before"] if loss_first is None else loss_first
                    loss_last = r["loss_after"]
        if do_qa:
            for qa in to_qa(page, max_qa=max_qa_per_page):
                r = learner.teach(qa["messages"], steps=qa_steps)
                if r:
                    n_qa += 1
                    loss_last = r["loss_after"]
    return {"pages": len(pages), "docs_studied": n_doc, "qa_taught": n_qa,
            "loss_first": loss_first, "loss_last": loss_last}


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", nargs="+", required=True)
    ap.add_argument("--allow", nargs="*", default=None,
                    help="izinli alan adlari (vars: seed'lerin alan adlari)")
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--no-robots", action="store_true")
    args = ap.parse_args()
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    allow = args.allow or [urllib.parse.urlsplit(s).netloc for s in args.seed]
    f = Fetcher(delay=args.delay, respect_robots=not args.no_robots)
    pages = crawl(args.seed, allow, max_pages=args.max_pages,
                  max_depth=args.max_depth, fetcher=f)
    print(f"\n[web] {len(pages)} sayfa cekildi. Ornek S/C ve dokuman:")
    for page in pages[:1]:
        qa = to_qa(page)
        docs = to_docs(page)
        print(f"\n== {page['title']} ==  ({len(page['sections'])} bolum)")
        for ex in qa[:4]:
            print(f"  S: {ex['messages'][0]['content']}")
            print(f"  C: {ex['messages'][1]['content'][:110]}...")
        print(f"  [dokuman ornegi] {docs[0][:160]!r}..." if docs else "  (dokuman yok)")
    if pages:
        print("\n[web] kaydet:", save_harvest(pages))


if __name__ == "__main__":
    _main()
