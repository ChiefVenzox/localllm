"""
quick_intents.py
================
Small deterministic replies that should not wait for the language model.

The model is useful for open-ended chat, but some requests are better handled
before the model:
  * greetings / "nasilsin" -> stable local reply
  * daily life statements   -> reflective follow-up question
  * simple concept questions -> local Turkish explanation
  * weather questions      -> live web search snippets, no hallucination

Weather intentionally does not use a weather API. It searches the web through
HTML results and summarizes the best visible snippet. The default city is
Istanbul and can be changed with YERELLM_DEFAULT_CITY.
"""
from __future__ import annotations

import os
import re
import unicodedata
import web_search

DEFAULT_CITY = os.environ.get("YERELLM_DEFAULT_CITY", "Istanbul")


def _ascii_lower(text: str) -> str:
    text = text.lower().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def is_greeting(text: str) -> bool:
    t = _ascii_lower(text).strip(" ?!.")
    if len(t) > 80:
        return False
    patterns = (
        r"^(selam|merhaba|hey|gunaydin|iyi aksamlar|iyi gunler)$",
        r"^(selam|merhaba|gunaydin|iyi gunler|iyi aksamlar)[, ]+nasilsin\??$",
        r"^nasilsin(\s+bugun)?\??$",
        r"^naber\??$",
        r"^ne haber\??$",
    )
    return any(re.match(p, t) for p in patterns)


def greeting_reply(text: str) -> str | None:
    if not is_greeting(text):
        return None
    t = _ascii_lower(text)
    if "gunaydin" in t:
        return "Gunaydin! Ben iyiyim, yardima hazirim. Sen nasilsin?"
    if "iyi aksamlar" in t:
        return "Iyi aksamlar. Bugunun nasil gecti?"
    if "nasilsin" in t or "naber" in t or "ne haber" in t:
        return "Iyiyim, tesekkur ederim. Ben bir yapay zeka oldugum icin yorulmam; sana yardim etmeye hazirim. Sen nasilsin?"
    return "Merhaba. Bugun nasilsin, gunun nasil geciyor?"


def is_weather_question(text: str) -> bool:
    t = _ascii_lower(text)
    direct = (
        "hava durumu", "kac derece", "sicaklik", "sicakligi",
        "yagmur yagacak", "yagmur var mi", "ruzgar kac", "nem kac",
    )
    if any(w in t for w in direct):
        return True
    if "hava" not in t:
        return False
    question_words = ("nasil", "ne durumda", "kac", "soguk mu", "sicak mi", "yagmurlu mu")
    return any(w in t for w in question_words)


def extract_weather_city(text: str, default_city: str = DEFAULT_CITY) -> str:
    raw = text.strip()
    lowered = _ascii_lower(raw)

    # "ankara'da hava nasil", "izmirde hava kac derece"
    m = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ\s]+?)(?:'?[dt][ae])\s+hava", raw, re.I)
    if m:
        city = m.group(1).strip(" ,?!.")
        if city:
            return city

    cleaned = lowered
    cleaned = re.sub(
        r"\b(bugun|yarin|su an|simdi|hava|durumu|nasil|kac|derece|"
        r"sicaklik|sicakligi|yagmur|yagacak|var mi|orada|burada|acaba|"
        r"lazim|soyle|bak|nedir|ne)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) >= 2:
        return cleaned.title()
    return default_city


def weather_reply(text: str, default_city: str = DEFAULT_CITY) -> str | None:
    if not is_weather_question(text):
        return None

    city_query = extract_weather_city(text, default_city)
    try:
        queries = [
            f"{city_query} bugun hava durumu",
            f"{city_query} hava durumu",
        ]
        degree_re = re.compile(r"[-+]?\d+(?:[,.]\d+)?\s*°\s*c?", re.I)
        result = None
        fallback = None
        for query in queries:
            results = web_search.search(query, max_results=8)
            if results and fallback is None:
                fallback = results[0]
            for item in results:
                hay = f"{item['title']} {item['snippet']}"
                if degree_re.search(hay):
                    result = item
                    break
            if result:
                break
        if result is None:
            result = fallback
        if not result:
            return f"Web'de hava durumunu aradim ama net sonuc bulamadim. Sehir adini yazar misin?"

        snippet = result["snippet"] or result["title"]
        # Keep it concise but do not invent details beyond the visible result.
        snippet = re.sub(r"\s+", " ", snippet).strip()
        domain = result["display_url"] or result["url"]
        return (
            f"Web'de aradim: {snippet}\n"
            f"Kaynak: {result['title']} ({domain})"
        )
    except Exception as e:
        return f"Hava durumunu web'de ararken hata aldim ({e}). Biraz sonra tekrar deneyebiliriz."


_CONCEPTS = {
    "doga": (
        "Doga, insan yapimi olmayan canlilarin, maddelerin, olaylarin ve sureclerin "
        "butunudur: daglar, denizler, bitkiler, hayvanlar, hava olaylari ve ekosistemler "
        "buna dahildir. Daha felsefi bakarsak doga, insanin icinde yasadigi ama tamamen "
        "kontrol edemedigi temel gerceklik alanidir. Sence dogayi daha cok canlilar "
        "olarak mi, yoksa evrenin isleyisi olarak mi dusunuyorsun?"
    ),
    "gerceklik": (
        "Gerceklik, sadece dusundugumuz ya da hayal ettigimiz seylerden bagimsiz olarak "
        "var olan ya da etkisini gosteren seylere verdigimiz addir. Gunun icinde gerceklik "
        "dedigimiz sey; gorduklerimiz, olctuklerimiz, yasadiklarimiz ve baskalariyla "
        "paylasabildigimiz deneyimlerle kurulur. Felsefede zor kisim sudur: Biz gercekligi "
        "dogrudan mi biliriz, yoksa zihnimizin yorumladigi haliyle mi? Sen bunu daha cok "
        "bilimsel mi, felsefi mi merak ediyorsun?"
    ),
    "bilinc": (
        "Bilinc, kisinin kendini, cevresini, dusuncelerini ve deneyimlerini fark edebilme "
        "halidir. Bir seyi sadece islemek degil, o seyin farkinda olmak anlamina gelir. "
        "Bu yuzden bilinc hem psikolojinin hem felsefenin zor konularindan biridir. "
        "Bunu insan bilinci olarak mi, yapay zeka bilinci olarak mi soruyorsun?"
    ),
    "hayat": (
        "Hayat, biyolojik olarak canlilarin dogma, buyume, uyum saglama, ureme ve degisme "
        "surecleridir. Daha insani anlamda ise deneyimler, iliskiler, secimler ve anlam "
        "arayisi toplamidir. Bu soruyu bilimsel anlamda mi, yoksa 'hayatin anlami' gibi "
        "felsefi anlamda mi acalim?"
    ),
    "zaman": (
        "Zaman, olaylarin once-sonra iliskisi icinde siralanmasini saglayan boyut gibi "
        "dusunulebilir. Gunluk hayatta degisimi olcmemizi saglar; felsefede ise zamanin "
        "gercekten dis dunyada mi var oldugu, yoksa zihnin deneyimi duzenleme bicimi mi "
        "oldugu tartisilir. Bunu fizik tarafindan mi, felsefe tarafindan mi merak ediyorsun?"
    ),
    "ozgurluk": (
        "Ozgurluk, kisinin dusunme, secme ve davranma konusunda zorlayici engellerden "
        "olabildigince bagimsiz olmasidir. Ama mutlak sinirsizlik degildir; baskalarinin "
        "haklari, sorumluluklar ve kosullar ozgurlugun cercevesini belirler. Sen bunu "
        "kisisel ozgurluk mu, siyasi ozgurluk mu, irade ozgurlugu mu diye soruyorsun?"
    ),
}


def _concept_key(text: str) -> str | None:
    t = _ascii_lower(text).strip(" ?!.")
    t = re.sub(r"\b(bana|kisaca|acikla|anlat|nedir|ne demek|ne demektir|ne)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for key in _CONCEPTS:
        if t == key or t.startswith(key + " ") or key in t.split():
            return key
    return None


def concept_reply(text: str) -> str | None:
    t = _ascii_lower(text)
    if not any(p in t for p in ("nedir", "ne demek", "ne demektir", "acikla", "anlat")):
        return None
    key = _concept_key(text)
    return _CONCEPTS.get(key) if key else None


def daily_story_reply(text: str) -> str | None:
    raw = text.strip()
    if not raw or "?" in raw or len(raw) > 260:
        return None
    t = _ascii_lower(raw)
    if is_greeting(raw) or is_weather_question(raw):
        return None

    daily_markers = (
        "bugun", "dun", "okula", "okul", "derse", "ders", "sinav", "ise",
        "is", "arkadas", "gezdim", "gittim", "geldim", "yaptim", "izledim",
        "oynadim", "yurudum", "yoruldum", "mutluyum", "uzgunum", "keyifli",
    )
    if not any(m in t for m in daily_markers):
        return None

    positive = any(w in t for w in ("iyi", "guzel", "keyifli", "eglenceli", "verimli", "sevdim", "mutlu"))
    negative = any(w in t for w in ("kotu", "zor", "yorucu", "sikici", "uzgun", "moralim bozuk", "stres"))

    if "okul" in t or "okula" in t or "ders" in t or "sinav" in t:
        if negative:
            return "Anladim, okul bazen yorucu geciyor. En cok hangi kisim zor geldi: dersler mi, sinavlar mi, yoksa ortam mi?"
        if positive:
            return "Buna sevindim. Okulda gunun en iyi kismi neydi; dersler mi iyi gecti, arkadaslarla zaman mi guzeldi?"
        return "Okul gununu anlatman guzel. Bugun okulda en cok aklinda kalan sey ne oldu?"

    if "ise" in t or re.search(r"\bis\b", t):
        if negative:
            return "Anladim, is gunu yorucu gecmis. Seni en cok yoran sey neydi?"
        if positive:
            return "Buna sevindim. Is tarafinda bugun en verimli ya da en iyi giden sey neydi?"
        return "Is gunun nasil gecti, daha cok yogun mu yoksa sakin miydi?"

    if negative:
        return "Anladim. Bugun biraz zor gecmis gibi. Istersen anlat, seni en cok ne yordu?"
    if positive:
        return "Buna sevindim. Bugunun en guzel kismi neydi?"
    return "Anladim. Biraz daha anlatmak ister misin; bugun en cok ne aklinda kaldi?"


def quick_reply(text: str) -> str | None:
    """Return a deterministic reply for simple/tool-backed intents."""
    weather = weather_reply(text)
    if weather:
        return weather
    concept = concept_reply(text)
    if concept:
        return concept
    daily = daily_story_reply(text)
    if daily:
        return daily
    greet = greeting_reply(text)
    if greet:
        return greet
    return None


if __name__ == "__main__":
    for q in ["nasilsin?", "hava durumu bugun kac", "Ankara'da hava nasil?"]:
        print(q)
        print(quick_reply(q))
        print()
