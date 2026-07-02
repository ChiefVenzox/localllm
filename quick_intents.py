"""
quick_intents.py
================
Small deterministic replies that should not wait for the language model.

The model is useful for open-ended chat, but some requests are better handled
before the model:
  * greetings / "nasilsin" -> stable local reply
  * daily life statements   -> reflective follow-up question
  * simple concept questions -> local Turkish explanation
  * daily planning / writing / coding requests -> practical deterministic help
  * weather questions      -> live web search snippets, no hallucination

Weather intentionally does not use a weather API. It searches the web through
HTML results and summarizes the best visible snippet. The default city is
Istanbul and can be changed with YERELLM_DEFAULT_CITY.
"""
from __future__ import annotations

import os
import re
import unicodedata

try:
    import web_search
except Exception:
    web_search = None

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


def casual_reply(text: str) -> str | None:
    raw = text.strip()
    if not raw or len(raw) > 220:
        return None
    t = _ascii_lower(raw).strip(" ?!.")

    if any(p in t for p in ("yanlis cevap", "yanlis yanit", "hatali cevap", "sacma cevap")):
        return "Haklisin, onceki cevap uygun degildi. Normal sohbet modunda devam ediyorum; nereden devam edelim?"

    if any(p in t for p in ("anliyor musun", "anladin mi", "beni anladin mi", "sorun surda", "sorun su")):
        return (
            "Evet, anladim. Sorun tek bir cevabi hafizaya yazmak degil; temel sohbet "
            "davranisinin kayit gerektirmeden duzgun calismasi gerekiyor."
        )

    if any(p in t for p in ("kayit etmemize gerek kalmasin", "kaydetmemize gerek kalmasin", "hafizaya kaydetmeden")):
        return (
            "Dogru. Temel sohbet cevaplari hafiza kaydina bagli olmamali. "
            "Bunu genel sohbet kuralindan cevaplayacak sekilde ele alalim."
        )

    if re.fullmatch(r"(tesekkur|tesekkurler|sag ol|sagol|eyvallah)( ederim)?", t):
        return "Rica ederim. Devam edelim; siradaki istegini yazabilirsin."

    if re.fullmatch(r"(tamam|peki|olur|ok|anladim|aynen)", t):
        return "Tamam. Buradayim; ne yapmak istiyorsan oradan devam edebiliriz."

    if any(p in t for p in ("ne yapiyorsun", "su an ne yapiyorsun", "burada misin")):
        return "Buradayim. Mesajlarini okuyup yardimci olmaya hazirim."

    if any(p in t for p in ("kimsin", "adin ne", "sen nesin", "bilge kimsin")):
        return "Ben Bilge; bu bilgisayarda calisan yerel yapay zeka asistaninim. Sohbet, kod, ozet ve sistem islerinde yardim ederim."

    if any(p in t for p in ("neler yapabilirsin", "ne yapabilirsin", "yardim eder misin", "bana yardimci olur musun")):
        return (
            "Sohbet edebilirim, metin duzeltebilirim, basit kod ornekleri yazabilirim, "
            "gundelik plan cikarabilirim ve bu yerel LLM sisteminin ayarlarini kontrol edebilirim."
        )

    if "nasilsin" in t or "naber" in t or "ne haber" in t:
        if any(w in t for w in ("guzel", "iyi", "fena degil", "ben de", "bende", "sen")):
            return "Ben de iyiyim. Sohbete devam edebiliriz; bugun aklinda ne var?"

    if re.fullmatch(r"(iyi|iyiyim|ben iyiyim|guzel|super|fena degil|eh iste)( sen)?", t):
        return "Buna sevindim. Ben de iyiyim; istersen bugun aklinda kalan konudan devam edelim."

    if re.fullmatch(r"(iyiyim|ben de iyiyim|bende iyiyim|iyi sayilirim|fena degilim)[, ]*(sen|ya sen|peki sen)?", t):
        return "Ben de iyiyim, tesekkur ederim. Bugun ne hakkinda konusmak istersin?"

    if "yapay zeka" in t or re.search(r"\b(ai|llm|model)\b", t):
        if any(w in t for w in ("nedir", "ne demek", "acikla", "anlat")):
            return (
                "Yapay zeka, bilgisayarin veriden kalip ogrenip tahmin, metin, ses, goruntu "
                "ve karar uretmesini saglayan sistemlerin genel adidir. Bizim burada kurdugumuz "
                "sistem de mesajlardan ogrenip yerel worker'larla cevap uretmeye calisiyor."
            )
        return "Yapay zeka iyi bir konu. Istersen nasil calistigini, model egitimi tarafini veya kendi sistemimizi konusabiliriz."

    return None


def arithmetic_reply(text: str) -> str | None:
    t = _ascii_lower(text).replace(",", ".")
    patterns = (
        (r"(-?\d+(?:\.\d+)?)\s*(?:\+|arti|topla|ile)\s*(-?\d+(?:\.\d+)?)", "+"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:-|eksi|cikar)\s*(-?\d+(?:\.\d+)?)", "-"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:x|\*|carpi|kere)\s*(-?\d+(?:\.\d+)?)", "*"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:/|bolu)\s*(-?\d+(?:\.\d+)?)", "/"),
    )
    for pattern, op in patterns:
        match = re.search(pattern, t)
        if not match:
            continue
        left = float(match.group(1))
        right = float(match.group(2))
        if op == "/" and right == 0:
            return "Sifira bolme tanimsizdir."
        if op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op == "*":
            value = left * right
        else:
            value = left / right
        shown = str(int(value)) if value.is_integer() else str(round(value, 6)).rstrip("0").rstrip(".")
        return f"Sonuc: {shown}."
    return None


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
    if web_search is None:
        return "Web arama modulu bu worker'da henuz yok. Worker patch senkronundan sonra tekrar deneyebiliriz."

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


def factual_reply(text: str) -> str | None:
    # Kalici/hard-coded bilgi hafizasi kullanma. Bilgi sorulari server tarafinda
    # anlik lookup'a yonlendirilir; burada sadece arac fonksiyon sozlesmesi kalir.
    return None


def daily_story_reply(text: str) -> str | None:
    raw = text.strip()
    if not raw or "?" in raw or len(raw) > 260:
        return None
    t = _ascii_lower(raw)
    if is_greeting(raw) or is_weather_question(raw):
        return None
    # Gunluk sohbet yalnizca kullanicinin kendi gununu/anini anlattigi
    # ifadeler icindir. Teknik veya bilgi isteyen komutlari burada kesme.
    request_markers = (
        "acikla", "anlat", "nedir", "ne demek", "fark", "karsilastir",
        "hesapla", "yaz", "olustur", "hazirla", "listele", "madde",
        "python", "javascript", "kod", "fonksiyon", "tuple", "liste",
    )
    if any(marker in t for marker in request_markers):
        return None

    phrase_markers = (
        "bugun", "dun", "okula", "okul", "derse", "ders", "sinav", "ise",
        "arkadas", "gezdim", "gittim", "geldim", "yaptim", "izledim",
        "oynadim", "yurudum", "yoruldum", "mutluyum", "uzgunum", "keyifli",
    )
    word_markers = {"is"}
    words = set(re.findall(r"\b\w+\b", t))
    if not any(m in t for m in phrase_markers) and not (words & word_markers):
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


def daily_task_reply(text: str) -> str | None:
    t = _ascii_lower(text)
    if any(w in t for w in ("kod", "python", "javascript", "fastapi", "docker", "html", "css")):
        return None

    if any(w in t for w in ("ne pisir", "ne yemek", "yemek oner", "aksam yemegi", "kahvalti")):
        return (
            "Pratik bir secim yapalim:\n"
            "1. Evde yumurta/sebze varsa menemen veya sebzeli omlet yap.\n"
            "2. Tavuk varsa tavada tavuk + pilav/makarna hizli olur.\n"
            "3. Hafif bir sey istiyorsan corba + salata iyi gider.\n"
            "Elindeki malzemeleri yazarsan daha net menu cikaririm."
        )

    if any(w in t for w in ("alisveris", "market listesi", "ne alayim")):
        return (
            "Temel alisveris listesi:\n"
            "- Protein: yumurta, tavuk, yogurt veya peynir\n"
            "- Karbonhidrat: pirinc, makarna, ekmek veya patates\n"
            "- Sebze/meyve: domates, salatalik, yesillik, muz/elma\n"
            "- Ek: zeytinyagi, baharat, cay/kahve\n"
            "Butceni ve kac gunluk istedigini yazarsan listeyi daraltirim."
        )

    if any(w in t for w in ("ders calis", "sinava", "calisma plani", "odaklan")):
        return (
            "Basit calisma plani:\n"
            "1. 25 dakika tek konu sec ve sadece ona calis.\n"
            "2. 5 dakika mola ver.\n"
            "3. Sonra 20 dakika soru coz veya tekrar yap.\n"
            "4. En sonda 3 maddelik ozet yaz.\n"
            "Konu ve sureyi yazarsan sana saat saat plan cikaririm."
        )

    if any(w in t for w in ("spor", "egzersiz", "antrenman", "hareket")):
        return (
            "Evde basit 20 dakikalik rutin:\n"
            "1. 3 dk isinma\n"
            "2. 3 tur: 10 squat, 8 sinav, 20 sn plank\n"
            "3. 3 dk esneme\n"
            "Yeni basliyorsan tekrar sayisini azalt; agri olursa dur."
        )

    if any(w in t for w in ("temizlik", "oda topla", "evi topla", "duzenle")):
        return (
            "15 dakikalik toparlama:\n"
            "1. Copleri topla.\n"
            "2. Masadaki gereksizleri tek kutuya koy.\n"
            "3. Kirli camasir/bulasiklari ayir.\n"
            "4. Son 3 dakikada yuzeyleri sil.\n"
            "Tam temizlik degil, hizli kontrol hissi hedefle."
        )

    if any(w in t for w in ("uyuyamiyorum", "uyku", "gece uyan", "erken kalk")):
        return (
            "Uyku icin pratik adimlar:\n"
            "1. Ekrani 20-30 dakika azalt.\n"
            "2. Odayi serin ve karanlik tut.\n"
            "3. Kafandaki isleri kagida yaz.\n"
            "4. Kafeini gec saatlere birakma.\n"
            "Uzun suruyorsa bir uzmana danismak iyi olur."
        )

    if any(w in t for w in ("moralim", "stres", "kaygili", "canim sikiliyor", "ne yapayim")):
        return (
            "Su an en iyi hamle kucuk ve somut olsun:\n"
            "1. 2 dakika nefesini yavaslat.\n"
            "2. Tek bir isi sec: su icmek, dus almak, 5 dakika yurume gibi.\n"
            "3. Sonra ne hissettigini tekrar yaz.\n"
            "Cozum aramadan once sistemi biraz sakinlestirmek ise yarar."
        )

    return None


def is_reasoning_prompt(text: str) -> bool:
    t = _ascii_lower(text)
    if re.search(r"\d+\s*[-+x*/]\s*\d+", t):
        return True
    markers = (
        "kac", "arti", "eksi", "carpi", "kere", "bolu", "topla", "cikar",
        "sonuc", "hesap", "sayi", "problem", "buyuk", "kucuk", "fazla",
        "az", "mantik", "siradaki", "yarin", "dun", "hangi gun",
    )
    return any(marker in t for marker in markers)


def is_bad_model_reply(answer: str, prompt: str = "") -> bool:
    raw = (answer or "").strip()
    if not raw:
        return True
    t = _ascii_lower(raw)
    p = _ascii_lower(prompt)
    if len(raw) <= 16 and re.fullmatch(r"(tabii|tamam|olur|anladim|peki)[?.! ]*", t):
        return True
    info_request = any(w in p for w in (
        "acikla", "anlat", "nedir", "ne demek", "fark", "karsilastir",
        "madde", "listele", "ornek", "neden", "nasil", "neresi",
        "kimdir", "ne zaman", "baskent", "başkent",
    ))
    if info_request and len(raw) < 40:
        return True
    if info_request and any(fragment in t for fragment in ("uy eder", "bir dille bir dille", "orta bir dille")):
        return True
    code_prompt = any(w in p for w in (
        "kod", "python", "javascript", "html", "css", "php", "fastapi", "docker",
        "fonksiyon", "script", "api", "program",
    ))
    greeting_prompt = is_greeting(prompt)
    casual_prompt = greeting_prompt or bool(re.fullmatch(
        r"(iyi|iyiyim|ben iyiyim|ben de iyiyim|bende iyiyim|guzel|super|fena degil|eh iste)( sen| ya sen| peki sen)?",
        p.strip(" ?!."),
    )) or any(w in p for w in ("nasilsin", "naber", "ne haber", "tesekkur", "sag ol", "tamam", "anliyor musun"))
    plain_chat_prompt = not code_prompt and not is_reasoning_prompt(prompt) and not info_request
    if not code_prompt and any(fragment in t for fragment in ("print(", "def ", "function ", "console.log", "<?php", "<html")):
        return True
    if casual_prompt and (re.search(r"\d+\s*[-+x*/=]\s*\d+", t) or "dakikada" in t or "bir dille" in t):
        return True
    if greeting_prompt and (
        re.search(r"-?\d+", t)
        or any(fragment in t for fragment in ("antik", "caglarda", "sonuc", "hesap", "denklem"))
    ):
        return True
    if plain_chat_prompt and re.search(r"\d{4,}", t) and re.search(r"[-+x*/=]", t):
        return True
    if plain_chat_prompt and any(fragment in t for fragment in ("bir dille bir", "gibi uy", "ntik artarak")):
        return True
    words = re.findall(r"\b\w+\b", t)
    for size in (2, 3):
        grams = [" ".join(words[i:i + size]) for i in range(0, max(0, len(words) - size + 1))]
        if grams and max(grams.count(gram) for gram in set(grams)) >= 3:
            return True
    if "python" in p and "liste" in p and "tuple" in p:
        expected = ("liste", "tuple", "degistir", "degismez", "parantez", "eleman")
        if sum(1 for word in expected if word in t) < 2:
            return True
    settings_prompt = any(w in p for w in ("parametre", "ayar", "sicaklik", "temperature", "top_p", "top_k"))
    if re.search(r"\b(sicaklik|temperature)\s*=\s*0(?:\.0+)?\b", t) and not settings_prompt:
        return True
    if re.match(r"^dusunelim\s*:", t) and not is_reasoning_prompt(prompt):
        return True
    if t in {"none", "null", "undefined", "nan"}:
        return True
    return False


def programming_fallback_reply(text: str) -> str | None:
    t = _ascii_lower(text)
    if "html" in t and any(w in t for w in ("kod", "ornek", "dilinde", "istiyorum", "olsun", "yapi", "sayfa", "form")):
        return (
            "HTML ile basit ve calisir bir sayfa ornegi:\n\n"
            "```html\n"
            "<!doctype html>\n"
            "<html lang=\"tr\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>Merhaba</title>\n"
            "  <style>\n"
            "    body { font-family: Arial, sans-serif; margin: 40px; }\n"
            "    button { padding: 10px 14px; cursor: pointer; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>Merhaba</h1>\n"
            "  <p>Bu basit bir HTML sayfasidir.</p>\n"
            "  <button onclick=\"alert('Calisiyor')\">Tikla</button>\n"
            "</body>\n"
            "</html>\n"
            "```\n\n"
            "Bunu `index.html` olarak kaydedip tarayicida acabilirsin."
        )
    if "php" in t and any(w in t for w in ("kod", "ornek", "dilinde", "istiyorum", "olsun", "form", "api")):
        return (
            "PHP ile basit ve guvenli bir ornek:\n\n"
            "```php\n"
            "<?php\n"
            "$isim = $_GET[\"isim\"] ?? \"ziyaretci\";\n"
            "$guvenliIsim = htmlspecialchars($isim, ENT_QUOTES, \"UTF-8\");\n"
            "echo \"Merhaba, \" . $guvenliIsim;\n"
            "?>\n"
            "```\n\n"
            "Calistirma:\n"
            "```bash\n"
            "php -S 127.0.0.1:8080\n"
            "```\n"
            "Sonra `http://127.0.0.1:8080/?isim=Hakan` adresini ac."
        )
    if "python" in t and "liste" in t and "tuple" in t and any(w in t for w in ("fark", "karsilastir", "acikla")):
        return (
            "1. Liste degistirilebilir; eleman ekleyebilir, silebilir veya mevcut elemani "
            "degistirebilirsin. Tuple olusturulduktan sonra degistirilemez.\n"
            "2. Liste genelde koseli parantezle yazilir: `[1, 2]`. Tuple genelde parantezle "
            "yazilir: `(1, 2)`. Degismeyecek veriler icin tuple daha uygun olur."
        )
    if "python" in t and "fonksiyon" in t and any(w in t for w in ("yaz", "ornek", "kod")):
        return (
            "Basit bir Python fonksiyonu soyle yazilir:\n\n"
            "```python\n"
            "def topla(a, b):\n"
            "    return a + b\n"
            "```\n\n"
            "`def` fonksiyonu baslatir, `return` sonucu geri verir."
        )
    if "fastapi" in t and any(w in t for w in ("endpoint", "api", "ornek", "kod", "yaz")):
        return (
            "FastAPI icin basit endpoint ornegi:\n\n"
            "```python\n"
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/health')\n"
            "def health():\n"
            "    return {'ok': True}\n"
            "```\n\n"
            "Calistirma:\n"
            "```bash\n"
            "uvicorn app:app --host 0.0.0.0 --port 8000\n"
            "```"
        )
    if "docker" in t and ("fastapi" in t or "api" in t) and any(w in t for w in ("dockerfile", "paket", "container", "ornek")):
        return (
            "FastAPI icin sade Dockerfile:\n\n"
            "```dockerfile\n"
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8000\n"
            "CMD [\"uvicorn\", \"server.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            "```\n\n"
            "Buyuk model/veri dosyalarini image'e koyma; volume olarak bagla."
        )
    if "javascript" in t and any(w in t for w in ("fetch", "api", "istek", "ornek", "kod")):
        return (
            "JavaScript ile API istegi ornegi:\n\n"
            "```javascript\n"
            "async function getHealth() {\n"
            "  const res = await fetch('/api/health');\n"
            "  if (!res.ok) throw new Error('API hatasi');\n"
            "  return await res.json();\n"
            "}\n"
            "```\n"
        )
    if any(w in t for w in ("kod oner", "kod yaz", "kod yazabilir", "nasil kodlarim", "fonksiyon yaz")):
        return (
            "Evet, kod yazabilirim. Hangi dilde ve ne yapmasini istedigini yazarsan "
            "dogrudan calisir ornek cikaririm.\n\n"
            "Ornek: `HTML ile basit sayfa`, `PHP ile form`, `Python ile dosya oku`."
        )
    return None


def text_work_reply(text: str) -> str | None:
    raw = text.strip()
    t = _ascii_lower(raw)
    wants_text = any(w in t for w in (
        "metni duzelt", "yaziyi duzelt", "imla", "daha resmi", "daha kibar",
        "mail yaz", "mesaj yaz", "ozetle", "baslik oner", "duzenle",
    ))
    if not wants_text:
        return None

    payload = ""
    quoted = re.search(r"['\"]([^'\"]{4,})['\"]", raw)
    if quoted:
        payload = quoted.group(1).strip()
    elif ":" in raw:
        payload = raw.split(":", 1)[1].strip()
    elif "\n" in raw:
        payload = raw.split("\n", 1)[1].strip()

    if any(w in t for w in ("mail yaz", "mesaj yaz")):
        tone = "resmi" if "resmi" in t else "kibar"
        topic = payload or "konuyu"
        return (
            f"{tone.title()} taslak:\n\n"
            "Merhaba,\n\n"
            f"{topic} hakkinda size kisaca bilgi vermek istiyorum. "
            "Uygun oldugunuzda degerlendirmenizi rica ederim.\n\n"
            "Tesekkurler."
        )

    if "ozetle" in t:
        if not payload:
            return "Ozetlemem icin metni iki nokta ust uste sonrasina yaz: `Ozetle: ...`"
        sentences = re.split(r"(?<=[.!?])\s+", payload)
        short = " ".join(sentences[:2]).strip()
        return f"Kisa ozet: {short}"

    if "baslik oner" in t:
        topic = payload or "bu konu"
        clean = re.sub(r"\s+", " ", topic).strip(" .")
        return f"Baslik onerisi: {clean[:60].strip().title()}"

    if not payload:
        return "Duzeltmem icin metni iki nokta ust uste sonrasina yaz: `Metni duzelt: ...`"

    fixed = re.sub(r"\s+", " ", payload).strip()
    if fixed:
        fixed = fixed[0].upper() + fixed[1:]
    fixed = re.sub(r"^(Merhaba)(\s+)", r"\1, ", fixed)
    fixed = re.sub(r"\b(kusura bakmay[ıi]n)\b", "Kusura bakmayın", fixed, flags=re.I)
    fixed = re.sub(r"(ge[cç] kald[ıi]m)\s+(Kusura)", r"\1. \2", fixed, flags=re.I)
    if fixed and fixed[-1] not in ".!?":
        fixed += "."
    if any(w in t for w in ("daha resmi", "daha kibar")):
        fixed = "Merhaba, " + fixed[0].lower() + fixed[1:]
    return f"Duzeltilmis metin:\n{fixed}"


def should_lookup(text: str) -> bool:
    t = _ascii_lower(text)
    if any(w in t for w in ("metni", "mail", "mesaj", "ozetle", "duzelt")):
        return False
    tech_terms = ("kod", "python", "javascript", "fastapi", "docker", "api", "html", "css")
    build_terms = ("kod yaz", "kod oner", "ornek", "endpoint", "fonksiyon", "dockerfile", "container", "paketle")
    if any(term in t for term in tech_terms) and any(term in t for term in build_terms):
        return False
    return any(w in t for w in ("nedir", "kimdir", "ne demek", "neresi", "ne zaman", "kim yapti", "kim yazdi"))


def fallback_reply(text: str) -> str:
    for fn in (
        arithmetic_reply,
        weather_reply,
        concept_reply,
        factual_reply,
        programming_fallback_reply,
        text_work_reply,
        daily_task_reply,
        greeting_reply,
        casual_reply,
        daily_story_reply,
    ):
        reply = fn(text)
        if reply:
            return reply
    return "Anladim. Bunu biraz daha acalim; tam olarak hangi kismi konusmak istersin?"


def clean_model_reply(prompt: str, answer: str) -> tuple[str, bool]:
    if is_bad_model_reply(answer, prompt):
        return fallback_reply(prompt), True
    return answer, False


def quick_reply(text: str) -> str | None:
    """Return a deterministic reply for simple/tool-backed intents."""
    arithmetic = arithmetic_reply(text)
    if arithmetic:
        return arithmetic
    weather = weather_reply(text)
    if weather:
        return weather
    code = programming_fallback_reply(text)
    if code:
        return code
    writing = text_work_reply(text)
    if writing:
        return writing
    daily_task = daily_task_reply(text)
    if daily_task:
        return daily_task
    factual = factual_reply(text)
    if factual:
        return factual
    concept = concept_reply(text)
    if concept:
        return concept
    greet = greeting_reply(text)
    if greet:
        return greet
    casual = casual_reply(text)
    if casual:
        return casual
    daily = daily_story_reply(text)
    if daily:
        return daily
    return None


def contextual_reply(messages: list[dict]) -> str | None:
    """Return a quick reply that can use the current chat history."""
    if not messages:
        return None

    normalized: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
        else:
            role = str(getattr(message, "role", "") or "")
            content = str(getattr(message, "content", "") or "")
        normalized.append((role, content))

    last_index = -1
    last_user = ""
    for index in range(len(normalized) - 1, -1, -1):
        role, content = normalized[index]
        if role == "user" and content.strip():
            last_index = index
            last_user = content.strip()
            break
    if not last_user:
        return None

    direct = quick_reply(last_user)
    if direct:
        return direct

    previous = " ".join(content for _, content in normalized[max(0, last_index - 5):last_index])
    context = _ascii_lower(previous)
    current = _ascii_lower(last_user).strip(" ?!.")
    code_context = any(w in context for w in (
        "kod", "code", "fonksiyon", "hangi dil", "dilde", "html", "php", "javascript", "python",
    ))
    if not code_context:
        return None

    language_only = {
        "html": "html kod ornegi",
        "php": "php kod ornegi",
        "javascript": "javascript fetch api kod ornegi",
        "js": "javascript fetch api kod ornegi",
        "python": "python fonksiyon kod ornegi",
    }
    if current in language_only:
        return programming_fallback_reply(language_only[current])
    for language, prompt in language_only.items():
        if language in current and any(w in current for w in ("dilinde", "olsun", "istiyorum", "ornek", "kod")):
            return programming_fallback_reply(prompt)
    return None


if __name__ == "__main__":
    for q in ["nasilsin?", "guzel sen nasilsin", "yapay zeka kaldi", "12 + 8 kac eder?", "hava durumu bugun kac", "Ankara'da hava nasil?"]:
        print(q)
        print(quick_reply(q))
        print()
