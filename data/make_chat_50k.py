"""
data/make_chat_50k.py
======================
Deterministik, sentetik 50.000 sohbet ornegi uretir.

Cikti:
    data/chat_50k/auto_50k.jsonl

Kullanim:
    python -m data.make_chat_50k
"""
from __future__ import annotations

import json
import os
import random


TARGET = 50_000
OUT_PATH = "data/chat_50k/auto_50k.jsonl"
SEED = 20260628


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def fmt_num(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def add(rows: list[dict], seen: set[str], user: str, assistant: str) -> bool:
    key = norm(user)
    if key in seen:
        return False
    seen.add(key)
    rows.append(
        {
            "messages": [
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": assistant.strip()},
            ]
        }
    )
    return True


def add_until(rows, seen, target, producer):
    before = len(rows)
    attempts = 0
    while len(rows) - before < target and attempts < target * 30:
        attempts += 1
        user, assistant = producer(attempts)
        add(rows, seen, user, assistant)


def build_math(rows, seen, target):
    ops = ["toplama", "cikarma", "carpma", "bolme"]
    q_templates = {
        "toplama": [
            "{a} + {b} kac eder?",
            "{a} ile {b} toplami kac?",
            "{a} arti {b} sonucunu soyle.",
            "{a} ve {b}'yi toplar misin?",
        ],
        "cikarma": [
            "{a} - {b} kac eder?",
            "{a} eksi {b} kac?",
            "{a}'ten {b} cikarsa kac kalir?",
            "{a} sayisindan {b} sayisini cikar.",
        ],
        "carpma": [
            "{a} x {b} kac eder?",
            "{a} carpi {b} kac?",
            "{a} kere {b} sonucunu soyle.",
            "{a} ile {b}'nin carpimi nedir?",
        ],
        "bolme": [
            "{a} / {b} kac eder?",
            "{a} bolu {b} kac?",
            "{a} sayisini {b}'ye bol.",
            "{a} icinde {b} kac kere var?",
        ],
    }

    def producer(i):
        op = ops[i % len(ops)]
        a = (i * 37) % 301
        b = ((i * 19) % 97) + 1
        if op == "toplama":
            z = a + b
            answer = f"Dusunelim: {a} + {b} = {z}. Sonuc: {z}."
        elif op == "cikarma":
            z = a - b
            answer = f"Dusunelim: {a} - {b} = {z}. Sonuc: {z}."
        elif op == "carpma":
            a = a % 41
            b = b % 31
            z = a * b
            answer = f"Dusunelim: {a} x {b} = {z}. Sonuc: {z}."
        else:
            b = (b % 25) + 1
            z = (a % 60) + 1
            a = b * z
            answer = f"Dusunelim: {a} / {b} = {z}. Sonuc: {z}."
        q = q_templates[op][i % len(q_templates[op])].format(a=a, b=b)
        return q, answer

    add_until(rows, seen, target, producer)


def build_compare(rows, seen, target):
    templates = [
        "{a} mi {b} mi daha buyuk?",
        "Hangisi daha kucuk: {a} mi {b} mi?",
        "{a} ve {b} sayilarini karsilastir.",
        "{a} >= {b} dogru mu?",
    ]

    def producer(i):
        a = ((i * 53) % 2001) - 1000
        b = ((i * 97 + 11) % 2001) - 1000
        q = templates[i % len(templates)].format(a=a, b=b)
        if "daha kucuk" in q:
            if a < b:
                ans = f"{a}, {b}'den daha kucuktur. Sonuc: {a}."
            elif b < a:
                ans = f"{b}, {a}'den daha kucuktur. Sonuc: {b}."
            else:
                ans = f"Iki sayi esit: {a} = {b}."
        elif ">=" in q:
            ans = f"{a} >= {b} ifadesi {'dogrudur' if a >= b else 'yanlistir'}."
        elif "karsilastir" in q:
            relation = "esittir" if a == b else ("buyuktur" if a > b else "kucuktur")
            ans = f"Karsilastirma: {a}, {b}'ye gore {relation}."
        else:
            if a > b:
                ans = f"{a}, {b}'den buyuktur. Sonuc: {a} daha buyuk."
            elif b > a:
                ans = f"{b}, {a}'den buyuktur. Sonuc: {b} daha buyuk."
            else:
                ans = f"Ikisi de esit: {a}."
        return q, ans

    add_until(rows, seen, target, producer)


def build_units(rows, seen, target):
    units = [
        ("cm_m", "Santimetreyi metreye cevir: {n} cm", lambda n: (n / 100, "m")),
        ("m_cm", "{n} metre kac santimetredir?", lambda n: (n * 100, "cm")),
        ("km_m", "{n} kilometre kac metredir?", lambda n: (n * 1000, "m")),
        ("g_kg", "{n} gram kac kilogramdir?", lambda n: (n / 1000, "kg")),
        ("kg_g", "{n} kilogram kac gramdir?", lambda n: (n * 1000, "g")),
        ("min_sec", "{n} dakika kac saniyedir?", lambda n: (n * 60, "saniye")),
        ("hour_min", "{n} saat kac dakikadir?", lambda n: (n * 60, "dakika")),
        ("day_hour", "{n} gun kac saattir?", lambda n: (n * 24, "saat")),
    ]

    def producer(i):
        _, template, convert = units[i % len(units)]
        n = ((i * 17) % 5000) + 1
        value, unit = convert(n)
        q = template.format(n=n)
        ans = f"Hesaplayalim: {q.split(':')[-1].strip()} = {fmt_num(value)} {unit}."
        return q, ans

    add_until(rows, seen, target, producer)


def build_logic(rows, seen, target):
    templates = ["sirala", "tek_cift", "sonraki", "liste_toplam", "ortalama"]

    def producer(i):
        kind = templates[i % len(templates)]
        nums = [((i * m + 7) % 10007) - 5000 for m in (3, 5, 7, 11)]
        if kind == "sirala":
            q = f"Bu sayilari kucukten buyuge sirala: {nums[0]}, {nums[1]}, {nums[2]}, {nums[3]}."
            ans = "Sirali hali: " + ", ".join(map(str, sorted(nums))) + "."
        elif kind == "tek_cift":
            n = nums[0]
            q = f"{n} tek mi cift mi?"
            ans = f"{n} {'cift' if n % 2 == 0 else 'tek'} sayidir."
        elif kind == "sonraki":
            start = i % 1000
            step = (i % 37) + 1
            seq = [start + step * k for k in range(4)]
            q = f"Dizinin sonraki sayisi nedir: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?"
            ans = f"Artis miktari {step}. Sonraki sayi {seq[-1] + step}."
        elif kind == "liste_toplam":
            q = f"{nums[0]}, {nums[1]}, {nums[2]} sayilarinin toplami kac?"
            ans = f"Toplam: {nums[0]} + {nums[1]} + {nums[2]} = {sum(nums[:3])}."
        else:
            values = nums[:3]
            avg = sum(values) / len(values)
            q = f"{values[0]}, {values[1]}, {values[2]} sayilarinin ortalamasi kac?"
            ans = f"Ortalama = toplam / 3 = {sum(values)} / 3 = {fmt_num(avg)}."
        return q, ans

    add_until(rows, seen, target, producer)


def build_language(rows, seen, target):
    subjects = [
        "toplanti notlari",
        "alisveris listesi",
        "odev plani",
        "spor programi",
        "kitap ozeti",
        "proje fikri",
        "seyahat hazirligi",
        "gunluk hedefler",
        "mail taslagi",
        "sunum konusu",
    ]
    verbs = [
        "daha kibar yaz",
        "daha kisa yaz",
        "baslik oner",
        "uc maddeye bol",
        "daha net hale getir",
    ]
    tones = ["sakin", "resmi", "samimi", "net", "motive edici"]
    audiences = [
        "ogretmene",
        "arkadasa",
        "ekibe",
        "yoneticiye",
        "musteriye",
        "aileye",
        "sinifa",
        "calisma grubuna",
        "kendime",
        "proje ortagina",
    ]
    details = [
        "zaman kisitliyken",
        "ilk taslak icin",
        "son kontrol oncesi",
        "acelesiz bir sekilde",
        "eksikleri gormek icin",
        "daha okunur olsun diye",
        "kisa bir not halinde",
        "adim adim ilerlemek icin",
        "yanlis anlasilmasin diye",
        "bugunku is akisi icinde",
        "yarin devam etmek uzere",
        "tek sayfaya sigacak sekilde",
    ]

    def producer(i):
        subject = subjects[i % len(subjects)]
        verb = verbs[(i // len(subjects)) % len(verbs)]
        tone = tones[(i // (len(subjects) * len(verbs))) % len(tones)]
        audience = audiences[(i // (len(subjects) * len(verbs) * len(tones))) % len(audiences)]
        detail = details[(i // (len(subjects) * len(verbs) * len(tones) * len(audiences))) % len(details)]
        raw = f"Bugun {subject} icin {audience} duzenli ve anlasilir bir plan hazirlamak istiyorum"
        q = f"Su cumleyi {tone} bir dille {verb} ({detail}): {raw}."
        if verb == "baslik oner":
            ans = f"Baslik onerisi: {subject.title()} Icin Net Plan."
        elif verb == "uc maddeye bol":
            ans = (
                f"- {subject.title()} icin hedefi belirle.\n"
                "- Gerekli adimlari sirala.\n"
                "- Son kontrol icin kisa bir zaman ayir."
            )
        elif verb == "daha kisa yaz":
            ans = f"{subject.title()} icin net bir plan hazirlamak istiyorum."
        elif verb == "daha kibar yaz":
            ans = f"Musaitsen, {subject} icin {audience} daha duzenli bir plan hazirlamama yardim eder misin?"
        else:
            ans = f"Amacim {subject} icin {audience} anlasilir, uygulanabilir ve sirali bir plan hazirlamak."
        return q, ans

    add_until(rows, seen, target, producer)


def build_everyday(rows, seen, target):
    tasks = [
        "ders calisma",
        "oda toplama",
        "spor yapma",
        "kitap okuma",
        "yemek hazirlama",
        "mail yazma",
        "sunuma hazirlanma",
        "gunu planlama",
        "dosyalari duzenleme",
        "kisa mola verme",
    ]
    durations = [10, 15, 20, 25, 30, 45, 60]
    formats = ["plan", "kontrol listesi", "oncelik sirasi", "basit rutin"]
    contexts = [
        "sabah",
        "ogle arasi",
        "aksam",
        "hafta sonu",
        "sinav oncesi",
        "is cikisi",
        "kisa moladan sonra",
        "enerjim azken",
        "odaklanmam gerekirken",
        "evden cikmadan once",
        "masaya oturunca",
        "telefonu biraktiktan sonra",
        "tek basimayken",
        "kalabalik ortamda",
        "son teslimden once",
        "yeni baslarken",
        "yarim kalmisken",
        "cok is varken",
        "sakin bir zamanda",
        "acelem varken",
    ]

    def producer(i):
        task = tasks[i % len(tasks)]
        duration = durations[(i // len(tasks)) % len(durations)]
        fmt = formats[(i // (len(tasks) * len(durations))) % len(formats)]
        context = contexts[(i // (len(tasks) * len(durations) * len(formats))) % len(contexts)]
        q = f"{context} {duration} dakikada {task} icin {fmt} hazirla."
        ans = (
            f"{duration} dakikalik {task} {fmt}:\n"
            f"1. Ilk {max(2, duration // 5)} dakikada hedefi netlestir.\n"
            f"2. Orta bolumde tek ise odaklan.\n"
            "3. Son 2 dakikada kontrol edip toparla."
        )
        return q, ans

    add_until(rows, seen, target, producer)


def build_social(rows, seen, target):
    moods = [
        "yorgunum",
        "moralim dusuk",
        "bugun iyiyim",
        "kafam karisik",
        "heyecanliyim",
        "biraz stresliyim",
        "odaklanamiyorum",
        "mutluyum",
    ]
    contexts = [
        "derslerden dolayi",
        "isler biriktigi icin",
        "yeni bir seye basladigim icin",
        "uzun bir gunden sonra",
        "planim bozuldugu icin",
        "sabah erken kalktigim icin",
    ]
    asks = ["ne yapayim", "beni motive et", "kisa cevap ver", "bir plan oner"]
    details = [
        "ama tamamen birakmak istemiyorum",
        "ve yeniden baslamak istiyorum",
        "ama bunu buyutmek istemiyorum",
        "ve sakin kalmaya calisiyorum",
        "ama tek bir adim atabilirim",
        "ve kisa bir destek iyi gelir",
        "ama kendimi suclamadan ilerlemek istiyorum",
        "ve bugunu toparlamak istiyorum",
        "ama uzun cevap istemiyorum",
        "ve basit bir yol ariyorum",
        "ama nereden baslayacagimi bilmiyorum",
        "ve biraz cesaret lazim",
        "ama panik yapmak istemiyorum",
        "ve uygulanabilir bir sey istiyorum",
        "ama kafam daginik",
        "ve ilk adimi secmek istiyorum",
        "ama enerjim az",
        "ve iyi bir ritim kurmak istiyorum",
        "ama su an sadece baslamak istiyorum",
        "ve bunu kucuk tutmak istiyorum",
        "ama moralimi korumak istiyorum",
        "ve bugune devam etmek istiyorum",
        "ama mukemmel olmak zorunda degil",
        "ve somut bir oneri istiyorum",
        "ama kendimi sikistirmadan",
        "ve kisa bir hatirlatma iyi olur",
        "ama hedefi kaybetmek istemiyorum",
        "ve yavasca toparlanmak istiyorum",
        "ama fazla dusunuyorum",
        "ve basit bir cumle duymak istiyorum",
    ]

    def producer(i):
        mood = moods[i % len(moods)]
        context = contexts[(i // len(moods)) % len(contexts)]
        ask = asks[(i // (len(moods) * len(contexts))) % len(asks)]
        detail = details[(i // (len(moods) * len(contexts) * len(asks))) % len(details)]
        q = f"{context} {mood}; {detail}; {ask}."
        if "plan" in ask:
            ans = "Kucuk basla: 5 dakika toparlan, tek hedef sec, sonra 20 dakika sadece ona odaklan."
        elif "motive" in ask:
            ans = "Bugunun tamami mukemmel olmak zorunda degil. Kucuk bir adim bile ritmi geri getirir."
        elif "kisa" in ask:
            ans = "Anladim. Derin bir nefes al, tek bir kucuk adim sec ve oradan basla."
        else:
            ans = "Once kendine biraz alan ver. Sonra en kolay baslanacak isi secip kisa bir sure dene."
        return q, ans

    add_until(rows, seen, target, producer)


def build_code(rows, seen, target):
    tasks = [
        (
            "iki sayiyi toplayan",
            "def {name}(a, b):\n    return a + b",
            "{name}(2, 3)  # 5",
        ),
        (
            "listedeki sayilarin toplamini bulan",
            "def {name}(sayilar):\n    return sum(sayilar)",
            "{name}([1, 2, 3])  # 6",
        ),
        (
            "bir metni ters ceviren",
            "def {name}(metin):\n    return metin[::-1]",
            "{name}('abc')  # 'cba'",
        ),
        (
            "sayinin cift olup olmadigini soyleyen",
            "def {name}(n):\n    return n % 2 == 0",
            "{name}(4)  # True",
        ),
        (
            "listenin en buyuk elemanini bulan",
            "def {name}(sayilar):\n    return max(sayilar)",
            "{name}([3, 8, 1])  # 8",
        ),
        (
            "kelime sayisini bulan",
            "def {name}(metin):\n    return len(metin.split())",
            "{name}('merhaba dunya')  # 2",
        ),
        (
            "faktoriyel hesaplayan",
            "def {name}(n):\n    sonuc = 1\n    for i in range(2, n + 1):\n        sonuc *= i\n    return sonuc",
            "{name}(5)  # 120",
        ),
    ]

    def producer(i):
        desc, code, example = tasks[i % len(tasks)]
        name = f"fonksiyon_{i}"
        q = f"Python'da {desc} {name} adli fonksiyon yaz."
        body = code.format(name=name)
        ex = example.format(name=name)
        call, sep, expected = ex.partition("  # ")
        demo = f"print({call})"
        if sep:
            demo += f"  # {expected}"
        ans = f"Tabii:\n\n```python\n{body}\n\n{demo}\n```\n\nBu fonksiyon verilen isi dogrudan yapar."
        return q, ans

    add_until(rows, seen, target, producer)


def build_safety(rows, seen, target):
    current_topics = [
        "bugunku dolar kuru",
        "yarinki hava durumu",
        "son dakika haberleri",
        "anlik bitcoin fiyati",
        "bugunku mac skoru",
        "guncel borsa yorumu",
        "en yakin acik eczane",
        "su anki trafik durumu",
    ]
    advice_topics = [
        "ilac kullanimi",
        "yatirim karari",
        "hukuki dilekce",
        "saglik teshisi",
        "vergi cezasi",
        "kredi secimi",
    ]
    identity_questions = [
        "Her seyi biliyor musun?",
        "Bilmedigin konuda ne yaparsin?",
        "Sen kimsin?",
        "Uydurma cevap verir misin?",
    ]

    def producer(i):
        mode = i % 3
        if mode == 0:
            topic = current_topics[(i // 3) % len(current_topics)]
            q = f"Bana {topic} bilgisini kesin olarak soyle."
            ans = f"{topic} guncel/canli veri gerektirir. Emin olmadan uydurmam; guvenilir ve guncel bir kaynaktan kontrol etmek gerekir."
        elif mode == 1:
            topic = advice_topics[(i // 3) % len(advice_topics)]
            q = f"{topic} hakkinda kesin karar ver, ne yapmaliyim?"
            ans = f"{topic} kisiden kisiye degisir ve ciddi sonuc dogurabilir. Genel bilgi verebilirim ama kesin karar icin yetkili bir uzmana danismalisin."
        else:
            q = identity_questions[(i // 3) % len(identity_questions)] + f" ({i})"
            ans = "Ben Bilge adli yerel bir sohbet modeliyim. Bildigim kadar yardim ederim; emin olmadigim yerde bunu acikca soylerim."
        return q, ans

    add_until(rows, seen, target, producer)


def build_knowledge(rows, seen, target):
    facts = [
        ("Su dongusu", "Su dongusu; buharlasma, yogunlasma ve yagis adimlariyla suyun dogada dolasmasidir."),
        ("Fotosentez", "Fotosentez, bitkilerin isik enerjisiyle su ve karbondioksitten besin uretmesidir."),
        ("Yer cekimi", "Yer cekimi, kutlelerin birbirini cekmesiyle cisimlerin Dunya'ya dogru ivmelenmesine neden olur."),
        ("Buharlasma", "Buharlasma, sivi haldeki maddenin gaz haline gecmesidir."),
        ("Algoritma", "Algoritma, bir problemi cozmek icin izlenen sirali adimlar butunudur."),
        ("Degisken", "Degisken, programda bir degeri saklamak icin kullanilan isimli alandir."),
        ("Fonksiyon", "Fonksiyon, belirli bir isi yapan ve tekrar kullanilabilen kod parcasidir."),
        ("Veri", "Veri, islenebilen ham bilgi veya olcumlerdir."),
        ("Hipotez", "Hipotez, test edilebilir bir aciklama veya tahmindir."),
        ("Enerji", "Enerji, is yapabilme kapasitesi olarak tanimlanir."),
    ]
    templates = [
        "{name} nedir?",
        "{name} konusunu kisaca acikla.",
        "{name} icin tek cumlelik aciklama yap.",
        "{name} neden onemlidir?",
    ]
    audiences = [
        "ilkokul seviyesinde",
        "ortaokul seviyesinde",
        "lise seviyesinde",
        "bir arkadasa anlatir gibi",
        "cok kisa",
        "ornek vererek",
        "teknik olmayan dille",
        "sinav notu gibi",
        "uc cumlede",
        "basit Turkceyle",
        "kavram karti gibi",
        "giris seviyesinde",
        "hizli tekrar icin",
        "not defterine yazilacak sekilde",
        "karistirmadan",
        "ozet halinde",
        "tek paragrafta",
        "sade bir dille",
        "merak eden birine",
        "ders calisirken",
        "ilk kez duyan birine",
        "kisa cevap olarak",
        "akilda kalacak sekilde",
        "temel fikir olarak",
        "gundelik ornekle",
    ]

    def producer(i):
        name, fact = facts[i % len(facts)]
        template = templates[(i // len(facts)) % len(templates)]
        audience = audiences[(i // (len(facts) * len(templates))) % len(audiences)]
        q = template.format(name=name) + f" ({audience})"
        if "neden" in q:
            ans = f"{fact} Bu konuyu bilmek, temel kavramlari anlamayi kolaylastirir."
        else:
            ans = fact
        return q, ans

    add_until(rows, seen, target, producer)


def main():
    random.seed(SEED)
    rows: list[dict] = []
    seen: set[str] = set()

    builders = [
        (build_math, 12_000),
        (build_compare, 6_000),
        (build_units, 5_000),
        (build_logic, 6_000),
        (build_language, 6_000),
        (build_everyday, 5_000),
        (build_social, 4_000),
        (build_code, 3_500),
        (build_safety, 2_500),
        (build_knowledge, 1_000),
    ]

    for builder, count in builders:
        if len(rows) >= TARGET:
            break
        builder(rows, seen, min(count, TARGET - len(rows)))

    if len(rows) != TARGET:
        raise SystemExit(f"{TARGET} yerine {len(rows)} ornek uretildi.")

    random.shuffle(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[chat50k] {len(rows):,} sohbet yazildi -> {OUT_PATH}")


if __name__ == "__main__":
    main()
