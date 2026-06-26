"""
tsetlin/machine.py
==================
SIFIRDAN, kutuphanesiz (sadece numpy) bir COK-SINIFLI Tsetlin Makinesi.
Hicbir ML kutuphanesi (torch/sklearn) yok. Ogrenme GRADYANSIZ: her ozellik icin
[1, 2N] arasi bir tam sayi durum (Tsetlin otomati) tutulur, odul/ceza ile +1/-1
kaydirilir. Karar = mantiksal AND (clause) + oylama. Tip I / Tip II geri bildirim
gercek formulleriyle.

Kararlar OKUNABILIR: her clause, "su literaller mevcutsa bu sinif" diyen bir
mantik kuralidir (get_rules ile cikarilir).
"""
from __future__ import annotations
import numpy as np


class MultiClassTsetlinMachine:
    def __init__(self, n_classes, n_features, n_clauses=80, N=100, T=12, s=3.9, seed=0):
        assert n_clauses % 2 == 0, "clause sayisi cift olmali (yari +, yari -)"
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_lit = 2 * n_features          # literaller: ozellik + degili
        self.n_clauses = n_clauses
        self.N = N
        self.T = T
        self.s = s
        self.rng = np.random.default_rng(seed)
        # durumlar [1, 2N], hepsi N'de baslar (tum literaller HARIC) -> clause'lar bos
        self.ta = np.full((n_classes, n_clauses, self.n_lit), N, dtype=np.int16)
        # kutupluluk: ilk yari + (lehte), ikinci yari - (aleyhte)
        self.signs = np.ones(n_clauses, dtype=np.int8)
        self.signs[n_clauses // 2:] = -1

    # ---- literal vektoru ----
    @staticmethod
    def _lits(x):                            # x: [F] -> [2F]  (x, 1-x)
        return np.concatenate([x, 1 - x])

    # ---- tek sinif icin clause ciktilari ----
    def _clauses(self, c, lits, eval_mode):
        inc = self.ta[c] > self.N            # [J, 2F]  dahil mi
        bad = inc & (lits == 0)              # dahil ama literal 0 -> ihlal
        cl = ~bad.any(axis=1)                # [J]  ihlal yoksa 1 (bos clause -> 1)
        if eval_mode:
            cl &= inc.any(axis=1)            # uretimde bos clause 0 sayilir
        return cl.astype(np.int8)

    def _class_scores_all(self, x, eval_mode=True):
        lits = self._lits(x)
        sc = np.empty(self.n_classes, dtype=np.int32)
        for c in range(self.n_classes):
            cl = self._clauses(c, lits, eval_mode)
            sc[c] = int((cl * self.signs).sum())
        return sc

    def predict(self, X):
        X = np.asarray(X, dtype=np.int8)
        out = np.empty(len(X), dtype=np.int32)
        for i in range(len(X)):
            out[i] = int(self._class_scores_all(X[i], eval_mode=True).argmax())
        return out

    def decision(self, x):
        """Tek ornek icin (tahmin, skorlar) dondurur (kapi/abstain icin)."""
        sc = self._class_scores_all(np.asarray(x, dtype=np.int8), eval_mode=True)
        return int(sc.argmax()), sc

    # ---- geri bildirim (bir sinifin secili clause'larina) ----
    def _type_i(self, c, idx, cl, lits):
        if idx.size == 0:
            return
        St = self.ta[c, idx]                 # [K, 2F]  (kopya degil; gelismis indeksleme -> kopya!)
        C = cl[idx][:, None]                 # [K,1]
        L = lits[None, :]                    # [1,2F]
        r = self.rng.random(St.shape)
        inc = (C == 1) & (L == 1) & (r < (self.s - 1) / self.s)   # dahil et (odul)
        dec = ((C == 0) | (L == 0)) & (r < 1.0 / self.s)          # haric (regularize)
        St = St + inc.astype(np.int16) - dec.astype(np.int16)
        np.clip(St, 1, 2 * self.N, out=St)
        self.ta[c, idx] = St

    def _type_ii(self, c, idx, cl, lits):
        if idx.size == 0:
            return
        St = self.ta[c, idx]
        C = cl[idx][:, None]
        L = lits[None, :]
        inc = (C == 1) & (L == 0) & (St <= self.N)   # yanlis pozitifi bastir
        St = St + inc.astype(np.int16)
        np.clip(St, 1, 2 * self.N, out=St)
        self.ta[c, idx] = St

    def _update_class(self, c, lits, is_target):
        cl = self._clauses(c, lits, eval_mode=False)         # egitimde bos=1
        v = int(np.clip((cl * self.signs).sum(), -self.T, self.T))
        p = (self.T - v) / (2 * self.T) if is_target else (self.T + v) / (2 * self.T)
        fb = self.rng.random(self.n_clauses) <= p            # hangi clause geri bildirim alir
        pos = self.signs > 0
        if is_target:
            ti = np.where(fb & pos)[0]; tii = np.where(fb & ~pos)[0]
        else:
            ti = np.where(fb & ~pos)[0]; tii = np.where(fb & pos)[0]
        self._type_i(c, ti, cl, lits)
        self._type_ii(c, tii, cl, lits)

    def fit(self, X, y, epochs=20, verbose=True, eval_every=5, Xte=None, yte=None):
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.int64)
        n = len(X)
        for ep in range(1, epochs + 1):
            order = self.rng.permutation(n)
            for i in order:
                lits = self._lits(X[i])
                tgt = int(y[i])
                neg = int(self.rng.integers(self.n_classes - 1))
                if neg >= tgt:
                    neg += 1                                  # tgt'den farkli rastgele sinif
                self._update_class(tgt, lits, is_target=True)
                self._update_class(neg, lits, is_target=False)
            if verbose and (ep % eval_every == 0 or ep == 1 or ep == epochs):
                tr = (self.predict(X) == y).mean()
                msg = f"[tsetlin] epoch {ep:3d} | egitim dogrulugu {tr:.3f}"
                if Xte is not None:
                    te = (self.predict(Xte) == np.asarray(yte)).mean()
                    msg += f" | test {te:.3f}"
                print(msg)
        return self

    # ---- kalici kayit ----
    def save(self, path):
        np.savez(path, ta=self.ta, signs=self.signs, s=np.array([self.s]),
                 meta=np.array([self.n_classes, self.n_features, self.n_clauses,
                                self.N, self.T], dtype=np.int64))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        nc, nf, ncl, N, T = (int(x) for x in d["meta"])
        m = cls(nc, nf, n_clauses=ncl, N=N, T=T, s=float(d["s"][0]))
        m.ta = d["ta"].astype(np.int16)
        m.signs = d["signs"]
        return m

    # ---- okunabilir kurallar ----
    def get_rules(self, feature_names, top_clauses=3, max_lits=6):
        """Her sinif icin en 'kararli' pozitif clause'lari okunabilir kurala cevirir."""
        rules = {}
        for c in range(self.n_classes):
            inc = self.ta[c] > self.N                 # [J,2F]
            conf = (self.ta[c] - self.N) * inc        # dahil literallerin gucu
            strength = conf.sum(axis=1)               # [J]
            order = np.argsort(-strength)
            picked = []
            for j in order:
                if self.signs[j] < 0 or strength[j] <= 0:
                    continue
                lits_on = np.where(inc[j])[0]
                terms = []
                for k in lits_on[np.argsort(-conf[j, lits_on])][:max_lits]:
                    if k < self.n_features:
                        terms.append(f"'{feature_names[k]}'")
                    else:
                        terms.append(f"NOT '{feature_names[k - self.n_features]}'")
                if terms:
                    picked.append(" VE ".join(terms))
                if len(picked) >= top_clauses:
                    break
            rules[c] = picked
        return rules


if __name__ == "__main__":
    # mini oz-test: "sinif 1 = ozellik0 VE ozellik1" ogrenebiliyor mu?
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(2000, 4)).astype(np.int8)
    y = (X[:, 0] & X[:, 1]).astype(np.int64)          # AND problemi
    tm = MultiClassTsetlinMachine(n_classes=2, n_features=4, n_clauses=20, T=8, s=3.0)
    tm.fit(X, y, epochs=20, eval_every=5)
    print("kurallar:", tm.get_rules(["f0", "f1", "f2", "f3"]))
