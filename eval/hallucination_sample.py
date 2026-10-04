#!/usr/bin/env python3
"""Échantillon pour la vérification humaine des hallucinations (AUDESIA.md, §7) : les mêmes silences, tirés au sort,
pour deux profils, à juger à l'aveugle sur la vidéo.

  python eval/hallucination_sample.py out/corpus/small out/corpus/large    100 silences, page hallucinations.html
  python eval/hallucination_sample.py RUN_A RUN_B -n 20                     deux runs d'une vidéo, 20 silences
  python eval/hallucination_sample.py --score reponses.json                 taux d'hallucinations par profil

La page s'écrit à côté des runs. Elle montre chaque silence en vidéo et ses deux descriptions, dans un ordre tiré au
sort ; pour chacune : fidèle, invente ou se trompe, ou je ne sais pas. Elle ne contient pas le nom des profils : qui a
écrit quoi est dans hallucinations.json, à côté. Les réponses restent dans le navigateur au fil de l'eau ; le bouton
les enregistre dans reponses.json, à mettre à côté de la page : --score les rapproche du tirage.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.judge import texts  # noqa: E402

TEMPLATE = Path(__file__).with_name("hallucinations.html")
DATA = '<script id="donnees" type="application/json">null</script>'
SEED = 2026  # même tirage à chaque fois : les deux profils sont jugés sur les mêmes silences


def draw(a, b, n):
    """Silences communs aux deux runs (au moins une description placée), n tirés au sort ; page écrite à côté."""
    single = (a / "metrics.json").exists()
    pairs = [(a, b, a.name)] if single else [(a / v.name, b / v.name, v.name) for v in sorted(a.iterdir())
                                             if (v / "metrics.json").exists() and (b / v.name / "metrics.json").exists()]
    page_dir = a.parent
    pool = []
    for run_a, run_b, name in pairs:
        for (ida, win, ta), (idb, winb, tb) in zip(texts(run_a), texts(run_b)):
            if ida == idb and abs(win[0] - winb[0]) <= 0.5 and (ta or tb):
                pool.append((name, ida, win, {a.name: ta, b.name: tb}, os.path.relpath(run_a / "clip.mp4", page_dir)))
    rng = random.Random(SEED)
    sample = []
    for name, wid, win, by_profile, clip in sorted(rng.sample(pool, min(n, len(pool)))):
        profiles = list(by_profile)
        rng.shuffle(profiles)  # ordre tiré au sort : la page ne dit pas qui a écrit quoi
        sample.append({"video": name, "id": wid, "debut": round(win[0], 2), "fin": round(win[1], 2), "clip": clip.replace(os.sep, "/"),
                       "horaire": f"{int(win[0] // 60)}:{win[0] % 60:04.1f} – {int(win[1] // 60)}:{win[1] % 60:04.1f}".replace(".", ","),
                       "descriptions": [{"place": k + 1, "profil": p, "texte": by_profile[p]} for k, p in enumerate(profiles)]})
    key = hashlib.sha1(json.dumps(sample).encode()).hexdigest()[:12]  # change si les textes changent
    (page_dir / "hallucinations.json").write_text(json.dumps({"cle": key, "echantillon": sample}, ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
    blind = [{**s, "descriptions": [{k: v for k, v in d.items() if k != "profil"} for d in s["descriptions"]]}
             for s in sample]
    blob = json.dumps({"cle": key, "echantillon": blind}, ensure_ascii=False).replace("</", "<\\/")
    page = page_dir / "hallucinations.html"
    page.write_text(TEMPLATE.read_text(encoding="utf-8").replace(DATA, DATA.replace("null", blob)), encoding="utf-8")
    print(f"{len(sample)} silences tirés sur {len(pool)} : ouvrir {page}")


def score(path):
    """Taux d'hallucinations par profil : descriptions jugées « invente ou se trompe » sur celles jugées."""
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    drawn = json.loads(path.with_name("hallucinations.json").read_text(encoding="utf-8"))
    if saved["cle"] != drawn["cle"]:
        sys.exit("ces réponses portent sur un autre tirage que hallucinations.json")
    who = {(s["video"], s["id"], d["place"]): d["profil"] for s in drawn["echantillon"] for d in s["descriptions"]}
    answers = [{**r, "profil": who[r["video"], r["id"], r["place"]]} for r in saved["reponses"]]
    lines = ["## Hallucinations (vérification humaine, à l'aveugle)", "",
             "| Profil | Fidèles | Inventent ou se trompent | Je ne sais pas | Taux d'hallucinations |", "| --- | --- | --- | --- | --- |"]
    for profile in sorted({r["profil"] for r in answers}):
        v = [r.get("verdict") for r in answers if r["profil"] == profile]
        ok, bad, unknown = v.count("fidele"), v.count("invente"), v.count("inconnu")
        rate = f"{bad / (ok + bad) * 100:.0f} %" if ok + bad else "—"
        lines.append(f"| {profile} | {ok} | {bad} | {unknown} | {rate} |")
    report = "\n".join(lines)
    path.with_name("hallucinations.md").write_text(report + "\n", encoding="utf-8")
    print(report)


def main():
    p = argparse.ArgumentParser(description="Échantillon pour la vérification humaine des hallucinations.")
    p.add_argument("runs", nargs="*", type=Path, help="deux dossiers : profils du corpus, ou runs d'une vidéo")
    p.add_argument("-n", type=int, default=100, help="silences à tirer (100 par défaut)")
    p.add_argument("--score", type=Path, help="reponses.json enregistré par la page")
    a = p.parse_args()
    if a.score:
        return score(a.score)
    if len(a.runs) != 2:
        p.error("indiquer deux dossiers, ou --score")
    draw(*a.runs, a.n)


if __name__ == "__main__":
    main()
