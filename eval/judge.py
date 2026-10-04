#!/usr/bin/env python3
"""Juge VLM : compare à l'aveugle deux runs des mêmes vidéos, fenêtre par fenêtre, sur les mêmes images ; ou, avec un
seul run, note chacune de ses descriptions.

  python eval/judge.py out/corpus/small out/corpus/large      tout le corpus (vidéos présentes des deux côtés)
  python eval/judge.py RUN_A RUN_B                            deux dossiers de sortie d'une même vidéo
  python eval/judge.py out/corpus/small                       notation seule d'un profil (ou d'un run), sans comparaison
  python eval/judge.py A B --judge autre.toml                 un autre juge que configs/judge.toml

Les deux runs partent du même précalcul : mêmes silences, mêmes fenêtres. Pour chaque fenêtre, le juge voit 6 images
du plan, extraites pour lui, et les deux descriptions, dans un ordre tiré au sort ; il note chacune de 1 à 5 sur
l'exactitude, la pertinence, la cohérence des personnages et la concision, puis désigne la meilleure. Le juge doit
être absent des chaînes jugées. Résultats : juge_<A>_vs_<B>.json et .md (juge_<A> en notation seule), à côté des runs.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audesia_p0 import as_image, chat, json_list  # noqa: E402

CRITERIA = ("exactitude", "pertinence", "coherence", "concision")
SCALE = """- exactitude : tout ce qu'elle dit se voit sur les images, rien n'est inventé ni faux ;
- pertinence : elle dit l'essentiel de l'action et des personnages, sans détail superflu ;
- coherence : les personnages sont désignés comme dans la description précédente, sans confusion ;
- concision : phrases courtes, au présent, à la troisième personne, sans « on voit » ni interprétation.
"""
JUDGE = """Tu évalues des audiodescriptions françaises d'un film. Tu reçois des images d'un plan et deux \
descriptions, 1 et 2, écrites par deux systèmes pour le silence de ce plan, avec la description précédente de chacun. \
« aucune » veut dire que le système n'a rien placé. Note chaque description présente de 1 à 5 sur :
""" + SCALE + """Une description absente n'est pas notée (null), mais compte dans le choix de la meilleure : mieux vaut \
rien qu'une invention. Réponds uniquement en JSON : {"notes": [{"description": 1, "exactitude": 4, "pertinence": 3, \
"coherence": 5, "concision": 4}, {"description": 2, ...}], "meilleure": "1", "raison": "..."} ; meilleure vaut \
"1", "2" ou "egalite"."""
SOLO = """Tu évalues une audiodescription française d'un film. Tu reçois des images d'un plan et la description \
écrite pour le silence de ce plan, avec la description précédente. Note-la de 1 à 5 sur :
""" + SCALE + """Réponds uniquement en JSON : {"exactitude": 4, "pertinence": 3, "coherence": 5, "concision": 4, \
"raison": "..."} ; raison dit en une phrase ce qui est faux ou manque."""


def texts(run):
    """Fenêtres d'un run et texte placé de chacune (None si rien n'a été placé)."""
    descs = json.loads((run / "descriptions.json").read_text(encoding="utf-8"))
    placed = {p["id"]: p["text"] for p in json.loads((run / "ad.json").read_text(encoding="utf-8"))}
    return [(d["id"], d["window"], placed.get(d["id"])) for d in descs]


def frames(clip, win, folder, n=6):
    """n images réparties sur la fenêtre, extraites pour le juge (les deux runs n'en ont pas le même nombre)."""
    paths = []
    for k in range(n):
        t = win[0] + (win[1] - win[0]) * (k + 0.5) / n
        p = folder / f"{t:08.2f}.jpg"
        if not p.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1",
                            "-vf", "scale=1024:-2", "-q:v", "3", str(p)], check=True)
        paths.append(p)
    return paths


def json_obj(raw):
    """Premier objet JSON de raw ; {} s'il est illisible."""
    try:
        v = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def judge_video(llm, cfg, a, b, name, folder):
    """Notes et verdict de chaque fenêtre commune aux deux runs, A et B rendus anonymes par un ordre tiré au sort ;
    sans B, notes de chaque description de A. Les fenêtres partent en parallèle (parallel dans la configuration du
    juge) : vLLM les traite en lot."""
    tasks, prev = [], {"A": "aucune", "B": "aucune"}
    folder.mkdir(parents=True, exist_ok=True)
    rows_a = texts(a)
    for (ida, win, ta), (idb, winb, tb) in zip(rows_a, texts(b) if b else [(i, w, None) for i, w, _ in rows_a]):
        if ida != idb or abs(win[0] - winb[0]) > 0.5 or abs(win[1] - winb[1]) > 0.5:
            print(f"  {name} {ida} : fenêtres différentes, sautée")
            continue
        if ta or tb:
            tasks.append((ida, win, ta, tb, dict(prev)))
        prev = {"A": ta or prev["A"], "B": tb or prev["B"]}

    def one(task):
        ida, win, ta, tb, before = task
        images = [{"type": "text", "text": "Images du plan :"},
                  *[as_image(p) for p in frames(a / "clip.mp4", win, folder)]]
        if b is None:
            raw, _ = chat(llm, cfg, SOLO, [*images, {"type": "text", "text": f"Description (précédente : "
                                                                              f"{before['A']}) : {ta}"}], json_mode=True)
            obj = json_obj(raw)
            notes, winner = {"A": {c: obj.get(c) for c in CRITERIA}}, None
            print(f"  {name} {ida} : {notes['A']}", flush=True)
        else:
            swap = random.Random(f"{name}:{ida}").random() < 0.5  # ordre tiré au sort, reproductible
            order = ("B", "A") if swap else ("A", "B")
            shown = {"A": ta or "aucune", "B": tb or "aucune"}
            prompt = "\n".join(f"Description {i + 1} (précédente : {before[s]}) : {shown[s]}"
                               for i, s in enumerate(order))
            raw, _ = chat(llm, cfg, JUDGE, [*images, {"type": "text", "text": prompt}], json_mode=True)
            obj = json_obj(raw)
            notes = {order[n["description"] - 1]: {c: n.get(c) for c in CRITERIA}
                     for n in json_list(raw, "notes") if isinstance(n, dict) and n.get("description") in (1, 2)}
            winner = {"1": order[0], "2": order[1]}.get(str(obj.get("meilleure", "")).strip(), "=")
            print(f"  {name} {ida} : meilleure {winner} | A {notes.get('A')} | B {notes.get('B')}", flush=True)
        return {"video": name, "id": ida, "window": win, "A": ta, "B": tb, "notes": notes, "meilleure": winner,
                "raison": obj.get("raison", "")}

    with ThreadPoolExecutor(max_workers=cfg.get("parallel", 1)) as pool:
        return list(pool.map(one, tasks))


def mean(rows, side, c):
    v = [r["notes"][side][c] for r in rows if isinstance(r["notes"].get(side, {}).get(c), (int, float))]
    return (f"{statistics.mean(v):.2f}".replace(".", ",") + f" ({len(v)})") if v else "—"


def summary(rows, name_a, name_b, model):
    """Moyennes par critère (descriptions présentes seulement) et victoires."""
    lines = [f"## Juge {model} : {name_a} (A) contre {name_b} (B), à l'aveugle", "",
             "| Critère | A | B |", "| --- | --- | --- |"]
    lines += [f"| {c} | {mean(rows, 'A', c)} | {mean(rows, 'B', c)} |" for c in CRITERIA]
    wins = {k: sum(r["meilleure"] == k for r in rows) for k in ("A", "B", "=")}
    lines += ["", f"Meilleure description : A {wins['A']} fois, B {wins['B']} fois, égalité {wins['=']} fois, "
                  f"sur {len(rows)} fenêtres."]
    return "\n".join(lines)


def summary_solo(rows, name, model):
    """Moyennes par vidéo et au total, puis les descriptions les moins exactes, avec la raison donnée par le juge."""
    exact = lambda r: r["notes"]["A"].get("exactitude")  # noqa: E731
    low = [r for r in rows if isinstance(exact(r), (int, float)) and exact(r) <= 2]
    lines = [f"## Juge {model} : {name}, notation seule", "",
             "| Vidéo | " + " | ".join(CRITERIA) + " | Exactitude ≤ 2 |", "| --- " * 6 + "|"]
    for video in dict.fromkeys(r["video"] for r in rows):
        vr = [r for r in rows if r["video"] == video]
        lines.append(f"| {video} | " + " | ".join(mean(vr, "A", c) for c in CRITERIA)
                     + f" | {sum(r in low for r in vr)} |")
    lines.append("| **Total** | " + " | ".join(mean(rows, "A", c) for c in CRITERIA) + f" | {len(low)} |")
    if low:
        lines += ["", "Descriptions les moins exactes :", "", "| Vidéo | Moment | Description | Raison du juge |",
                  "| --- | --- | --- | --- |"]
        for r in sorted(low, key=exact)[:10]:
            t = r["window"][0]
            cells = [r["video"], f"{int(t // 60)}:{int(t % 60):02d}", r["A"], str(r["raison"])]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Compare deux runs à l'aveugle avec un juge VLM, ou note un run seul.")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path, nargs="?", help="second run ; sans lui, notation seule de A")
    p.add_argument("--judge", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "judge.toml")
    a = p.parse_args()
    cfg = tomllib.loads(a.judge.read_text(encoding="utf-8"))["judge"]
    cfg["base_url"] = os.environ.get("AUDESIA_JUDGE_URL", cfg["base_url"])  # sous Docker, le serveur du juge par son nom
    from openai import OpenAI
    llm = OpenAI(base_url=cfg["base_url"], api_key="local")
    single = (a.a / "metrics.json").exists()  # un run d'une vidéo, ou un dossier de profil du corpus
    pairs = [(a.a, a.b, a.a.name)] if single else [
        (a.a / v.name, a.b and a.b / v.name, v.name) for v in sorted(a.a.iterdir())
        if (v / "metrics.json").exists() and (not a.b or (a.b / v.name / "metrics.json").exists())]
    stem = a.a.parent / (f"juge_{a.a.name}_vs_{a.b.name}" if a.b else f"juge_{a.a.name}")
    rows = []
    for run_a, run_b, name in pairs:
        rows += judge_video(llm, cfg, run_a, run_b, name, stem.with_name(stem.name + "_images") / name)
    report = summary(rows, a.a.name, a.b.name, cfg["model"]) if a.b else summary_solo(rows, a.a.name, cfg["model"])
    # with_name et non with_suffix : les noms de vidéos contiennent des points (…1080p_1.35-3.35)
    stem.with_name(stem.name + ".json").write_text(json.dumps({"judge": cfg["model"], "A": str(a.a), "B": str(a.b or ""),
                                                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    stem.with_name(stem.name + ".md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
