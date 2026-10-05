#!/usr/bin/env python3
"""Rapport chiffré du corpus : un tableau par profil (out/corpus/<profil>/), puis ceux du juge et de la vérification
des hallucinations s'ils existent, écrit dans out/corpus/rapport.md.

  python eval/report.py
  python eval/report.py results       le même rapport sur l'archive (eval/archive.py), dans results/rapport.md
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "out" / "corpus"


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ratio(num, den):
    return f"{num} sur {den} ({num / den * 100:.0f} %)" if den else "—"


def num(x):
    return f"{x:.1f}".replace(".", ",")


def mmss(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


RELIABLE_DB = 3.0  # en dessous, le résidu garde presque toute la musique : vérité terrain non probante


def table(folder):
    """Une ligne par vidéo, puis le total. La vérité terrain n'existe que pour les vidéos à piste musique + effets, et ne
    compte au total que si la soustraction de cette piste a vraiment retiré la musique (RELIABLE_DB)."""
    lines = ["| Vidéo | Durée | Placées | Couverture | Sans chevauchement des répliques | Sans chevauchement d'aucune voix "
             "| Musique retirée du résidu | Calcul par minute de vidéo | Jetons envoyés / générés | Pic GPU | Hors ligne |",
             "| --- " * 11 + "|"]
    t = dict(duration=0.0, placed=0, windows=0, covered=0.0, usable=0.0, words=0, strict=0, judged=0, compute=0.0)
    models = set()
    for d in sorted(p for p in folder.iterdir() if (p / "metrics.json").exists()):
        m, o = load(d / "metrics.json"), load(d / "overlap.json")
        compute = sum(m["timings_s"].values())  # sans le précalcul, sauté s'il était déjà fait
        tokens = [sum(u[k] for u in m.get("llm_usage", {}).values()) for k in ("prompt_tokens", "completion_tokens")]
        models.add(" + ".join(dict.fromkeys(m.get("models", {}).values())) or "?")
        offline = {True: "non", False: "oui"}.get(m.get("outbound_network"), "—")
        reliable = bool(o) and o["music_attenuation_db"] >= RELIABLE_DB
        note = " (non probant)" if o and not reliable else ""
        lines.append(
            f"| {d.name} | {mmss(m['duration_s'])} | {m['placed']} sur {m['windows']} | {m['coverage'] * 100:.0f} % "
            f"| {ratio(o['clean_words'], o['descriptions']) if o else '—'}{note}"
            f"{f', {n} hors piste musique + effets' if o and (n := o.get('outside_truth')) else ''} "
            f"| {ratio(o['clean'], o['descriptions']) if o else '—'}{note} "
            f"| {num(o['music_attenuation_db']) + ' dB' if o else '—'} "
            f"| {num(compute / m['duration_s'])} min | {tokens[0] / 1000:.0f} k / {tokens[1] / 1000:.0f} k "
            f"| {num(m['peak_gpu_gib_this_process'])} Gio | {offline} |")
        t["duration"] += m["duration_s"]
        t["placed"] += m["placed"]
        t["windows"] += m["windows"]
        t["covered"] += m["coverage"] * m["usable_silence_s"]
        t["usable"] += m["usable_silence_s"]
        t["compute"] += compute
        if reliable:
            t["words"] += o["clean_words"]
            t["strict"] += o["clean"]
            t["judged"] += o["descriptions"]
    if t["duration"]:
        lines.append(
            f"| **Total** | {mmss(t['duration'])} | {t['placed']} sur {t['windows']} "
            f"| {t['covered'] / t['usable'] * 100 if t['usable'] else 0:.0f} % | {ratio(t['words'], t['judged'])} "
            f"| {ratio(t['strict'], t['judged'])} | | {num(t['compute'] / t['duration'])} min | | | |")
    return lines, models


def memory(folder, corpus):
    """Mémoire de l'hôte au plus haut et disponible au plus bas pendant chaque étape (relevé à 1 Hz de
    scripts/bench_memory.sh, rattaché aux étapes horodatées de metrics.json) ; rien sans relevé."""
    csv = next((p for p in (corpus / f"memoire_{folder.name}.csv", corpus.parent / "memoire.csv") if p.exists()), None)
    if not csv:
        return []
    rows = [(datetime.fromisoformat(t).timestamp(), float(used), float(free)) for t, used, free, *_ in
            (line.split(",") for line in csv.read_text(encoding="utf-8").splitlines()[1:] if line.strip())]
    peaks = {}
    for d in sorted(p for p in folder.iterdir() if (p / "metrics.json").exists()):
        for s in load(d / "metrics.json").get("stages", []):
            a, b = (datetime.fromisoformat(s[k]).timestamp() for k in ("debut", "fin"))
            seen = [(u, f) for t, u, f in rows if a <= t <= b]
            if seen:
                p = peaks.setdefault(s["etape"], [0.0, float("inf")])
                p[:] = max(p[0], max(u for u, _ in seen)), min(p[1], min(f for _, f in seen))
    return ["| Étape | Mémoire utilisée au plus haut | Mémoire disponible au plus bas |", "| --- | --- | --- |",
            *[f"| {k} | {num(u)} Gio | {num(f)} Gio |" for k, (u, f) in peaks.items()]] if peaks else []


def main(corpus=CORPUS):
    out = [f"# Rapport du corpus ({date.today():%d/%m/%Y})", "",
           "Calcul par minute de vidéo : durée des étapes de ce run, sans le précalcul (extrait, parole, plans) "
           "quand il était déjà fait. Chevauchement : contre la vérité terrain tirée de la piste musique + effets, "
           "pour les vidéos qui en ont une ; « aucune voix » compte aussi les cris et les souffles. Musique retirée du "
           f"résidu : sous {num(RELIABLE_DB)} dB, la piste ne retire presque rien du mixage, la mesure n'est pas probante "
           "et ne compte pas au total.", ""]
    for folder in sorted(p for p in corpus.iterdir() if p.is_dir() and p.name != "commun") if corpus.exists() else []:
        lines, models = table(folder)
        if len(lines) > 2:
            out += [f"## Profil {folder.name} : {', '.join(sorted(models))}", "", *lines, ""]
        if mem := memory(folder, corpus):
            out += ["Mémoire de l'hôte, relevée chaque seconde par scripts/bench_memory.sh, étape par étape (toutes "
                    "vidéos confondues ; serveurs de modèles compris) :", "", *mem, ""]
    # Qualité : tableaux écrits par eval/judge.py et eval/hallucination_sample.py --score, s'ils existent.
    for extra in [*sorted(corpus.glob("juge_*.md")), *sorted(corpus.glob("hallucinations*.md"))] if corpus.exists() else []:
        out += [extra.read_text(encoding="utf-8").strip(), ""]
    report = "\n".join(out)
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "rapport.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS)
