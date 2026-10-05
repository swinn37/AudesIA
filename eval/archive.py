#!/usr/bin/env python3
"""Archive dans results/ les résultats texte d'un profil du corpus : métriques, chevauchement, descriptions, notes du
juge, tirage de l'échantillon d'hallucinations et journaux, puis le rapport (eval/report.py results). Ni vidéo, ni
audio, ni image. Pour les vidéos marquées publish = false dans corpus/corpus.toml (enfants à l'écran), seuls les
chiffres restent : leurs descriptions sont masquées partout.

  python eval/archive.py small
"""
import json
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import report  # noqa: E402
from eval.judge import summary, summary_solo  # noqa: E402

SRC, DST = ROOT / "out" / "corpus", ROOT / "results"
FILES = ("metrics.json", "overlap.json", "descriptions.json", "ad.json", "ad.vtt", "relecture.json", "personnages.json")
HIDDEN = "(description non publiée : évaluation locale seulement)"


def main():
    profile = sys.argv[1]
    private = {v["id"] for v in tomllib.loads((ROOT / "corpus" / "corpus.toml").read_text(encoding="utf-8"))["video"]
               if v.get("publish") is False}
    for run in sorted(p for p in (SRC / profile).iterdir() if (p / "metrics.json").exists()):
        out = DST / profile / run.name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("metrics.json",) if run.name in private else FILES:
            if (run / f).exists():
                shutil.copy2(run / f, out / f)

    for path in SRC.glob(f"juge_{profile}*.json"):  # notes du juge : les chiffres restent, les textes sont masqués
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data["rows"]:
            if r["video"] in private:
                r.update(A=r["A"] and HIDDEN, B=r["B"] and HIDDEN, raison="")
        (DST / path.name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        a, b = Path(data["A"]).name, data["B"] and Path(data["B"]).name
        md = summary(data["rows"], a, b, data["judge"]) if b else summary_solo(data["rows"], a, data["judge"])
        (DST / path.with_suffix(".md").name).write_text(md + "\n", encoding="utf-8")

    for path in SRC.glob(f"hallucinations_{profile}*.json"):  # tirage : gardé pour compter les réponses plus tard
        data = json.loads(path.read_text(encoding="utf-8"))
        for s in data["echantillon"]:
            for d in s["descriptions"] if s["video"] in private else []:
                d["texte"] = d["texte"] and HIDDEN
        (DST / path.name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for path in [*SRC.glob(f"hallucinations_{profile}*.md"), *SRC.glob(f"reponses_{profile}*.json")]:
        shutil.copy2(path, DST / path.name)  # comptage et réponses : sans texte de description

    log = SRC / f"run_{profile}.log"
    if log.exists():  # journal du run : la sortie des vidéos non publiables est retirée
        lines, hide = [], False
        # errors="replace" : cmd.exe y écrit dans la page de code de la console (« 'sox' n'est pas reconnu… »)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("$ ", "en échec", "RUN ")):
                hide = line.startswith("$ ") and any(f"{profile}{sep}{v}" in line for v in private for sep in "\\/")
                lines += [line, HIDDEN] if hide else [line]
            elif not hide:
                lines.append(line)
        (DST / log.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if (SRC / f"juge_{profile}.log").exists():
        shutil.copy2(SRC / f"juge_{profile}.log", DST)  # le juge n'y écrit que ses notes, sans les descriptions
    report.main(DST)


if __name__ == "__main__":
    main()
