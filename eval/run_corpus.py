#!/usr/bin/env python3
"""Traite le corpus (corpus/corpus.toml) avec un profil, puis mesure le chevauchement contre la vérité terrain quand
la vidéo a sa piste musique + effets. Une étape déjà faite est sautée : on peut relancer après un arrêt.

  python eval/run_corpus.py --precompute               extrait, parole et plans de tout le corpus (sur la 5080)
  python eval/run_corpus.py --profile small            référence sur la 5080 (Ollama)
  python eval/run_corpus.py --profile large --docker   sur le GX10 (serveurs lancés par scripts/docker.sh start gx10)
  python eval/run_corpus.py --profile small sintel-vf  seulement ces vidéos

Le précalcul va dans out/corpus/commun/<vidéo>/ et sert à tous les profils ; chaque profil écrit dans
out/corpus/<profil>/<vidéo>/. Ensuite : python eval/report.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

MEDIA = Path("corpus/media")
SHARED = ("clip.mp4", "audio16k.wav", "audio48k.wav", "segments.json", "vocal.json", "shots.json")  # communs aux profils


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def main():
    p = argparse.ArgumentParser(description="Traite le corpus d'évaluation avec un profil.")
    p.add_argument("ids", nargs="*", help="vidéos du corpus (par défaut : toutes)")
    p.add_argument("--profile", default="small")
    p.add_argument("--precompute", action="store_true", help="extrait, parole et plans seulement")
    p.add_argument("--docker", action="store_true", help="passer par scripts/docker.sh (GX10)")
    a = p.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)  # chemins relatifs à la racine, comme sous Docker
    pipeline = ["bash", "scripts/docker.sh", "run"] if a.docker else [sys.executable, "audesia_p0.py"]
    python = ["bash", "scripts/docker.sh", "py"] if a.docker else [sys.executable]
    failed = []
    for v in tomllib.loads(Path("corpus/corpus.toml").read_text(encoding="utf-8"))["video"]:
        if a.ids and v["id"] not in a.ids:
            continue
        video, shared = MEDIA / v["file"], Path("out/corpus/commun") / v["id"]
        if not video.exists():
            print(f"{v['id']} : {video} manque (python corpus/download.py)")
            failed.append(v["id"])
            continue
        if a.precompute:
            done = all((shared / f).exists() for f in SHARED)
            if not done and not run(pipeline + [str(video), "--precompute", "--out", str(shared)]):
                failed.append(v["id"])
            continue
        out = Path("out/corpus") / a.profile / v["id"]
        out.mkdir(parents=True, exist_ok=True)
        for f in SHARED:  # l'extrait, la parole et les plans ne sont pas refaits pour chaque profil
            if (shared / f).exists() and not (out / f).exists():
                try:  # lien physique : l'extrait réencodé pèse des centaines de Mo, et le pipeline ne le réécrit pas
                    os.link(shared / f, out / f)
                except OSError:
                    shutil.copy2(shared / f, out / f)
        if not (out / "metrics.json").exists() and not run(pipeline + [str(video), "--profile", a.profile, "--out", str(out)]):
            failed.append(v["id"])
            continue
        if v.get("me") and not (out / "overlap.json").exists():
            duration = json.loads((out / "segments.json").read_text(encoding="utf-8"))["duration"]
            if not run(python + ["eval/overlap.py", str(out), "--video", str(video), "--me", str(MEDIA / v["me"]),
                                 "--end", str(duration)]):
                failed.append(v["id"])
    if failed:
        sys.exit(f"en échec ou manquantes : {', '.join(failed)}")


if __name__ == "__main__":
    main()
