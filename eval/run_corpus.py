#!/usr/bin/env python3
"""Traite le corpus (corpus/corpus.toml) avec un profil, puis mesure le chevauchement contre la vérité terrain quand
la vidéo a sa piste musique + effets. Une étape déjà faite est sautée : on peut relancer après un arrêt.

  python eval/run_corpus.py --precompute               extrait, parole et plans de tout le corpus (sur la 5080)
  python eval/run_corpus.py --profile small            référence sur la 5080 (Ollama)
  python eval/run_corpus.py --profile large --docker   sur le GX10 (serveurs lancés par scripts/docker.sh start gx10)
  python eval/run_corpus.py --profile small sintel-vf  seulement ces vidéos
  python eval/run_corpus.py --profile large --docker --jobs 2   deux vidéos à la fois (journaux : <profil>/<vidéo>.log)

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MEDIA = Path("corpus/media")
SHARED = ("clip.mp4", "audio16k.wav", "audio48k.wav", "segments.json", "vocal.json", "shots.json")  # communs aux profils


def run(cmd, log=None):
    """Lance une commande ; avec log (vidéos en parallèle), sa sortie va dans ce journal plutôt qu'à l'écran."""
    print("$", " ".join(cmd), flush=True)
    if not log:
        return subprocess.run(cmd).returncode == 0
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def main():
    p = argparse.ArgumentParser(description="Traite le corpus d'évaluation avec un profil.")
    p.add_argument("ids", nargs="*", help="vidéos du corpus (par défaut : toutes)")
    p.add_argument("--profile", default="small")
    p.add_argument("--precompute", action="store_true", help="extrait, parole et plans seulement")
    p.add_argument("--docker", action="store_true", help="passer par scripts/docker.sh (GX10)")
    p.add_argument("--jobs", type=int, default=1, help="vidéos traitées en même temps (débit, sur le GX10)")
    a = p.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)  # chemins relatifs à la racine, comme sous Docker
    # Sous Windows, un sous-processus dont la sortie est redirigée (journal) écrit en cp1252 et plante sur un caractère
    # absent de cp1252 (Whisper hallucine du chinois) : mode UTF-8 pour tous.
    os.environ["PYTHONUTF8"] = "1"
    # Chemins en barres obliques (as_posix) : sous Windows, bash retirait les « \ » avant Docker (corpusmediatears…).
    # Chemin complet de bash : sous Windows, « bash » seul lance celui de WSL (System32 passe avant le PATH), où le
    # cache de modèles monté n'existe pas.
    bash = shutil.which("bash") or "bash"
    pipeline = [bash, "scripts/docker.sh", "run"] if a.docker else [sys.executable, "audesia_p0.py"]
    python = [bash, "scripts/docker.sh", "py"] if a.docker else [sys.executable]

    def one(v):
        """Une vidéo : précalcul, ou pipeline puis chevauchement ; renvoie son nom si elle échoue ou manque."""
        video, shared = MEDIA / v["file"], Path("out/corpus/commun") / v["id"]
        out = shared if a.precompute else Path("out/corpus") / a.profile / v["id"]
        log = out.parent / f"{v['id']}.log" if a.jobs > 1 else None  # en parallèle, un journal par vidéo
        if not video.exists():
            print(f"{v['id']} : {video} manque (python corpus/download.py)")
            return v["id"]
        out.mkdir(parents=True, exist_ok=True)
        if a.precompute:
            done = all((shared / f).exists() for f in SHARED)
            ok = done or run(pipeline + [video.as_posix(), "--precompute", "--out", shared.as_posix()], log)
            return None if ok else v["id"]
        for f in SHARED:  # l'extrait, la parole et les plans ne sont pas refaits pour chaque profil
            if (shared / f).exists() and not (out / f).exists():
                try:  # lien physique : l'extrait réencodé pèse des centaines de Mo, et le pipeline ne le réécrit pas
                    os.link(shared / f, out / f)
                except OSError:
                    shutil.copy2(shared / f, out / f)
        if not (out / "metrics.json").exists() and not run(
                pipeline + [video.as_posix(), "--profile", a.profile, "--out", out.as_posix()], log):
            return v["id"]
        if v.get("me") and not (out / "overlap.json").exists():
            duration = json.loads((out / "segments.json").read_text(encoding="utf-8"))["duration"]
            if not run(python + ["eval/overlap.py", out.as_posix(), "--video", video.as_posix(),
                                 "--me", (MEDIA / v["me"]).as_posix(), "--end", str(duration)], log):
                return v["id"]
        return None

    videos = [v for v in tomllib.loads(Path("corpus/corpus.toml").read_text(encoding="utf-8"))["video"]
              if not a.ids or v["id"] in a.ids]
    # ponytail: un fil par vidéo ; chaque pipeline charge sa propre voix et sa transcription (~6 à 8 Gio de plus)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        failed = [x for x in pool.map(one, videos) if x]
    if failed:
        sys.exit(f"en échec ou manquantes : {', '.join(failed)}")


if __name__ == "__main__":
    main()
