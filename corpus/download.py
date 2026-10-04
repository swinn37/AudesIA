#!/usr/bin/env python3
"""Télécharge le corpus d'évaluation (corpus.toml) dans corpus/media/, sans retélécharger ce qui est déjà là.

  python corpus/download.py                 tout le corpus
  python corpus/download.py sintel-vf ...   seulement ces vidéos
"""
import sys
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MEDIA = HERE / "media"


def corpus():
    return tomllib.loads((HERE / "corpus.toml").read_text(encoding="utf-8"))["video"]


def fetch(url, dest):
    """Écrit dans un .part renommé à la fin : un téléchargement interrompu ne laisse pas de fichier tronqué."""
    if dest.exists():
        print(f"déjà là : {dest.name}")
        return
    part = dest.with_name(dest.name + ".part")
    print(f"téléchargement : {dest.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "audesia-corpus"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.replace(dest)


def main():
    MEDIA.mkdir(exist_ok=True)
    wanted = set(sys.argv[1:])
    for v in corpus():
        if wanted and v["id"] not in wanted:
            continue
        for name, url in (("file", "url"), ("me", "me_url"), ("subs", "subs_url")):
            if v.get(url):
                fetch(v[url], MEDIA / v[name])


if __name__ == "__main__":
    main()
