#!/usr/bin/env python3
"""Page web de démonstration (docs/specs/2026-10-06-page-web.md) : déposer une vidéo, suivre le traitement, relire,
corriger, régénérer une description, écouter et exporter, sans ligne de commande. Locale (127.0.0.1), un traitement à
la fois, dans out/web/<nom>/.

  python serveur.py                     http://127.0.0.1:8000 (profil small, Ollama)
  python serveur.py --profile large     un autre profil de configs/
  python serveur.py --selftest          vérifications sans GPU ni modèle

À lancer avec le Python du venv du pipeline : chaque traitement est lancé avec ce même interpréteur.
"""
import argparse
import atexit
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audesia_p0 import load_profile, secs  # noqa: E402

WEB = ROOT / "out" / "web"  # un dossier par traitement, qui est aussi le dossier de sortie du pipeline
ETAPES = ("extrait", "parole", "voix isolée", "plans", "description", "voix", "mixage")  # noms des step()
VIDEOS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
VOIX = ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee")
TEMPS = re.compile(r"\d+(:\d{1,2}){0,2}(\.\d+)?")  # 95, 1:35, 0:01:35.5
PROFIL = {"nom": "small"}    # choisi au lancement (--profile)
RUN = {"proc": None}         # le traitement en cours : un seul à la fois
LOCK = threading.Lock()      # lancement
FICHIERS = threading.Lock()  # etat.json, lu par les pages pendant que le fil d'attente l'écrit


class Refus(Exception):
    """Refus expliqué à la page : code HTTP et message en français."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def slug(texte):
    """Nom de traitement : minuscules sans accents, [a-z0-9-] ; « video » s'il ne reste rien."""
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:60].strip("-") or "video"


def unique(nom):
    """nom, ou nom-2, nom-3… s'il est déjà pris."""
    n, i = nom, 1
    while (WEB / n).exists():
        i += 1
        n = f"{nom}-{i}"
    return n


def progress(journal):
    """Étape en cours, d'après le journal du dernier lancement : « [étape] durée » marque une étape finie (step()) ;
    pendant la description, « k/N … décrit » compte la lecture des images, puis une ligne par fenêtre rédigée."""
    run = journal.rsplit("\n$ ", 1)[-1]  # launch() fait précéder chaque lancement de « \n$ commande »
    faites = [ETAPES.index(e) for e in re.findall(r"^\[([^\]]+)\] [\d.]+ s$", run, re.M) if e in ETAPES]
    rang = max(faites, default=-1) + 1
    if rang == len(ETAPES):
        return {"rang": rang, "total": rang, "etape": "fini", "detail": ""}
    vues = re.findall(r"^\s*\d+/(\d+)\s.*décrit", run, re.M)
    ecrites = len(re.findall(r"^\s+[\d.]+–\s*[\d.]+ s\s+\[", run, re.M))
    detail = ("" if ETAPES[rang] != "description" or not vues
              else f"rédaction {ecrites} sur {vues[-1]}" if ecrites else f"vision {len(vues)} sur {vues[-1]}")
    return {"rang": rang + 1, "total": len(ETAPES), "etape": ETAPES[rang], "detail": detail}


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}".replace(".", ",")


def script(ad):
    """Script texte des descriptions placées (ad.json) : « début – fin  texte », une ligne chacune."""
    return "".join(f"{mmss(d['start'])} – {mmss(d['end'])}  {d['text']}\n" for d in ad)


def read_etat(nom):
    with FICHIERS:
        return json.loads((WEB / nom / "etat.json").read_text(encoding="utf-8"))


def write_etat(nom, **champs):
    with FICHIERS:
        p = WEB / nom / "etat.json"
        e = {**(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}), **champs}
        p.write_text(json.dumps(e, ensure_ascii=False, indent=1), encoding="utf-8")
        return e


def mark_interrupted():
    """Au démarrage : un traitement resté « en cours » a été interrompu avec le serveur."""
    for p in WEB.glob("*/etat.json"):
        if read_etat(p.parent.name).get("en_cours"):
            write_etat(p.parent.name, en_cours=False, code="interrompu")


def status(nom):
    e, log = read_etat(nom), WEB / nom / "journal.log"
    journal = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    code = e.get("code")
    etat = ("en_cours" if e.get("en_cours") else "interrompu" if code == "interrompu" else "fini" if code == 0
            else "en_attente" if code is None else "echec")
    return {"nom": nom, "etat": etat, **progress(journal), "journal": journal.splitlines()[-20:],
            "relecture": (WEB / nom / "relecture.html").exists(),
            "parametres": {k: e.get(k) for k in ("debut", "fin", "voix", "profil")}}


def command(nom, extra=()):
    """Commande du pipeline pour ce traitement, avec les paramètres d'etat.json (-u : journal ligne à ligne)."""
    e = read_etat(nom)
    cmd = [sys.executable, "-X", "utf8", "-u", str(ROOT / "audesia_p0.py"), str(WEB / nom / e["source"]),
           "--start", e["debut"]]
    if e["fin"]:
        cmd += ["--end", e["fin"]]
    return cmd + ["--voice", e["voix"], "--profile", e["profil"], "--out", str(WEB / nom), *extra]


def ollama_ok(profil):
    """Le serveur Ollama du profil répond-il ? Vrai pour vLLM (profil large) : rien à vérifier ici."""
    base = load_profile(profil)["vlm"]["base_url"]
    if ":11434" not in base:
        return True
    try:
        urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/api/version", timeout=3).read()
        return True
    except OSError:
        return False


def ready():
    """Refus si un traitement tourne déjà, ou si Ollama ne répond pas (l'application s'est déjà fermée seule)."""
    if RUN["proc"] is not None and RUN["proc"].poll() is None:
        raise Refus(409, "Un traitement tourne déjà : attendez qu'il finisse.")
    if not ollama_ok(PROFIL["nom"]):
        raise Refus(503, "Ollama ne répond pas : relancez l'application Ollama, puis réessayez.")


def launch(nom, extra=()):
    """Lance le pipeline en sous-processus : sa sortie s'ajoute à journal.log, son code de retour va dans etat.json."""
    with LOCK:
        ready()
        cmd = command(nom, extra)
        log = open(WEB / nom / "journal.log", "a", encoding="utf-8")
        log.write("\n$ " + " ".join(cmd) + "\n")  # en début de ligne, même après une sortie coupée net
        log.flush()
        RUN["proc"] = proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        write_etat(nom, en_cours=True, code=None, commande=" ".join(cmd))

    def wait():
        code = proc.wait()
        log.close()
        write_etat(nom, en_cours=False, code=code)

    threading.Thread(target=wait, daemon=True).start()


def stop():
    """À l'arrêt du serveur, le traitement en cours s'arrête aussi : il continuerait sinon sans suivi."""
    if RUN["proc"] is not None and RUN["proc"].poll() is None:
        RUN["proc"].terminate()


def selftest():
    global WEB
    base = Path(tempfile.mkdtemp())
    WEB = base / "web"
    WEB.mkdir()
    assert slug("Sintel VF — extrait 1") == "sintel-vf-extrait-1" and slug("../../etc") == "etc" and slug("!!") == "video"
    (WEB / "sintel").mkdir()
    assert unique("sintel") == "sintel-2" and unique("autre") == "autre"
    journal = ("\n$ python audesia_p0.py a.mp4\n[parole] 0.0 s\n[voix isolée] 0.0 s\n[plans] 0.0 s\n"
               "  1/79     0.0–   2.2 s  décrit (3 images, 0 éclaircies)\n"
               "  2/79     2.2–   5.0 s  décrit (4 images, 1 éclaircies)\n")
    assert progress(journal) == {"rang": 5, "total": 7, "etape": "description", "detail": "vision 2 sur 79"}
    journal += "  registre : ['la jeune femme']\n     0.0–   2.2 s  [zoom : un fruit] ['Elle court.']\n"
    assert progress(journal)["detail"] == "rédaction 1 sur 79"
    assert progress(journal + "[description] 12.0 s\n")["etape"] == "voix"
    assert progress(journal + "[description] 1.0 s\n[voix] 2.0 s\n[mixage] 3.0 s\n")["etape"] == "fini"
    assert progress(journal + "[voix] 2.0 s\n\n$ relance\n[parole] 0.0 s\n")["etape"] == "voix isolée"  # dernier lancement
    assert script([{"start": 75.5, "end": 79.3, "text": "Elle court."}]) == "1:15,5 – 1:19,3  Elle court.\n"
    (WEB / "sintel" / "etat.json").write_text('{"en_cours": true}', encoding="utf-8")
    mark_interrupted()  # serveur arrêté pendant un traitement
    assert status("sintel")["etat"] == "interrompu"
    write_etat("sintel", source="sintel.mp4", debut="1:35", fin="3:35", voix="Vivian", profil="small")
    assert command("sintel", ["--regenerer", "d_0001"])[-6:] == ["--profile", "small", "--out", str(WEB / "sintel"),
                                                              "--regenerer", "d_0001"]
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description="Page web de démonstration d'Audesia (locale).")
    p.add_argument("--profile", default="small", help="profil de configs/ (small : Ollama sur la RTX 5080)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--selftest", action="store_true", help="vérifications sans GPU ni modèle")
    a = p.parse_args()
    if a.selftest:
        return selftest()


if __name__ == "__main__":
    main()
