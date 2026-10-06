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


def tracks(path):
    """Types des pistes du fichier (video, audio…) ; vide si ce n'est pas un média lisible."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return set(r.stdout.split())


def window_ids(nom):
    """Fenêtres de la relecture (relecture.json, écrite à la fin du premier traitement)."""
    p = WEB / nom / "relecture.json"
    if not p.exists():
        raise Refus(409, "Pas encore de relecture : attendez la fin du traitement.")
    return {r["id"] for r in json.loads(p.read_text(encoding="utf-8"))}


def valid_corrections(nom, data):
    """{fenêtre: texte} pour des fenêtres de la relecture, textes nettoyés de leurs espaces ; sinon refus 400."""
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise Refus(400, "Corrections illisibles : un objet {fenêtre: texte} est attendu.")
    inconnues = sorted(set(data) - window_ids(nom))
    if inconnues:
        raise Refus(400, f"Fenêtre inconnue : {', '.join(inconnues)}.")
    return {k: v.strip() for k, v in sorted(data.items())}


def make_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Audesia")

    @app.exception_handler(Refus)
    async def refus(_, e):
        return JSONResponse({"detail": str(e)}, status_code=e.code)

    @app.middleware("http")
    async def sans_cache(request, call_next):  # voix et vidéos réécrites sous le même nom à chaque relance
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    def job(nom):
        if not re.fullmatch(r"[a-z0-9-]+", nom) or not (WEB / nom / "etat.json").exists():
            raise Refus(404, f"Traitement inconnu : {nom}.")
        return nom

    @app.get("/")
    def accueil():
        return FileResponse(ROOT / "accueil.html")

    @app.get("/api/pret")
    def pret():
        ready()  # demandé par la page avant l'envoi : refusée pendant l'envoi, une grosse vidéo couperait la connexion
        return {"pret": True}

    @app.get("/api/traitements")
    def liste():
        return [status(p.parent.name) for p in sorted(WEB.glob("*/etat.json"))]

    @app.post("/api/traitements")
    async def deposer(request: Request, fichier: str, nom: str = "", debut: str = "0", fin: str = "",
                      voix: str = "Vivian"):
        ext = Path(fichier).suffix.lower()
        debut, fin = debut.strip().replace(",", ".") or "0", fin.strip().replace(",", ".")
        if ext not in VIDEOS:
            raise Refus(400, f"Format non pris en charge : {ext or 'sans extension'} (MP4, MKV, MOV, WebM…).")
        if not TEMPS.fullmatch(debut) or (fin and not TEMPS.fullmatch(fin)):
            raise Refus(400, "Début ou fin illisible : en secondes (95) ou en minutes:secondes (1:35).")
        if fin and secs(fin) <= secs(debut):
            raise Refus(400, "La fin doit venir après le début.")
        if voix not in VOIX:
            raise Refus(400, f"Voix inconnue : {voix}.")
        ready()
        n = unique(slug(nom or Path(fichier).stem))
        (WEB / n).mkdir(parents=True)
        src = WEB / n / f"{n}{ext}"  # la page de relecture prend son titre du nom de la vidéo
        try:
            with open(src, "wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
            if not {"video", "audio"} <= tracks(src):
                raise Refus(400, "Ce fichier n'est pas une vidéo avec du son : il faut une piste vidéo et une piste audio.")
        except BaseException:
            shutil.rmtree(WEB / n, ignore_errors=True)  # envoi refusé ou interrompu : rien de laissé à moitié
            raise
        write_etat(n, source=src.name, debut=debut, fin=fin, voix=voix, profil=PROFIL["nom"], en_cours=False, code=None)
        launch(n)  # refus possible si un traitement a démarré entre-temps : le dossier reste « en attente », à relancer
        return status(n)

    @app.get("/api/traitements/{nom}")
    def etat(nom: str):
        return status(job(nom))

    @app.post("/api/traitements/{nom}/relancer")
    def relancer(nom: str):
        launch(job(nom))
        return status(nom)

    @app.post("/api/traitements/{nom}/corrections")
    async def corriger(nom: str, request: Request):
        job(nom)
        try:
            data = await request.json()
        except ValueError:
            raise Refus(400, "Corrections illisibles : JSON attendu.")
        fixes = valid_corrections(nom, data)
        ready()  # refus avant d'écrire : la page garde ses modifications non enregistrées
        (WEB / nom / "corrections.json").write_text(json.dumps(fixes, ensure_ascii=False, indent=1) + "\n",
                                                    encoding="utf-8")
        launch(nom)
        return status(nom)

    @app.post("/api/traitements/{nom}/regenerer/{fenetre}")
    def regenerer(nom: str, fenetre: str):
        if fenetre not in window_ids(job(nom)):
            raise Refus(400, f"Fenêtre inconnue : {fenetre}.")
        launch(nom, ["--regenerer", fenetre])
        return status(nom)

    @app.get("/api/traitements/{nom}/export/{fmt}")
    def exporter(nom: str, fmt: str):
        d = WEB / job(nom)
        sources = {"mkv": "clip_ad.mkv", "mp4": "clip_ad.mp4", "vtt": "ad.vtt", "mp3": "ad_mix.wav", "txt": "ad.json"}
        if fmt not in sources:
            raise Refus(404, f"Format inconnu : {fmt}.")
        if not (d / sources[fmt]).exists():
            raise Refus(404, "Pas encore d'export : le traitement n'est pas fini.")
        if fmt == "txt":
            return PlainTextResponse(script(json.loads((d / "ad.json").read_text(encoding="utf-8"))),
                                     headers={"Content-Disposition": f'attachment; filename="{nom}.txt"'})
        p = d / sources[fmt]
        if fmt == "mp3":  # bande-son atténuée et voix : la piste d'audiodescription que YouTube attend
            p = d / "ad.mp3"
            if not p.exists() or p.stat().st_mtime < (d / "ad_mix.wav").stat().st_mtime:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "ad_mix.wav"), "-b:a", "192k",
                                str(p)], check=True)
        return FileResponse(p, filename=f"{nom}{p.suffix}")

    app.mount("/t", StaticFiles(directory=WEB, check_dir=False), name="t")
    return app


def selftest():
    global WEB, command, ollama_ok
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
    from fastapi.testclient import TestClient
    command = lambda nom, extra=(): [sys.executable, "-c", "import time; print('[parole] 0.0 s'); time.sleep(1)"]  # noqa: E731
    ollama_ok = lambda profil: True  # noqa: E731
    c = TestClient(make_app())

    def attendre(nom):  # fin du traitement lancé (commande factice : 1 s)
        for _ in range(100):
            etat = c.get(f"/api/traitements/{nom}").json()["etat"]
            if etat != "en_cours":
                return etat
            time.sleep(0.1)

    video = base / "essai.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=1", "-f", "lavfi",
                    "-i", "anullsrc=d=1", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video)], check=True)
    assert c.get("/api/pret").status_code == 200
    r = c.post("/api/traitements?fichier=Essai%20VF.mp4&debut=0&voix=Vivian", content=video.read_bytes())
    assert r.status_code == 200 and r.json()["nom"] == "essai-vf", r.text
    assert c.get("/api/pret").status_code == 409  # la page ne lance pas l'envoi
    assert c.post("/api/traitements?fichier=b.mp4", content=video.read_bytes()).status_code == 409  # un à la fois
    assert attendre("essai-vf") == "fini" and (WEB / "essai-vf" / "essai-vf.mp4").exists()
    r = c.post("/api/traitements?fichier=faux.mp4", content=b"pas une video")
    assert r.status_code == 400 and not (WEB / "faux").exists()  # rien de laissé à moitié
    assert c.post("/api/traitements?fichier=a.txt", content=b"x").status_code == 400
    assert c.post("/api/traitements?fichier=a.mp4&debut=demain", content=b"x").status_code == 400
    assert c.post("/api/traitements?fichier=a.mp4&debut=1:35&fin=1:00", content=b"x").status_code == 400
    r = c.get("/t/essai-vf/essai-vf.mp4", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and len(r.content) == 10  # déplacement dans la vidéo
    assert r.headers["cache-control"] == "no-cache"  # voix et vidéos refaites sous le même nom à chaque relance
    (base / "secret.txt").write_text("x")
    assert c.get("/t/%2e%2e/secret.txt").status_code == 404 and c.get("/api/traitements/%2e%2e").status_code == 404
    assert c.post("/api/traitements/essai-vf/regenerer/d_0000").status_code == 409  # pas encore de relecture
    (WEB / "essai-vf" / "relecture.json").write_text('[{"id": "d_0000"}]', encoding="utf-8")
    assert c.post("/api/traitements/essai-vf/corrections", json={"d_0042": "x"}).status_code == 400
    assert c.post("/api/traitements/essai-vf/corrections", json={"d_0000": 3}).status_code == 400
    assert c.post("/api/traitements/essai-vf/regenerer/d_0042").status_code == 400
    assert c.post("/api/traitements/essai-vf/corrections", json={"d_0000": " Elle court. "}).status_code == 200
    fixes = WEB / "essai-vf" / "corrections.json"
    assert json.loads(fixes.read_text(encoding="utf-8")) == {"d_0000": "Elle court."}
    assert c.post("/api/traitements/essai-vf/corrections", json={"d_0000": "y"}).status_code == 409  # en cours
    assert json.loads(fixes.read_text(encoding="utf-8")) == {"d_0000": "Elle court."}  # rien d'écrit
    assert attendre("essai-vf") == "fini"
    (WEB / "essai-vf" / "ad.json").write_text('[{"start": 1, "end": 2.5, "text": "Elle court."}]', encoding="utf-8")
    assert c.get("/api/traitements/essai-vf/export/txt").text == "0:01,0 – 0:02,5  Elle court.\n"
    assert c.get("/api/traitements/essai-vf/export/mkv").status_code == 404  # pas encore d'export
    assert c.get("/api/traitements/essai-vf/export/zip").status_code == 404
    launch("essai-vf")
    stop()  # arrêt du serveur : le traitement en cours s'arrête aussi
    assert RUN["proc"].wait(5) != 0
    ollama_ok = lambda profil: False  # noqa: E731
    assert c.post("/api/traitements/essai-vf/relancer").status_code == 503
    assert c.get("/api/pret").status_code == 503
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description="Page web de démonstration d'Audesia (locale).")
    p.add_argument("--profile", default="small", help="profil de configs/ (small : Ollama sur la RTX 5080)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--selftest", action="store_true", help="vérifications sans GPU ni modèle")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    load_profile(a.profile)  # un profil invalide arrête tout de suite
    PROFIL["nom"] = a.profile
    WEB.mkdir(parents=True, exist_ok=True)
    mark_interrupted()
    atexit.register(stop)
    import uvicorn
    print(f"Audesia : http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(make_app(), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
