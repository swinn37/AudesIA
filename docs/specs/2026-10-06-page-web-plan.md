# Page web de démonstration — plan d'implémentation

> Exécution : tâche par tâche, dans l'ordre ; les cases (`- [ ]`) suivent l'avancement, et chaque tâche se termine
> par un commit.

**But :** filmer la démonstration d'Audesia depuis le navigateur : dépôt d'une vidéo, avancement, relecture,
correction, régénération d'une description, écoute et export, sans ligne de commande.

**Architecture :** `serveur.py` (FastAPI, local) lance le pipeline existant, `audesia_p0.py`, en sous-processus, un
traitement à la fois, dans `out/web/<nom>/`, et lit son journal pour l'avancement. `accueil.html` dépose et suit les
traitements ; `relecture.html`, servie par le serveur, lui envoie les corrections et les régénérations. Le pipeline
gagne l'option `--regenerer`.

**Outils :** Python 3.12 du venv `C:\Users\swinn\.venvs\audesia` (FastAPI 0.142, uvicorn 0.54, starlette 1.7 et httpx
0.28, déjà installés), ffmpeg et ffprobe, HTML, CSS et JavaScript natifs.

**Spécification :** [2026-10-06-page-web.md](2026-10-06-page-web.md)

## Contraintes globales

- Toutes les commandes `python` s'entendent avec le Python du venv, activé : le serveur lance le pipeline avec ce même
  interpréteur (`sys.executable`). Le `python` du PATH est un 3.13 hors venv.
- Écoute sur 127.0.0.1 seulement ; port 8000 par défaut (`--port`).
- Un seul traitement à la fois : les autres demandes sont refusées (409).
- Noms de traitement réduits à `[a-z0-9-]` ; dossiers dans `out/web/`, déjà ignoré par git comme tout `out/`.
- Pages en HTML natif, sans ressource externe, en français, utilisables au clavier et au lecteur d'écran.
- `relecture.html` ouverte depuis le disque se comporte exactement comme aujourd'hui.
- Aucune dépendance nouvelle. Tests par `--selftest` (assertions) ; les adresses du serveur avec
  `fastapi.testclient`.
- Commits en français, sans aucune mention de l'outil d'assistance, ni dans les fichiers ni dans les messages :
  `git grep -i` sur son nom, sans résultat, avant chaque commit. Ne jamais toucher à `video/` ni à la visibilité du
  dépôt.

## Points d'attention

1. Voix ou vidéo d'avant la relance gardée par le navigateur (mêmes noms de fichiers) : toute réponse porte
   `Cache-Control: no-cache` (test de l'en-tête, tâche 3).
2. Dépôt refusé pendant l'envoi d'une grosse vidéo : la connexion serait coupée sans message. La page interroge
   `/api/pret` avant d'envoyer, et un envoi refusé ou interrompu ne laisse aucun dossier (tests `/api/pret` 409 et 503,
   fichier sans vidéo → 400 et dossier supprimé, tâche 3).
3. Corrections envoyées pendant un traitement : refus 409 avant d'écrire `corrections.json`, et la page garde ses
   modifications (test 409 et fichier inchangé, tâche 3 ; `enregistre` ne change qu'en cas de succès, tâche 5).
4. Régénération d'une fenêtre corrigée à la main : sa correction est retirée, sinon le texte relu masquerait la
   nouvelle description (test de `drop_corrections`, tâche 1).
5. Serveur arrêté pendant un traitement : le sous-processus s'arrête avec lui, et le traitement apparaît « interrompu »,
   avec Relancer, au redémarrage (tests de `stop` et de `mark_interrupted`, tâches 2 et 3).

---

### Tâche 1 : option `--regenerer` du pipeline

**Fichiers :**
- Modifier : `audesia_p0.py` (`chat`, `describe`, `selftest`, `main` ; deux fonctions nouvelles avant `duck_envelope`)

**Interfaces :**
- Produit : option `--regenerer ID[,ID]` ; `regen_ids(arg: str, known: list[str]) -> set[str]` (arrêt clair si une
  fenêtre est inconnue) ; `drop_corrections(fixes: dict, ids: set) -> dict` ; `chat(..., temperature=0)` ;
  `describe(clip, segs, shots, profile, cps, out, redo=frozenset())`.

- [ ] **Étape 1 : tests qui échouent.** Dans `selftest()`, juste avant `import tempfile` :

```python
    assert regen_ids("d_0001, d_0003", ["d_0001", "d_0002", "d_0003"]) == {"d_0001", "d_0003"}
    try:
        regen_ids("d_0009", ["d_0001"])
        raise AssertionError("fenêtre inconnue acceptée")
    except SystemExit as e:
        assert "d_0009" in str(e)
    assert drop_corrections({"d_0001": "Elle court.", "d_0002": ""}, {"d_0001"}) == {"d_0002": ""}  # le texte relu cède
```

- [ ] **Étape 2 : vérifier l'échec.** `python -X utf8 audesia_p0.py --selftest` → `NameError: name 'regen_ids' is not
  defined`.

- [ ] **Étape 3 : les deux fonctions**, juste avant `def duck_envelope` :

```python
def regen_ids(arg, known):
    """Fenêtres de --regenerer (« d_0003,d_0007 »), toutes connues de descriptions.json ; sinon arrêt clair."""
    ids = {x.strip() for x in arg.split(",") if x.strip()}
    unknown = sorted(ids - set(known))
    if unknown or not ids:
        sys.exit(f"--regenerer : fenêtre inconnue {', '.join(unknown) or arg!r} (voir descriptions.json)")
    return ids


def drop_corrections(fixes, ids):
    """Corrections sans celles des fenêtres régénérées : le texte relu masquerait sinon la nouvelle description."""
    return {k: v for k, v in fixes.items() if k not in ids}
```

- [ ] **Étape 4 : vérifier le succès.** `python -X utf8 audesia_p0.py --selftest` → `selftest OK`.

- [ ] **Étape 5 : température réglable.** Dans `chat`, nouveau paramètre, docstring et appel :

```python
def chat(llm, cfg, system, content, json_mode=False, temperature=0):
    """Une requête au serveur d'un rôle, avec l'extra_body du profil ; à température 0, sauf pour régénérer une
    description. Rend le texte et l'usage."""
```

et, dans l'appel, `temperature=0` devient `temperature=temperature`. Dans `describe`, la température passe par `ask`,
`look` et `zoom` (le reste de leur corps ne change pas) :

```python
    def ask(role, system, content, json_mode=False, temperature=0):
        text, usage = chat(clients[role], profile[role], system, content, json_mode, temperature)
```

```python
    def look(k, w, temperature=0):
```

```python
        raw = ask("vlm", DESCRIBE, [{"type": "text", "text": "Décris cette suite d'images."}, *images, *extra],
                  temperature=temperature)
```

```python
    def zoom(x, temperature=0):
```

```python
        return ask("vlm", ZOOM, [{"type": "text", "text": "Plan et détails agrandis :"}, *zoomed],
                   temperature=temperature)
```

- [ ] **Étape 6 : régénération dans `describe`.** Signature et docstring :

```python
def describe(clip, segs, shots, profile, cps, out, redo=frozenset()):
    """1. Par fenêtre, en parallèle : description de la suite d'images seule (vlm), puis détails agrandis si un objet
    reste vague. 2. Sur l'ensemble : registre des personnages (vlm). 3. Par fenêtre, dans l'ordre : révision sur les
    mêmes images avec le registre et les plans voisins, puis 3 variantes calées (writer). Avec redo (--regenerer),
    seules ces fenêtres sont relues sur les images, puis révisées et rédigées ; les autres sont reprises telles quelles
    de descriptions.json."""
```

Remplacer le bloc du cache de vision, de `cache, key = …` jusqu'à la fin du `else` (son `cache.write_text(…)`
compris), par le bloc suivant ; les trois lignes qui suivent (`registry = …`, `personnages.json`, `registre : …`) ne
changent pas :

```python
    cache, key = out / "vision.json", {"vlm": profile["vlm"]["model"], "max_images": n_max}  # autre vision : à refaire
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None

    def save_vision():
        cache.write_text(json.dumps({"key": key, "items": [{k: v for k, v in x.items() if k not in ("images", "extra")}
                                                           for x in items], "cast": cast}, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    if saved and saved.get("key") == key and [x["window"] for x in saved["items"]] == wins:
        items, cast = saved["items"], saved["cast"]
        for k, x in enumerate(items):
            if x["id"] in redo:  # nouvelle lecture des images : à température 0, ce serait la même, mot pour mot
                items[k] = look(k, x["window"], temperature=0.7)
                items[k]["zoom"] = zoom(items[k], temperature=0.7)
            elif not redo:
                x["images"], x["extra"] = pictures(x["frames"])
        if redo:
            save_vision()
        print(f"  vision reprise de {cache.name} : {len(items)} fenêtres", flush=True)
    elif redo:
        sys.exit("--regenerer : vision.json manque ou ne correspond plus à ces fenêtres (autre modèle de vision, autre "
                 "découpage) : relancer le traitement complet")
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            items = list(pool.map(look, range(len(wins)), wins))
            for x, z in zip(items, list(pool.map(zoom, items))):
                x["zoom"] = z
        # Une image par plan, pour que la même personne garde la même désignation tout au long de l'extrait.
        picks = items if len(items) <= MAX_CAST else [items[int(i * len(items) / MAX_CAST)] for i in range(MAX_CAST)]
        cast = [c for c in json_list(ask("vlm", CAST, [{"type": "text", "text": "Images, dans l'ordre :"},
                                                      *[x["images"][len(x["images"]) // 2] for x in picks]],
                                         json_mode=True), "personnages") if isinstance(c, dict)]
        save_vision()
```

Puis le début de la boucle de révision et de rédaction devient (seules la ligne `kept = …` et le bloc `if redo …` sont
nouveaux) :

```python
    kept = {d["id"]: d for d in json.loads((out / "descriptions.json").read_text(encoding="utf-8"))} if redo else {}
    said, previous = [], "aucun"
    for k, x in enumerate(items):
        if redo and x["id"] not in redo:  # régénération : les autres fenêtres restent telles quelles
            items[k] = kept[x["id"]]
            previous, said = items[k]["description"], said + items[k]["variants"][:1]
            continue
        w = x["window"]
        visuals = x.pop("images") + x.pop("extra")  # pas de base64 dans le cache JSON
```

La fenêtre régénérée est révisée avec le registre de `vision.json`, la description révisée précédente, la description
brute suivante et les variantes déjà dites, comme dans un traitement complet.

- [ ] **Étape 7 : l'option dans `main`.** Après l'argument `--voice` :

```python
    p.add_argument("--regenerer", metavar="ID[,ID]",
                   help="refait la description de ces fenêtres (d_0007), lecture des images comprise, puis la voix")
```

Juste après `out.mkdir(parents=True, exist_ok=True)` :

```python
    redo = set()
    if a.regenerer:  # avant les étapes longues : une fenêtre inconnue arrête tout de suite
        old = out / "descriptions.json"
        if not old.exists():
            sys.exit("--regenerer : pas encore de descriptions dans ce dossier ; lancer d'abord le traitement complet")
        redo = regen_ids(a.regenerer, [d["id"] for d in json.loads(old.read_text(encoding="utf-8"))])
        fixes = out / "corrections.json"
        if fixes.exists():  # sinon le texte relu masquerait la nouvelle description
            fixes.write_text(json.dumps(drop_corrections(json.loads(fixes.read_text(encoding="utf-8-sig")), redo),
                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
```

Remplacer l'appel de l'étape « description » par :

```python
    descs = step("description", lambda: describe(clip, segs, shots, profile, a.cps, out, redo),
                 None if redo else out / "descriptions.json")
    if redo:  # sans cache donné, step() refait la description mais ne l'écrit pas
        (out / "descriptions.json").write_text(json.dumps(descs, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Étape 8 : vérifier.** `python -X utf8 audesia_p0.py --selftest` → `selftest OK`. Puis, Ollama lancé, sur une
  copie du dossier de l'A/B de *Sprite Fright* (35 fenêtres, vision en cache pour le profil small), avec une correction
  sur la fenêtre régénérée :

```bash
cp -r out/ab/sprite-fright-vf out/essai-regen
printf '{"d_0008": "Une cassette entre dans un lecteur."}\n' > out/essai-regen/corrections.json
python -X utf8 audesia_p0.py corpus/media/sprite-fright.fr.touhoppai.858p.mp4 --out out/essai-regen --regenerer d_0008
cat out/essai-regen/corrections.json
python -X utf8 -c "import json; a, b = (json.load(open(f'{d}/descriptions.json', encoding='utf-8')) for d in ('out/ab/sprite-fright-vf', 'out/essai-regen')); print([x['id'] for x, y in zip(a, b) if x != y])"
python -X utf8 audesia_p0.py corpus/media/sprite-fright.fr.touhoppai.858p.mp4 --out out/essai-regen --regenerer d_0099
rm -r out/essai-regen
```

Attendu :
- une seule ligne « 9/35 … décrit » et une seule ligne de rédaction ; la voix ne synthétise que la phrase nouvelle ;
- `corrections.json` vaut `{}` ;
- la comparaison affiche `['d_0008']` (si la nouvelle lecture redonne le même texte, relancer : la température 0,7 le
  fait varier) ;
- `--regenerer d_0099` s'arrête aussitôt sur « fenêtre inconnue d_0099 ».

- [ ] **Étape 9 : commit.**

```bash
git add audesia_p0.py
git commit -m "Ajoute la régénération d'une fenêtre (--regenerer)"
```

---

### Tâche 2 : cœur du serveur (noms, état, avancement, lancement)

**Fichiers :**
- Créer : `serveur.py`

**Interfaces :**
- Consomme : `audesia_p0.load_profile(name) -> dict` (sections `vlm` et `writer` avec `base_url` ; `sys.exit` si le
  profil est invalide) ; `audesia_p0.secs(t) -> float` (« 1:35 » → 95.0). Importer `audesia_p0` ne charge que la
  bibliothèque standard.
- Produit : `WEB: Path` ; `ETAPES: tuple` ; `Refus(code, message)` ; `slug(texte) -> str` ; `unique(nom) -> str` ;
  `progress(journal: str) -> dict` (`rang`, `total`, `etape`, `detail`) ; `script(ad: list) -> str` ;
  `read_etat(nom) -> dict` ; `write_etat(nom, **champs) -> dict` ; `mark_interrupted()` ; `status(nom) -> dict`
  (`nom`, `etat` parmi `en_cours`, `fini`, `echec`, `interrompu`, `en_attente`, puis les clés de `progress`,
  `journal` : 20 dernières lignes, `relecture` : bool, `parametres`) ; `command(nom, extra=()) -> list` ;
  `ollama_ok(profil) -> bool` ; `ready()` (lève `Refus` 409 ou 503) ; `launch(nom, extra=())` ; `stop()`.

- [ ] **Étape 1 : le fichier, avec ses tests et sans les fonctions** (le test échoue tant qu'elles manquent) :

```python
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
```

- [ ] **Étape 2 : vérifier l'échec.** `python -X utf8 serveur.py --selftest` → `NameError: name 'slug' is not defined`.

- [ ] **Étape 3 : les fonctions**, entre `FICHIERS = …` et `def selftest()` :

```python
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
```

- [ ] **Étape 4 : vérifier le succès.** `python -X utf8 serveur.py --selftest` → `selftest OK`.

- [ ] **Étape 5 : commit.**

```bash
git add serveur.py
git commit -m "Ajoute le cœur du serveur de la page web : état, avancement, lancement"
```

---

### Tâche 3 : adresses du serveur

**Fichiers :**
- Modifier : `serveur.py` (`tracks`, `window_ids`, `valid_corrections`, `make_app` ; fin de `main` ; tests du
  `selftest`)

**Interfaces :**
- Consomme : tout ce que produit la tâche 2.
- Produit : `make_app() -> FastAPI`, avec les adresses de la spécification et `GET /api/pret` ; erreurs au format
  `{"detail": message}` ; `POST /api/traitements?fichier=&nom=&debut=&fin=&voix=` (corps : la vidéo ; `fichier` : son
  nom d'origine) ; la vidéo est enregistrée sous `out/web/<nom>/<nom>.<ext>` (la page de relecture prend son titre du
  nom de la vidéo).

- [ ] **Étape 1 : tests qui échouent.** La première ligne de `selftest()` devient `global WEB, command, ollama_ok`
  (des fonctions factices remplacent le pipeline et Ollama). Puis, juste avant `print("selftest OK")` :

```python
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
```

- [ ] **Étape 2 : vérifier l'échec.** `python -X utf8 serveur.py --selftest` → `NameError: name 'make_app' is not
  defined`.

- [ ] **Étape 3 : les adresses**, juste avant `def selftest()` :

```python
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
```

La fin de `main()`, après le `selftest` :

```python
    load_profile(a.profile)  # un profil invalide arrête tout de suite
    PROFIL["nom"] = a.profile
    WEB.mkdir(parents=True, exist_ok=True)
    mark_interrupted()
    atexit.register(stop)
    import uvicorn
    print(f"Audesia : http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(make_app(), host="127.0.0.1", port=a.port, log_level="warning")
```

- [ ] **Étape 4 : vérifier le succès.** `python -X utf8 serveur.py --selftest` → `selftest OK`. Puis, Ollama lancé,
  `python -X utf8 serveur.py` en arrière-plan : `curl -s http://127.0.0.1:8000/api/pret` → `{"pret":true}` et
  `curl -s http://127.0.0.1:8000/api/traitements` → `[]` ; arrêter le serveur.

- [ ] **Étape 5 : commit.**

```bash
git add serveur.py
git commit -m "Ajoute les adresses du serveur de la page web"
```

---

### Tâche 4 : page d'accueil

**Fichiers :**
- Créer : `accueil.html`

**Interfaces :**
- Consomme : `GET /api/pret`, `GET /api/traitements`, `POST /api/traitements?fichier=&nom=&debut=&fin=&voix=` (corps :
  la vidéo), `POST /api/traitements/<nom>/relancer`, `GET /api/traitements/<nom>/export/<format>`,
  `/t/<nom>/relecture.html` ; erreurs `{"detail": message}`.

- [ ] **Étape 1 : la page** (mêmes couleurs et mêmes boutons que `relecture.html`) :

```html
<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audesia</title>
<!-- Accueil de la page web locale (serveur.py) : dépôt d'une vidéo, traitements, avancement, exports. -->
<style>
  :root {
    color-scheme: light dark;
    --fond: #f5f4f0; --surface: #fff; --texte: #1c1b19; --doux: #5b5953; --ligne: #d8d4cb;
    --accent: #1d5bb8; --sur-accent: #fff; --erreur: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fond: #151513; --surface: #1f1e1c; --texte: #ecebe7; --doux: #a9a69f; --ligne: #3b3a36;
      --accent: #8db6ff; --sur-accent: #0b1a33; --erreur: #ff8f85;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--fond); color: var(--texte); font: 16px/1.5 system-ui, "Segoe UI", sans-serif; }
  header, main { max-width: 1100px; margin: 0 auto; padding: 0 16px; }
  header { padding-top: 20px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 0 0 12px; }
  header p { margin: 0; color: var(--doux); }
  main { display: grid; gap: 20px; padding-top: 16px; padding-bottom: 32px; }
  section { background: var(--surface); border: 1px solid var(--ligne); border-radius: 10px; padding: 16px; min-width: 0; }
  form { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; align-items: end; }
  label { display: grid; gap: 4px; font-weight: 600; }
  input, select { font: inherit; color: inherit; background: var(--fond); border: 1px solid var(--ligne);
                  border-radius: 6px; padding: 6px 8px; min-width: 0; }
  button, .bouton { font: inherit; color: var(--texte); background: var(--surface); border: 1px solid var(--ligne);
                    border-radius: 6px; padding: 4px 10px; cursor: pointer; text-decoration: none; display: inline-block; }
  button:hover, .bouton:hover { border-color: var(--doux); }
  button.principal { background: var(--accent); color: var(--sur-accent); border-color: var(--accent); font-weight: 600;
                     padding: 8px 16px; }
  :focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  .aide { color: var(--doux); margin: 10px 0 0; }
  progress { width: 100%; margin-top: 12px; }
  #message { min-height: 1.5em; margin: 10px 0 0; font-weight: 600; }
  .erreur { color: var(--erreur); }
  .defile { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; vertical-align: top; padding: 10px 8px; border-top: 1px solid var(--ligne); }
  thead th { border-top: 0; font-size: .85rem; color: var(--doux); font-weight: 600; }
  .actions { display: flex; flex-wrap: wrap; gap: 6px; }
  pre { font-family: ui-monospace, Consolas, monospace; font-size: .8rem; color: var(--doux); background: var(--fond);
        border: 1px solid var(--ligne); border-radius: 6px; padding: 8px; white-space: pre-wrap; margin: 8px 0 0; }
</style>
</head>
<body>
<header>
  <h1>Audesia</h1>
  <p>Audiodescription automatique en français, 100 % locale.</p>
</header>
<main>
  <section aria-labelledby="t-depot">
    <h2 id="t-depot">Nouvelle vidéo</h2>
    <form id="depot">
      <label>Vidéo <input id="fichier" type="file" accept="video/*" required></label>
      <label>Nom <input id="nom" type="text" placeholder="d'après le fichier"></label>
      <label>Début <input id="debut" type="text" inputmode="decimal" value="0" aria-describedby="aide-temps"></label>
      <label>Fin <input id="fin" type="text" inputmode="decimal" placeholder="fin de la vidéo" aria-describedby="aide-temps"></label>
      <label>Voix <select id="voix"><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option></select></label>
      <button class="principal" type="submit">Déposer et lancer</button>
    </form>
    <p id="aide-temps" class="aide">Début et fin en secondes (95) ou en minutes:secondes (1:35). Sans fin : jusqu'au bout
      de la vidéo.</p>
    <progress id="envoi" max="1" value="0" hidden aria-label="Envoi de la vidéo"></progress>
    <p id="message" role="status"></p>
  </section>
  <section aria-labelledby="t-liste">
    <h2 id="t-liste">Traitements</h2>
    <div class="defile">
      <table>
        <thead><tr><th scope="col">Nom</th><th scope="col">État</th><th scope="col">Actions</th></tr></thead>
        <tbody id="liste"></tbody>
      </table>
    </div>
    <p id="vide" class="aide">Aucun traitement pour l'instant.</p>
  </section>
</main>
<script>
(() => {
  const $ = (s) => document.querySelector(s);
  const dire = (texte, erreur = false) => { $("#message").textContent = texte; $("#message").className = erreur ? "erreur" : ""; };
  const ETATS = {en_cours: "en cours", fini: "fini", echec: "en échec", interrompu: "interrompu", en_attente: "en attente"};
  const EXPORTS = [["mp4", "Vidéo MP4"], ["mkv", "MKV à deux pistes"], ["mp3", "Piste audiodécrite (MP3)"],
                   ["vtt", "Descriptions (WebVTT)"], ["txt", "Script"]];
  const lien = (href, texte, telecharger = false) => {
    const a = Object.assign(document.createElement("a"), {href, textContent: texte, className: "bouton"});
    if (telecharger) a.download = "";
    return a;
  };
  const detail = async (r) => { try { return (await r.json()).detail; } catch { return `Erreur ${r.status}`; } };
  let minuterie = null;

  const ligne = (t) => {
    const tr = document.createElement("tr"), nom = document.createElement("th");
    const etat = document.createElement("td"), actions = document.createElement("td"), boutons = document.createElement("div");
    nom.scope = "row";
    nom.textContent = t.nom;
    etat.textContent = ETATS[t.etat] ?? t.etat;
    if (t.etat === "en_cours") etat.textContent += ` : étape ${t.rang} sur ${t.total}, ${t.etape}${t.detail ? `, ${t.detail}` : ""}`;
    if (t.etat === "echec" || t.etat === "interrompu") {
      const pre = document.createElement("pre");
      pre.textContent = t.journal.slice(-8).join("\n");
      etat.append(pre);
    }
    boutons.className = "actions";
    if (t.relecture && t.etat !== "en_cours") boutons.append(lien(`/t/${t.nom}/relecture.html`, "Relire"));
    if (t.etat === "fini") for (const [f, texte] of EXPORTS) boutons.append(lien(`/api/traitements/${t.nom}/export/${f}`, texte, true));
    if (["echec", "interrompu", "en_attente"].includes(t.etat)) {
      const b = Object.assign(document.createElement("button"), {type: "button", textContent: "Relancer"});
      b.addEventListener("click", async () => {
        const r = await fetch(`/api/traitements/${t.nom}/relancer`, {method: "POST"});
        if (!r.ok) dire(await detail(r), true);
        charger();
      });
      boutons.append(b);
    }
    actions.append(boutons);
    tr.append(nom, etat, actions);
    return tr;
  };

  const charger = async () => {
    clearTimeout(minuterie);
    const liste = await (await fetch("/api/traitements")).json();
    $("#liste").replaceChildren(...liste.map(ligne));
    $("#vide").hidden = liste.length > 0;
    if (liste.some((t) => t.etat === "en_cours")) minuterie = setTimeout(charger, 2000);
  };

  $("#depot").addEventListener("submit", async (e) => {
    e.preventDefault();
    // Refus demandé avant l'envoi : refusée pendant l'envoi, une grosse vidéo couperait la connexion sans message.
    const pret = await fetch("/api/pret");
    if (!pret.ok) return dire(await detail(pret), true);
    const f = $("#fichier").files[0];
    const q = new URLSearchParams({fichier: f.name, nom: $("#nom").value, debut: $("#debut").value,
                                   fin: $("#fin").value, voix: $("#voix").value});
    const x = new XMLHttpRequest();  // et non fetch : seul XMLHttpRequest suit l'avancement d'un envoi
    x.open("POST", `/api/traitements?${q}`);
    x.upload.onprogress = (p) => { if (p.lengthComputable) { $("#envoi").hidden = false; $("#envoi").value = p.loaded / p.total; } };
    x.onload = () => {
      $("#envoi").hidden = true;
      let r;
      try { r = JSON.parse(x.responseText); } catch { r = {detail: `Erreur ${x.status}`}; }
      if (x.status === 200) { dire(`« ${r.nom} » est lancé.`); $("#depot").reset(); }
      else dire(r.detail, true);
      charger();
    };
    x.onerror = () => { $("#envoi").hidden = true; dire("Envoi interrompu.", true); };
    dire("Envoi de la vidéo…");
    x.send(f);
  });

  charger();
})();
</script>
</body>
</html>
```

- [ ] **Étape 2 : vérifier dans le navigateur intégré.** Ollama lancé, `python -X utf8 serveur.py` en arrière-plan, puis
  http://127.0.0.1:8000 :
  - la page s'affiche, avec « Aucun traitement pour l'instant. » ; Tab parcourt le formulaire dans l'ordre ;
  - le sélecteur de fichier ne se pilote pas depuis les outils du navigateur : remplir le champ par script, puis
    envoyer, et vérifier le message « Ce fichier n'est pas une vidéo avec du son… », sans dossier `out/web/faux` :

```js
const dt = new DataTransfer();
dt.items.add(new File(["pas une vidéo"], "faux.mp4", {type: "video/mp4"}));
document.querySelector("#fichier").files = dt.files;
document.querySelector("#depot").requestSubmit();
```

  Le dépôt d'une vraie vidéo est vérifié en tâche 6.

- [ ] **Étape 3 : commit.**

```bash
git add accueil.html
git commit -m "Ajoute la page d'accueil de la page web"
```

---

### Tâche 5 : relecture servie

**Fichiers :**
- Modifier : `relecture.html`

**Interfaces :**
- Consomme : `GET /api/traitements/<nom>` (`etat`, `rang`, `total`, `etape`, `detail`, `journal`),
  `POST /api/traitements/<nom>/corrections` (corps `{id: texte}`), `POST /api/traitements/<nom>/regenerer/<id>` ;
  erreurs `{"detail": message}`.

- [ ] **Étape 1 : commentaire, styles et HTML.** Le commentaire d'en-tête devient :

```html
<!-- Page de relecture. audesia_p0.py en écrit une copie remplie, relecture.html, dans chaque dossier de sortie : elle
     s'ouvre hors ligne, à côté de clip.mp4, clip_ad.mp4 et tts/. Servie par serveur.py (/t/<nom>/relecture.html), elle
     envoie les corrections au serveur, qui relance la voix et le mixage, et régénère une description. -->
```

Après la règle `#non-enregistre { … }` :

```css
  #progression { margin: 10px 0; font-weight: 600; color: var(--accent); }
  #progression:empty { display: none; }
```

Dans `<header>`, après `<p id="resume"></p>` :

```html
  <p id="retour" hidden><a href="/">Retour aux traitements</a></p>
```

Dans la section « Enregistrer et relancer », remplacer :

```html
      <p id="annonce" role="status"></p>
      <p>Enregistrez <code>corrections.json</code> dans le dossier de cette page, puis relancez la commande
        ci-dessous : seules les phrases modifiées sont resynthétisées. Rechargez ensuite la page.</p>
      <p><code id="commande"></code></p>
```

par :

```html
      <p id="annonce" role="status"></p>
      <p id="progression" role="status"></p>
      <p class="hors-serveur">Enregistrez <code>corrections.json</code> dans le dossier de cette page, puis relancez la commande
        ci-dessous : seules les phrases modifiées sont resynthétisées. Rechargez ensuite la page.</p>
      <p class="hors-serveur"><code id="commande"></code></p>
```

Dans le modèle de ligne, le bouton Régénérer devient :

```html
      <button class="regenerer" type="button" aria-disabled="true" title="Depuis la page web locale (python serveur.py)">Régénérer</button>
```

- [ ] **Étape 2 : script.** Après `let courante = null;` :

```js
  // Servie par serveur.py (/t/<nom>/relecture.html) : les corrections partent au serveur, qui relance la voix et le
  // mixage, et Régénérer refait une description. Ouverte depuis le disque, rien ne change.
  const nom = location.protocol.startsWith("http") && location.pathname.match(/^\/t\/([a-z0-9-]+)\//)?.[1];
  const api = nom ? `/api/traitements/${nom}` : null;
  let serveur = false, suivi = false;
  const suivre = (t) => {
    const p = $("#progression");
    if (t.etat === "en_cours") {
      suivi = true;
      p.textContent = `Traitement en cours : étape ${t.rang} sur ${t.total}, ${t.etape}${t.detail ? `, ${t.detail}` : ""}.`;
      setTimeout(() => fetch(api).then((r) => r.json()).then(suivre).catch(() => {
        p.textContent = "Le serveur ne répond plus : relancez python serveur.py, puis rechargez la page.";
      }), 2000);
    } else if (t.etat === "fini") {
      if (suivi) location.reload();  // nouvelles voix, nouveau mixage
    } else {
      p.textContent = `Le dernier traitement ne s'est pas terminé (${t.journal.at(-1) ?? "sans message"}) : ` +
        "relancez-le depuis la page d'accueil.";
    }
  };
  const activer = (t) => {
    serveur = true;
    $("#enregistrer").textContent = "Enregistrer et relancer";
    for (const e of document.querySelectorAll(".hors-serveur")) e.hidden = true;
    $("#retour").hidden = false;
    for (const b of document.querySelectorAll(".regenerer")) { b.removeAttribute("aria-disabled"); b.removeAttribute("title"); }
    suivre(t);
  };
```

Remplacer le gestionnaire du bouton Régénérer :

```js
    $(".regenerer", l.tr).addEventListener("click", () =>
      annonce("Régénérer demandera le serveur local prévu en P1 : pour l'instant, écrivez le texte ou supprimez la description."));
```

par :

```js
    $(".regenerer", l.tr).addEventListener("click", async () => {
      if (!serveur) return annonce("Régénérer fonctionne depuis la page web locale : lancez python serveur.py.");
      if (JSON.stringify(corrections()) !== enregistre) return annonce("Enregistrez d'abord vos corrections : la régénération relance le traitement.");
      const r = await fetch(`${api}/regenerer/${l.r.id}`, {method: "POST"}), t = await r.json();
      if (!r.ok) return annonce(t.detail);
      annonce(`${l.r.horaire} : nouvelle description en cours, la page se rechargera à la fin.`);
      suivre(t);
    });
```

Dans `enregistrer`, juste après sa première ligne (`const c = corrections(), texte = …;`) :

```js
    if (serveur) {  // le serveur écrit corrections.json et relance la voix et le mixage
      const r = await fetch(`${api}/corrections`, {method: "POST", headers: {"Content-Type": "application/json"},
                                                   body: JSON.stringify(c)});
      const t = await r.json();
      if (!r.ok) return annonce(t.detail);  // refusé (traitement en cours…) : les modifications restent à enregistrer
      enregistre = JSON.stringify(c);
      rafraichir();
      annonce("Corrections enregistrées : la voix et le mixage sont relancés, la page se rechargera à la fin.");
      return suivre(t);
    }
```

Enfin, après les deux lignes qui suivent la création des lignes du tableau (`enregistre = JSON.stringify(corrections());`
puis `rafraichir();`) :

```js
  if (api) fetch(api).then((r) => r.ok ? r.json() : null).then((t) => t && activer(t)).catch(() => {});
```

- [ ] **Étape 3 : vérifier.** Ouvrir le modèle depuis le disque (`file:///F:/git/AudesIA/relecture.html`) dans le
  navigateur intégré : le message « Cette page est le modèle… » s'affiche (le script s'exécute donc sans erreur de
  syntaxe), et la console est vide. Le mode serveur et la copie remplie hors ligne sont vérifiés en tâche 6.

- [ ] **Étape 4 : commit.**

```bash
git add relecture.html
git commit -m "Active la page de relecture quand le serveur la sert"
```

---

### Tâche 6 : de bout en bout, documentation

**Fichiers :**
- Modifier : `README.md` (relecture, page web, options, feuille de route), `AUDESIA.md` (P1)

- [ ] **Étape 1 : préparer.** Un extrait de 25 s de *Sintel* VF, servi par le serveur pour le remplir dans le champ
  de fichier (le dossier `_essai`, sans `etat.json`, n'est pas un traitement) :

```bash
mkdir -p out/web/_essai
ffmpeg -ss 95 -t 25 -i corpus/media/Sintel.fr.touhoppai.1080p.mp4 -c copy out/web/_essai/sintel-25s.mp4
```

  Ollama lancé, `python -X utf8 serveur.py` en arrière-plan, http://127.0.0.1:8000 dans le navigateur intégré.

- [ ] **Étape 2 : les tests de la spécification.**
  1. Dépôt : remplir le champ par script, puis « Déposer et lancer » (nom vide, voix Vivian) ; l'avancement passe par
     les 7 étapes (« vision k sur N » puis « rédaction k sur N » pendant la description), puis « fini », avec Relire
     et les cinq exports :

```js
const b = await (await fetch("/t/_essai/sintel-25s.mp4")).blob();
const dt = new DataTransfer();
dt.items.add(new File([b], "sintel-25s.mp4", {type: "video/mp4"}));
document.querySelector("#fichier").files = dt.files;
```

  2. « Relire » : titre « sintel-25s », bouton « Enregistrer et relancer », commande masquée, Régénérer actif.
     Corriger une phrase, enregistrer : l'avancement s'affiche, la page se recharge à la fin, la ligne est « relue ».
     `curl -sI http://127.0.0.1:8000/t/sintel-25s/tts/<id>.wav` montre une date récente (`last-modified`) et
     `cache-control: no-cache`.
  3. « Régénérer » sur une autre ligne : après rechargement, sa description factuelle et ses variantes ont changé
     (comparer `relecture.json` avant et après).
  4. Exports, par `curl -s -o out/web/_essai/export.<format> http://127.0.0.1:8000/api/traitements/sintel-25s/export/<format>`
     pour `mp4`, `mkv`, `mp3`, `vtt` et `txt` : `ffprobe` lit le MP4, le MKV (deux pistes audio et une piste de
     sous-titres) et le MP3 ; le VTT et le script se lisent.
  5. `file:///F:/git/AudesIA/out/web/sintel-25s/relecture.html` : « Enregistrer les corrections », commande affichée,
     Régénérer annonce qu'il faut la page web ; aucune erreur dans la console.
  6. Refus, avec leur message dans la page : un dépôt pendant une relance (« Un traitement tourne déjà… ») ; Ollama
     fermé (`Stop-Process -Name "ollama app", ollama`, puis un dépôt : « Ollama ne répond pas… » ; le relancer par
     `Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"`) ; le faux fichier de la tâche 4
     (« Ce fichier n'est pas une vidéo avec du son… »).

  Puis `rm -r out/web/_essai` ; le traitement `sintel-25s` reste, prêt pour la démonstration.

- [ ] **Étape 3 : documentation.** Dans `README.md` :
  - fin du paragraphe « Enregistrer les corrections » : « « Régénérer » attend le serveur local du P1. » devient
    « Depuis la page web (ci-dessous), le serveur relance tout lui-même, et « Régénérer » refait une description. » ;
  - la phrase « En P1, un serveur local servira la même page, relancera lui-même la voix et le mixage, et régénérera une
    description à la demande. » est remplacée par :

```markdown
### Page web

`python serveur.py` ouvre une page locale, http://127.0.0.1:8000, pour tout faire sans ligne de commande :

- déposer une vidéo, avec le début et la fin de l'extrait et la voix, puis suivre le traitement étape par étape ;
- relire : la même page de relecture, où « Enregistrer et relancer » envoie les corrections au serveur, qui relance
  lui-même la voix et le mixage, et où « Régénérer » refait une description ;
- exporter : vidéo MP4, MKV à deux pistes, piste audiodécrite en MP3 (téléversable comme piste d'audiodescription sur
  YouTube), descriptions WebVTT, script texte.

Le serveur n'écoute que la machine elle-même et traite une vidéo à la fois, dans `out/web/<nom>/`. Régénérer relit les
images de la seule fenêtre choisie, avec un peu de hasard, puis refait sa révision, sa rédaction et sa voix ; en ligne
de commande : `--regenerer d_0007`.
```

  - dans le tableau « Options utiles », après la ligne `--voice` :

```markdown
| `--regenerer` | Refait la description de ces fenêtres (`d_0007,d_0012`), lecture des images comprise, puis la voix et le mixage |
```

  - dans « Feuille de route », retirer « une page web pour tout faire sans ligne de commande (dépôt, suivi, relecture,
    export), à partir de la page de relecture du P0, » de la ligne P1.

  Dans `AUDESIA.md`, l'élément P1 « Une page web pour tout faire sans ligne de commande… » et ses quatre sous-points
  deviennent :

```markdown
- [x] Une page web pour tout faire sans ligne de commande : `python serveur.py` (FastAPI, http://127.0.0.1:8000, local seulement) :
  - dépôt de la vidéo (début, fin, voix), puis avancement étape par étape ;
  - relecture : la page `relecture.html` du P0, qui envoie les corrections au serveur (même `corrections.json`, voix et mixage relancés) et régénère une description (`audesia_p0.py --regenerer` : lecture des images de la seule fenêtre choisie, à température 0,7, puis révision, rédaction et voix) ;
  - export : MP4, MKV à deux pistes, piste audiodécrite en MP3, WebVTT, script texte ;
  - spécification et plan dans `docs/specs/`.
```

- [ ] **Étape 4 : vérifications et commit.**

```bash
python -X utf8 audesia_p0.py --selftest
python -X utf8 serveur.py --selftest
git add README.md AUDESIA.md
git commit -m "Documente la page web de démonstration"
git push
```

Attendu : `selftest OK` deux fois ; avant le commit, `git grep -i` sur le nom de l'outil d'assistance ne trouve rien.
