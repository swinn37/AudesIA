#!/usr/bin/env python3
"""Audesia P0 : audiodescription française d'un extrait vidéo, en un seul fichier.

Chaîne : ffmpeg → Silero VAD + Whisper large-v3 → PySceneDetect → VLM (description)
→ rédacteur (3 variantes calées sur le silence) → Qwen3-TTS → mixage → exports.
VLM et rédacteur passent par une API compatible OpenAI : passer au GX10 (vLLM)
ne demande que --base-url et --model.

Installation (WSL2, RTX 5080) :
  sudo apt install ffmpeg sox
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install qwen-tts silero-vad scenedetect openai
  curl -fsSL https://ollama.com/install.sh | sh
  OLLAMA_CONTEXT_LENGTH=8192 ollama serve &    # le contexte par défaut est trop court pour 4 images
  ollama pull gemma4:26b-a4b-it-qat

Sintel (Blender Foundation, CC BY 3.0) : https://download.blender.org/durian/movies/
De 1:35 à 3:35 (version originale) : la scène de la hutte, 12 répliques séparées
de silences courts, puis 51 s sans dialogue.

Usage :
  python audesia_p0.py sintel.mkv --start 1:35 --end 3:35
  python audesia_p0.py --selftest

Chaque étape coûteuse écrit un JSON dans le dossier de sortie et n'est pas refaite
s'il existe : supprimer descriptions.json pour relancer la rédaction. La voix et le
mixage sont refaits à chaque lancement (changer --voice ou --cps ne coûte que ça).
"""
import argparse
import base64
import difflib
import gc
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # contrainte n° 1 : aucune télémétrie

VAD_THRESHOLD = 0.35  # Silero plus sensible que par défaut (0,5) : dans le doute, c'est de la parole
VAD_PAD_MS = 200      # marge autour de chaque segment de parole
WHISPER_S_PER_WORD = 0.8  # s : un segment Whisper plus lent (+ 1 s) déborde sur la musique et est ignoré
WHISPER_PAD = 0.5         # s : Whisper fait commencer les chuchotements trop tard (« Hé… », « Chut »)
MIN_SILENCE = 2.0   # s : silence utilisable minimal (à 1,5 s, des descriptions d'un mot)
MARGIN = 0.2        # s : marge avant et après chaque description
WINDOW = 5.0        # s : durée visée d'une fenêtre dans un long silence (coupée aux changements de plan)
MAX_WINDOW = 10.0   # s
MAX_IMAGES = 4      # 8 suivait mieux les actions, mais la description passait de 9 à 50 min sur la 5080
FRAME_STEP = 0.7    # s : écart visé entre deux images dans les plans courts
MAX_CAST = 10       # images (une par plan) pour le registre des personnages
DARK_LEVEL = 40     # luminance sous laquelle un pixel compte comme très sombre (0-255)
DARK_SHARE = 0.25   # contre-jour : plus de 25 % de pixels très sombres…
BRIGHT_LEVEL = 170  # …et plus de 5 % de pixels clairs (> BRIGHT_LEVEL) : le modèle reçoit aussi une copie éclaircie
BRIGHT_SHARE = 0.05  # (mesuré : débris à contre-jour 42-55 % sombres / 9-14 % clairs ; hutte sombre 90-97 % / 0 %)
GAMMA = 2.2         # éclaircissement de cette copie : à contre-jour, des débris passaient pour un « tissu noir »
VAGUE = re.compile(r"\b(objets?|sphères?|sphériques?|masses?|formes?|choses?|éléments?)\b", re.I)
MAX_SPEEDUP = 1.10
REMENTION_S = 15.0  # s : au-delà, un personnage est redésigné en entier plutôt que par « elle » ou « il »
REPEAT_RATIO = 0.6  # similarité au-delà de laquelle une description redit la précédente (redite mesurée : 0,75)
DUCK = 0.3          # gain de la bande-son sous la voix
RAMP = 0.25         # s : rampe d'atténuation
TTS_INSTRUCT = "Voix calme et neutre de narrateur, diction claire, débit régulier."

DESCRIBE = (
    "Tu prépares l'audiodescription d'un film. Les images sont extraites, dans l'ordre, d'un passage sans "
    "dialogue ; des versions éclaircies d'images sombres peuvent suivre. Décris ce qui se passe : la suite des actions d'une "
    "image à l'autre (ce que font les personnages, ce qu'ils regardent, ce qu'ils tiennent), le lieu, les "
    "objets et les animaux désignés par leur nom courant quand ils sont reconnaissables, plutôt que par une "
    "catégorie vague, les états visibles (blessé, effrayé, souriant) et les textes lisibles. N'interprète pas "
    "les intentions et n'invente rien : ce qui est trop flou pour être reconnu, ne le mentionne pas, et ne "
    "remplace pas un objet ambigu par un objet familier. Réponds en français, en 3 à 5 phrases factuelles."
)
REVISE = """Tu révises la description d'un plan de film pour une audiodescription. Tu reçois les images du \
plan, le registre des personnages et les descriptions du plan précédent, de ce plan et du plan suivant. \
Réécris la description de ce plan pour qu'elle soit fidèle aux images et cohérente avec le reste du film :
- désigne chaque personnage avec la désignation du registre, toujours la même ;
- attribue une action, une main ou une silhouette à un personnage du registre quand les images (vêtements, gants, \
position) et l'enchaînement des plans le montrent ;
- nomme précisément ce que les images rendent évident, avec les états visibles (blessé, effrayé) ;
- des versions éclaircies peuvent suivre : ce sont les mêmes images, sers-t'en pour reconnaître ce qui est sombre ; \
une précision obtenue sur des détails agrandis peut aussi être fournie : reprends-la ;
- décris les gestes tels qu'ils se voient, sans leur prêter d'intention ;
- si tu reconnais le type d'un objet ou d'un être vivant mais pas son espèce exacte avec certitude, nomme le type \
et décris son aspect plutôt que de risquer un nom d'espèce ;
- retire tout ce que les images ne montrent pas clairement, et n'ajoute rien qui n'y soit visible.
Réponds uniquement par la description révisée, en français, en 2 à 4 phrases."""
CAST = (
    "Ces images viennent, dans l'ordre, d'un même extrait de film. Repère les personnages visibles (humains, "
    "animaux, créatures) et regroupe ceux qui sont la même personne d'une image à l'autre. Pour chacun, donne la "
    "désignation la plus précise que les images permettent, courte et stable pour une audiodescription (« le "
    "garçon au chapeau rouge » plutôt qu'« un enfant », « le chien noir » plutôt qu'« un animal »), et ses traits "
    "visuels distinctifs (cheveux, vêtements, accessoires). N'invente aucun personnage. Réponds uniquement en "
    'JSON : {"personnages": [{"designation": "...", "traits": "..."}]}'
)
ZOOM = (
    "La première image est un plan de film, les trois suivantes des détails agrandis de ce plan (gauche, centre, "
    "droite). Que tiennent ou manipulent les personnages, et quels objets reconnais-tu ? Nomme précisément ce que tu "
    "reconnais, n'invente rien. Deux phrases. Si tu ne reconnais rien de précis, réponds « inconnu »."
)
WRITE = """Tu es audiodescripteur. À partir de la description factuelle d'un passage, écris \
l'audiodescription qui sera lue par une voix de synthèse dans un silence entre deux dialogues.
Règles (Charte de l'audiodescription) :
- présent de l'indicatif, troisième personne, phrases courtes, mots simples et précis ;
- jamais « on voit » ni « nous voyons » ;
- uniquement ce qui est visible : qui, quoi, où ; ni interprétation ni anticipation ;
- privilégie l'action principale du plan et les états visibles des personnages (blessé, effrayé) ;
- n'ajoute rien qui ne figure pas dans la description ;
- ne répète ni les dialogues ni ce qui a déjà été décrit ;
- ne nomme un personnage que si son nom a été prononcé ou affiché ; sinon, sa désignation \
dans la liste des personnages, toujours la même ;
- ne répète pas la désignation complète d'un personnage déjà désigné juste avant : écris « elle », « il » \
ou une forme courte de sa désignation ; si deux personnages du même genre sont en jeu, garde leurs formes \
courtes plutôt que « elle » ou « il » ;
- lis les textes importants à l'écran ;
- chaque variante est complète et se termine par un point.
Réponds uniquement en JSON : {"variantes": ["...", "...", "..."]}, trois variantes de longueur \
décroissante, la première sans dépasser la longueur maximale indiquée."""

TIMINGS = {}


# --- Logique de calage (pure, couverte par --selftest) -------------------------------------------

def secs(t):
    """'1:35' → 95.0 ; accepte s, mm:ss ou hh:mm:ss."""
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(str(t).split(":"))))


def merge(intervals):
    """Fusionne les intervalles [début, fin] qui se chevauchent."""
    out = []
    for s, e in sorted(intervals):
        if e <= s:  # intervalle vide ou inversé (Whisper en produit) : il casserait silences()
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def silences(speech, duration, min_len=MIN_SILENCE):
    """Complément de la parole dans [0, durée], limité aux silences utilisables."""
    out, t = [], 0.0
    for s, e in merge(speech):
        if s - t >= min_len:
            out.append([t, s])
        t = max(t, e)
    if duration - t >= min_len:
        out.append([t, duration])
    return out


def plausible_segments(dialogues):
    """Segments Whisper au débit plausible : ils rattrapent les chuchotements que la VAD manque,
    sans les segments étirés sur la musique (une phrase de 15 mots sur 53 s, sur Sintel)."""
    return [[d["start"] - WHISPER_PAD, d["end"] + WHISPER_PAD] for d in dialogues
            if d["text"] and d["end"] - d["start"] <= WHISPER_S_PER_WORD * len(d["text"].split()) + 1.0]


def windows(silence, shots):
    """Découpe un silence en fenêtres d'au moins WINDOW s coupées aux changements de plan,
    puis en parts égales d'au plus MAX_WINDOW s : une description par fenêtre."""
    s, e = silence
    out, a = [], s
    for b in [st for st, _ in shots if s < st < e] + [e]:
        if b - a >= WINDOW or b == e:
            out.append([a, b])
            a = b
    if len(out) > 1 and out[-1][1] - out[-1][0] < MIN_SILENCE:
        out[-2:] = [[out[-2][0], out[-1][1]]]
    split = []
    for a, b in out:
        n = math.ceil((b - a) / MAX_WINDOW)
        split += [[a + (b - a) * i / n, a + (b - a) * (i + 1) / n] for i in range(n)]
    return split


def frame_times(win):
    """De 3 à MAX_IMAGES images réparties sur la fenêtre, environ FRAME_STEP s d'écart dans les plans courts."""
    a, b = win
    n = min(MAX_IMAGES, max(3, round((b - a) / FRAME_STEP)))
    return [a + (b - a) * (i + 0.5) / n for i in range(n)]


def budget(win, cps):
    """Nombre maximal de caractères lisibles dans la fenêtre, marges déduites."""
    return int((win[1] - win[0] - 2 * MARGIN) * cps)


def fit(variants, avail, synth):
    """Variante la plus longue dont la voix tient dans avail secondes ; sinon la plus courte accélérée
    d'au plus MAX_SPEEDUP ; sinon None (description abandonnée)."""
    # ponytail: pas de relance « raccourcis » vers le rédacteur en P0 ; fit_loop.py la fera en P1.
    got = None
    for text in sorted(variants, key=len, reverse=True):
        wav, sr = synth(text)
        if len(wav) / sr <= avail:
            return text, wav, sr, 1.0
        got = text, wav, sr, len(wav) / sr / avail
    return got if got and got[3] <= MAX_SPEEDUP else None


def short_form(name):
    """« la jeune fille aux cheveux roux » → « la jeune fille » : la désignation sans ses compléments."""
    return re.split(r" (?:aux?|à|en|avec|vêtue?s?|portant|dont) ", name, maxsplit=1)[0]


def lighten(text, names, recent, gap):
    """Évite de redire la désignation complète d'un personnage désigné à la description précédente : « elle » ou
    « il » en tête de phrase (si aucun autre personnage récent n'a le même genre), forme courte ailleurs."""
    if gap > REMENTION_S:
        return text
    for name in names:
        if name not in recent:
            continue
        article = name.split()[0].lower()
        pronoun = {"la": "Elle", "le": "Il"}.get(article)
        if pronoun and not any(n != name and n.split()[0].lower() == article for n in recent):
            text = re.sub(r"(^|[.!?]\s+)" + re.escape(name), lambda m: m.group(1) + pronoun, text, flags=re.I)
        text = re.sub(re.escape(name), short_form(name), text, flags=re.I)
    return re.sub(r"(^|[.!?]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), text)


def mentioned(text, names, recent):
    """Personnages désignés dans text, plus ceux de recent qu'un « elle » ou « il » initial continue de désigner."""
    low = text.lower()
    found = {n for n in names if n.lower() in low or short_form(n).lower() in low}
    lead = re.match(r"(elle|il)\b", low)
    if lead:
        found |= {n for n in recent if n.split()[0].lower() == {"elle": "la", "il": "le"}[lead.group(1)]}
    return found


def repeats(text, previous):
    """Vrai si text redit presque la description précédente (similarité des caractères ≥ REPEAT_RATIO)."""
    return difflib.SequenceMatcher(None, text.lower(), previous.lower()).ratio() >= REPEAT_RATIO


def duck_envelope(n, sr, spans):
    """Gain de la bande-son : DUCK pendant chaque description, rampes de RAMP s, 1 ailleurs."""
    import numpy as np
    env = np.ones(n, dtype=np.float32)
    for s, e in spans:
        i, j = max(int((s - RAMP) * sr), 0), min(int((e + RAMP) * sr) + 1, n)
        env[i:j] = np.minimum(env[i:j], np.interp(np.arange(i, j) / sr, [s - RAMP, s, e, e + RAMP], [1, DUCK, DUCK, 1]))
    return env


def json_list(raw, key):
    """Liste rangée sous key dans le premier objet JSON de raw ; [] si elle manque ou est illisible."""
    try:
        v = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])[key]
    except (ValueError, KeyError, TypeError):
        return []
    return v if isinstance(v, list) else []


def parse_variants(raw):
    """Variantes du JSON du rédacteur, dédoublonnées, d'au moins deux mots, de la plus longue à la plus courte."""
    return sorted({s.strip() for s in json_list(raw, "variantes") if isinstance(s, str) and len(s.split()) >= 2},
                  key=len, reverse=True)


def vtt(items):
    def ts(x):
        return f"{int(x // 3600):02}:{int(x % 3600 // 60):02}:{x % 60:06.3f}"
    return "WEBVTT\n\n" + "".join(f"{ts(i['start'])} --> {ts(i['end'])}\n{i['text']}\n\n" for i in items)


# --- Étapes ------------------------------------------------------------------------------------

def ff(*args):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ffmpeg : {r.stderr.strip()}")


def step(name, fn, cache=None):
    """Lance une étape, la chronomètre, et la met en cache dans un JSON si cache est donné."""
    t0 = time.perf_counter()
    if cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = fn()
        if cache:
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    TIMINGS[name] = round(time.perf_counter() - t0, 1)
    print(f"[{name}] {TIMINGS[name]} s", flush=True)
    return data


def detect_speech(wav16):
    """Segments de la VAD et transcription Whisper ; main() y ajoute les segments Whisper plausibles."""
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    from transformers import pipeline

    audio, sr = sf.read(wav16, dtype="float32")
    duration = len(audio) / sr
    vad = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=sr, threshold=VAD_THRESHOLD,
                                speech_pad_ms=VAD_PAD_MS, return_seconds=True)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", dtype=torch.float16, device="cuda:0")
    chunks = asr({"raw": audio, "sampling_rate": sr}, return_timestamps=True, chunk_length_s=30, batch_size=8,
                 generate_kwargs={"task": "transcribe"})["chunks"]
    del asr
    gc.collect()
    torch.cuda.empty_cache()  # libère la VRAM pour le VLM et la voix
    dialogues = [{"start": c["timestamp"][0], "end": c["timestamp"][1] or duration, "text": c["text"].strip()}
                 for c in chunks]
    return {"duration": duration, "speech": merge([[v["start"], v["end"]] for v in vad]), "dialogues": dialogues}


def detect_shots(clip):
    from scenedetect import AdaptiveDetector, detect
    return [[s.seconds, e.seconds] for s, e in detect(str(clip), AdaptiveDetector())]


def as_image(path):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}}


def backlit(path):
    """Contre-jour : une grande zone très sombre et un fond clair. Une scène entièrement sombre ne compte pas :
    éclaircie, elle ne gagne rien et prête à confusion."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    return bool((a < DARK_LEVEL).mean() > DARK_SHARE and (a > BRIGHT_LEVEL).mean() > BRIGHT_SHARE)


def brighten(path):
    """Copie éclaircie (gamma) d'une image à grande zone sombre."""
    from PIL import Image
    bright = path.with_name(path.stem + "_clair.jpg")
    Image.open(path).point(lambda v: round(255 * (v / 255) ** (1 / GAMMA))).save(bright, quality=90)
    return bright


def tiles(clip, t, folder):
    """Trois détails en pleine résolution (gauche, centre, droite) de l'image à t, pour nommer un petit objet :
    réduit, un fruit à piquants n'était qu'un « objet sphérique » ; agrandi, il est reconnu."""
    paths = []
    for i in range(3):
        p = folder / f"{t:08.2f}_tuile{i}.jpg"
        ff("-ss", t, "-i", clip, "-frames:v", 1, "-vf", f"crop=iw*0.4:ih:iw*0.3*{i}:0", "-q:v", 3, p)
        paths.append(p)
    return paths


def chat(llm, model, system, content, json_mode=False):
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    # Sans reasoning_effort="none", Gemma 4 sous Ollama « réfléchit » 2 à 3 min et rend un contenu vide.
    r = llm.chat.completions.create(model=model, temperature=0, reasoning_effort="none", messages=[
        {"role": "system", "content": system}, {"role": "user", "content": content}], **extra)
    return (r.choices[0].message.content or "").strip()


def describe(clip, segs, shots, a, out):
    """1. Par fenêtre : description de la suite d'images seule (VLM).
    2. Sur l'ensemble : registre des personnages.
    3. Par fenêtre : révision sur les mêmes images avec le registre et les plans voisins, puis 3 variantes calées."""
    # ponytail: révision en un appel par le même modèle ; vérification fait par fait en P1 si besoin.
    from openai import OpenAI
    llm = OpenAI(base_url=a.base_url, api_key="local")
    (out / "frames").mkdir(exist_ok=True)
    wins = [w for sil in segs["silences"] for w in windows(sil, shots)]
    items = []
    for k, w in enumerate(wins):
        times, images, bright = frame_times(w), [], []
        for t in times:
            f = out / "frames" / f"{t:08.2f}.jpg"
            ff("-ss", t, "-i", clip, "-frames:v", 1, "-vf", "scale=1024:-2", "-q:v", 3, f)
            images.append(as_image(f))
            if backlit(f):
                bright.append(as_image(brighten(f)))
        extra = [{"type": "text", "text": "Versions éclaircies des images sombres :"}, *bright] if bright else []
        # Images seules : en contexte, les répliques (« Cette lame… ») et les descriptions précédentes
        # amorçaient des inventions qui se propageaient d'une fenêtre à l'autre.
        raw = chat(llm, a.model, DESCRIBE, [{"type": "text", "text": "Décris cette suite d'images."}, *images, *extra])
        items.append({"id": f"d_{k:04d}", "window": w, "frames": times, "brightened": len(bright),
                      "raw_description": raw, "images": images, "extra": extra})
        print(f"  {k + 1}/{len(wins)}  {w[0]:6.1f}–{w[1]:6.1f} s  décrit ({len(images)} images, "
              f"{len(bright)} éclaircies)", flush=True)

    # Une image par plan, pour que la même personne garde la même désignation tout au long de l'extrait.
    picks = items if len(items) <= MAX_CAST else [items[int(i * len(items) / MAX_CAST)] for i in range(MAX_CAST)]
    cast = [c for c in json_list(chat(llm, a.model, CAST, [{"type": "text", "text": "Images, dans l'ordre :"},
                                                          *[x["images"][len(x["images"]) // 2] for x in picks]],
                                      json_mode=True), "personnages") if isinstance(c, dict)]
    registry = "\n".join(f"- {c.get('designation', '')} : {c.get('traits', '')}" for c in cast) or "aucun"
    (out / "personnages.json").write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  registre : {[c.get('designation') for c in cast]}", flush=True)

    said, previous = [], "aucun"
    for k, x in enumerate(items):
        w = x["window"]
        visuals = x.pop("images") + x.pop("extra")  # pas de base64 dans le cache JSON
        note, x["zoom"] = "", None
        if VAGUE.search(x["raw_description"]):
            # Objet nommé vaguement : question directe sur des détails agrandis de l'image centrale, sans la
            # description (elle ancrait la réponse sur « objet sphérique »). Envoyés en vrac dans la révision,
            # ces détails ne suffisaient pas à la faire sortir de « objet ».
            mid = x["frames"][len(x["frames"]) // 2]
            zoomed = [as_image(out / "frames" / f"{mid:08.2f}.jpg"), *[as_image(p) for p in tiles(clip, mid, out / "frames")]]
            x["zoom"] = chat(llm, a.model, ZOOM, [{"type": "text", "text": "Plan et détails agrandis :"}, *zoomed])
            if "inconnu" not in x["zoom"].lower():
                note = f"\nPrécision sur l'objet, d'après des détails agrandis : {x['zoom']}"
        following = items[k + 1]["raw_description"] if k + 1 < len(items) else "aucun"
        # Révision sur les images, avec le registre et les plans voisins : désignations stables, actions
        # rattachées au bon personnage, rien de ce que les images ne montrent pas.
        desc = chat(llm, a.model, REVISE, [{"type": "text", "text": (
            f"Registre des personnages :\n{registry}\n\nPlan précédent : {previous}\n"
            f"Ce plan : {x['raw_description']}{note}\nPlan suivant : {following}")}, *visuals]) or x["raw_description"]
        previous = desc
        heard = " / ".join(d["text"] for d in segs["dialogues"] if d["end"] <= w[1])[-600:] or "aucun"
        context = (f"Personnages :\n{registry}\nDéjà décrit : {' '.join(said[-3:]) or 'rien'}\n"
                   f"Dialogues entendus jusqu'ici : {heard}")
        b = budget(w, a.cps)
        limits = "\n".join(f"Variante {i + 1} : au plus {n} caractères (environ {max(n // 6, 1)} mots)."
                           for i, n in enumerate((b, b * 2 // 3, b // 2)))
        variants = parse_variants(chat(llm, a.model, WRITE, f"Description : {desc}\n{context}\n{limits}", json_mode=True))
        x.update(description=desc, budget_chars=b, variants=variants)
        said += variants[:1]
        print(f"  {w[0]:6.1f}–{w[1]:6.1f} s  {'[zoom : ' + x['zoom'][:60] + '] ' if x['zoom'] else ''}{variants[:1]}",
              flush=True)
    return items


def unload_ollama(base_url, model):
    """Ollama garde le modèle en VRAM 5 min : on le libère pour la voix. Sans effet sur vLLM ou llama.cpp."""
    import urllib.request
    req = urllib.request.Request(base_url.rsplit("/v1", 1)[0] + "/api/generate", method="POST",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except OSError:
        pass


def voice(descs, segs, a, out):
    """Synthétise la variante la plus longue qui tient dans chaque fenêtre et la place sur une piste."""
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(a.tts_model, device_map="cuda:0", dtype=torch.bfloat16)

    def synth(text):
        wavs, sr = tts.generate_custom_voice(text=text, language="French", speaker=a.voice, instruct=TTS_INSTRUCT)
        return np.asarray(wavs[0], dtype=np.float32), sr

    (out / "tts").mkdir(exist_ok=True)
    info = sf.info(out / "audio48k.wav")
    sr, n = info.samplerate, info.frames
    track = np.zeros(n, dtype=np.float32)
    registry = out / "personnages.json"
    names = [c["designation"].strip() for c in (json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else [])
             if isinstance(c, dict) and c.get("designation")]
    placed, dropped, recent, last_end, last_text = [], [], set(), -REMENTION_S, ""
    for d in descs:
        (w0, w1), start = d["window"], d["window"][0] + MARGIN
        # Fluidité : « elle » ou forme courte plutôt que la désignation complète répétée d'un plan à l'autre,
        # et pas de redite de la description qui vient d'être lue (le plan reste alors silencieux).
        variants = [lighten(v, names, recent, start - last_end) for v in d["variants"]]
        if start - last_end <= REMENTION_S:
            variants = [v for v in variants if not repeats(v, last_text)]
        got = fit(variants, w1 - w0 - 2 * MARGIN, synth)
        if not got:
            dropped.append(d["id"])
            continue
        text, wav, tsr, speed = got
        raw, final = out / "tts" / f"{d['id']}_raw.wav", out / "tts" / f"{d['id']}.wav"
        sf.write(raw, wav, tsr)
        ff("-i", raw, "-af", f"atempo={speed}", "-ar", sr, "-ac", 1, final)  # accélération ≤ 10 % + 48 kHz
        clip_audio = sf.read(final, dtype="float32")[0]
        i = int(start * sr)
        clip_audio = clip_audio[: n - i]
        track[i: i + len(clip_audio)] += clip_audio
        placed.append({"id": d["id"], "start": round(start, 3), "end": round(start + len(clip_audio) / sr, 3),
                       "text": text, "speedup": round(speed, 3), "chars_per_s": round(len(text) * tsr / len(wav), 1)})
        recent, last_end, last_text = mentioned(text, names, recent), placed[-1]["end"], text
    for p in placed:  # invariant : aucune description ne chevauche la parole détectée
        assert not any(s < p["end"] and p["start"] < e for s, e in segs["speech"]), p
    sf.write(out / "ad_voice.wav", track, sr)
    return placed, dropped


def mix_and_export(placed, out):
    """Bande-son atténuée sous la voix, puis MP4 avec AD, MKV à deux pistes et WebVTT."""
    # ponytail: atténuation par enveloppe ; sidechaincompress si le rendu manque de naturel.
    import numpy as np
    import soundfile as sf
    orig, sr = sf.read(out / "audio48k.wav", dtype="float32")
    track = sf.read(out / "ad_voice.wav", dtype="float32")[0]
    mix = orig * duck_envelope(len(orig), sr, [(p["start"], p["end"]) for p in placed])[:, None] + track[:, None]
    mix *= min(1.0, 0.99 / max(float(np.abs(mix).max()), 1e-9))  # évite la saturation
    sf.write(out / "ad_mix.wav", mix, sr)
    (out / "ad.vtt").write_text(vtt(placed), encoding="utf-8")
    clip = out / "clip.mp4"
    ff("-i", clip, "-i", out / "ad_mix.wav", "-map", "0:v", "-map", "1:a",
       "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out / "clip_ad.mp4")
    ff("-i", clip, "-i", out / "ad_mix.wav", "-i", out / "ad.vtt",
       "-map", "0:v", "-map", "0:a", "-map", "1:a", "-map", "2:s",
       "-c:v", "copy", "-c:a:0", "copy", "-c:a:1", "aac", "-b:a:1", "192k", "-c:s", "webvtt",
       "-metadata:s:a:1", "language=fra", "-metadata:s:a:1", "title=Audiodescription",
       "-metadata:s:s:0", "language=fra",
       "-disposition:a:0", "default", "-disposition:a:1", "visual_impaired", "-disposition:s:0", "descriptions",
       out / "clip_ad.mkv")


def selftest():
    import numpy as np
    assert secs("1:35") == 95 and secs("0:01:35.5") == 95.5 and secs(12) == 12
    assert merge([[3, 4], [0, 1], [0.5, 2]]) == [[0, 2], [3, 4]]
    assert merge([[0, 2], [29.8, 27.9], [5, 6]]) == [[0, 2], [5, 6]]  # intervalle inversé ignoré
    assert silences([[0, 2], [29.8, 27.9], [30.1, 31]], 40) == [[2, 30.1], [31, 40]]  # plus de silences qui se chevauchent
    speech = [[2, 4], [4.5, 6], [9, 10]]
    sil = silences(speech, 12)
    assert sil == [[0, 2], [6, 9], [10, 12]]                      # 4–4,5 s : trop court
    shots = [[0, 2], [2, 6], [6, 9], [9, 15], [15, 40]]
    assert windows([0, 17], shots) == [[0, 6], [6, 15], [15, 17]]
    assert windows([0, 7], [[0, 5.8], [5.8, 7]]) == [[0, 7]]      # reliquat < MIN_SILENCE fusionné
    assert len(windows([20, 71], [[0, 100]])) == 6                # 51 s d'un seul plan → 6 × 8,5 s
    for w in (w for s in sil for w in windows(s, shots)):         # invariant : jamais sur la parole
        assert not any(a < w[1] - MARGIN and w[0] + MARGIN < b for a, b in speech)
    assert plausible_segments([
        {"start": 113.7, "end": 115.3, "text": "Hé, c'est bientôt fini."},   # chuchotement : gardé
        {"start": 58.1, "end": 111.0, "text": "I've been alone for as long as I can remember. Oh, my God."},  # 53 s : ignoré
        {"start": 29.8, "end": 27.9, "text": ""},                          # vide : ignoré
    ]) == [[113.7 - WHISPER_PAD, 115.3 + WHISPER_PAD]]
    assert frame_times([0, 3]) == [0.375, 1.125, 1.875, 2.625]  # 4 images réparties sur la fenêtre
    assert len(frame_times([0, 17])) == MAX_IMAGES and len(frame_times([0, 1])) == 3
    fake = lambda text: (np.zeros(len(text) * 10), 100)           # voix factice : 10 caractères/s
    assert fit(["x" * 30, "x" * 20], 2.5, fake)[0] == "x" * 20    # 3 s ne tient pas, 2 s oui
    assert fit(["x" * 30], 2.8, fake)[3] == 3.0 / 2.8             # accélération ≤ 10 %
    assert fit(["x" * 40], 3.0, fake) is None and fit([], 3.0, fake) is None
    env = duck_envelope(1000, 100, [(2.0, 5.0)])
    assert env[350] == np.float32(DUCK) and env[0] == env[999] == 1 and DUCK < env[190] < 1
    assert parse_variants('Voici : {"variantes": ["Elle court.", "Elle court vers la porte.", "Elle court."]}') \
        == ["Elle court vers la porte.", "Elle court."]
    assert parse_variants("pas de JSON") == []
    assert VAGUE.search("Elle tient un objet sphérique et piquant.") and not VAGUE.search("Elle tient un fruit épineux.")
    girl, dragon = "la jeune fille aux cheveux roux", "le petit dragon"
    assert short_form(girl) == "la jeune fille" and short_form(dragon) == dragon
    assert lighten("La jeune fille aux cheveux roux grimpe.", [girl, dragon], {girl}, 4) == "Elle grimpe."
    assert lighten("Le petit dragon est blessé. La jeune fille aux cheveux roux regarde.", [girl, dragon], {girl}, 4) \
        == "Le petit dragon est blessé. Elle regarde."
    assert lighten("Le petit dragon regarde la jeune fille aux cheveux roux.", [girl, dragon], {girl, dragon}, 4) \
        == "Il regarde la jeune fille."
    assert lighten("La jeune fille aux cheveux roux grimpe.", [girl, dragon], {girl}, 20) == "La jeune fille aux cheveux roux grimpe."
    assert mentioned("Elle grimpe.", [girl, dragon], {girl, dragon}) == {girl}
    assert repeats("Elle tend sa main gantée vers le dragon blessé.",
                   "Accroupie sur les pavés, elle tend une main gantée vers le dragon blessé.")
    assert not repeats("Elle grimpe sur les façades.", "Dans une rue à colombages, elle contemple un fruit.")
    assert parse_variants('{"variantes": ["Bol.", "Elle tient un bol."]}') == ["Elle tient un bol."]  # un mot : refusé
    assert vtt([{"start": 3725.5, "end": 3727.25, "text": "Elle sourit."}]) \
        == "WEBVTT\n\n01:02:05.500 --> 01:02:07.250\nElle sourit.\n\n"
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description="Audesia P0 : audiodescription française d'un extrait vidéo.")
    p.add_argument("video", nargs="?")
    p.add_argument("--start", default="0", help="début de l'extrait (s ou mm:ss)")
    p.add_argument("--end", help="fin de l'extrait (par défaut : fin de la vidéo)")
    p.add_argument("--out", help="dossier de sortie (par défaut : out/<vidéo>_<début>-<fin>)")
    # 127.0.0.1 plutôt que localhost : sous Windows, localhost part d'abord en IPv6 et peut joindre un autre serveur.
    p.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="API compatible OpenAI (Ollama, llama.cpp, vLLM)")
    # 26B plutôt que 12B : sur Sintel, le 12B inventait un homme, un livre, une hache ; le 26B, non.
    # Sur 16 Go, Ollama en place une partie sur le CPU (environ 35 s par fenêtre).
    p.add_argument("--model", default="gemma4:26b-a4b-it-qat")
    p.add_argument("--cps", type=float, default=13.0, help="débit de la voix en caractères/s (mesuré : 13 pour Vivian)")
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="…-0.6B-CustomVoice si la VRAM manque")
    p.add_argument("--voice", default="Vivian", help="Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee")
    p.add_argument("--selftest", action="store_true", help="vérifie la logique de calage, sans GPU")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.video:
        p.error("indiquer la vidéo")

    out = Path(a.out or Path("out") / f"{Path(a.video).stem}_{a.start}-{a.end or 'fin'}".replace(":", "."))
    out.mkdir(parents=True, exist_ok=True)
    clip, a16, a48 = out / "clip.mp4", out / "audio16k.wav", out / "audio48k.wav"
    if not clip.exists():  # extrait réencodé en H.264/AAC : c'est aussi la version « sans AD »
        length = ["-t", secs(a.end) - secs(a.start)] if a.end else []
        step("extrait", lambda: ff("-ss", secs(a.start), "-i", a.video, *length, "-map", "0:v:0", "-map", "0:a:0",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", 18, "-c:a", "aac", "-b:a", "192k",
                                   clip))
    if not a48.exists():
        ff("-i", clip, "-ac", 1, "-ar", 16000, a16, "-ac", 2, "-ar", 48000, a48)

    segs = step("parole", lambda: detect_speech(a16), out / "segments.json")
    # Recalculés à chaque lancement : changer ces réglages ne refait pas l'ASR.
    segs["speech"] = merge(segs["speech"] + plausible_segments(segs["dialogues"]))
    segs["silences"] = silences(segs["speech"], segs["duration"])
    shots = step("plans", lambda: detect_shots(clip), out / "shots.json")
    descs = step("description", lambda: describe(clip, segs, shots, a, out), out / "descriptions.json")
    unload_ollama(a.base_url, a.model)
    placed, dropped = step("voix", lambda: voice(descs, segs, a, out))
    step("mixage", lambda: mix_and_export(placed, out))

    import torch
    usable = sum(e - s for s, e in segs["silences"])
    metrics = {
        "timings_s": TIMINGS,
        "duration_s": round(segs["duration"], 1),
        "usable_silence_s": round(usable, 1),
        "windows": len(descs),
        "placed": len(placed),
        "dropped": dropped,
        "sped_up": sum(p["speedup"] > 1 for p in placed),
        "coverage": round(sum(p["end"] - p["start"] for p in placed) / usable, 3) if usable else 0,
        "measured_chars_per_s": round(statistics.mean(p["chars_per_s"] for p in placed), 1) if placed else None,
        "peak_gpu_gib_this_process": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ad.json").write_text(json.dumps(placed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"Sans AD : {clip}\nAvec AD : {out / 'clip_ad.mp4'}\nMKV deux pistes : {out / 'clip_ad.mkv'}")


if __name__ == "__main__":
    main()
