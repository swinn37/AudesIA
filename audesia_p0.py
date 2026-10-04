#!/usr/bin/env python3
"""Audesia P0 : audiodescription française d'un extrait vidéo, en un seul fichier.

Chaîne : ffmpeg → Silero VAD + Whisper large-v3 → PySceneDetect → VLM (description)
→ rédacteur (3 variantes calées sur le silence) → Qwen3-TTS → mixage → exports.
VLM et rédacteur passent par une API compatible OpenAI : passer de la 5080 (Ollama) au GX10
(deux serveurs vLLM) ne demande que --profile large (configs/large.toml).

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
mixage sont refaits à chaque lancement ; les phrases déjà synthétisées sont en cache.

Relecture humaine (facultative) : ouvrir dans un navigateur relecture.html, écrite dans le
dossier de sortie. La page montre la vidéo avec ou sans audiodescription, fait écouter chaque
phrase et enregistre les textes corrigés dans corrections.json. Sans navigateur, relecture.json
donne l'horaire, la place disponible, le texte lu et le statut de chaque fenêtre, et l'on écrit
soi-même corrections.json, dans le même dossier :
  {"d_0004": "Elle regarde sous les débris.", "d_0007": ""}    (texte vide : fenêtre silencieuse)
Puis relancer la même commande : seules les phrases modifiées sont synthétisées, et tout est
remixé (1 min 15 s pour 4 phrases sur la 5080).
"""
import argparse
import base64
import contextlib
import difflib
import gc
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import zlib
from concurrent.futures import ThreadPoolExecutor
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
FRAME_STEP = 0.7    # s : écart visé entre deux images dans les plans courts (leur nombre maximal est dans le profil)
MAX_CAST = 10       # images (une par plan) pour le registre des personnages
DARK_LEVEL = 40     # luminance sous laquelle un pixel compte comme très sombre (0-255)
DARK_SHARE = 0.25   # contre-jour : plus de 25 % de pixels très sombres…
BRIGHT_LEVEL = 170  # …et plus de 5 % de pixels clairs (> BRIGHT_LEVEL) : le modèle reçoit aussi une copie éclaircie
BRIGHT_SHARE = 0.05  # (mesuré : débris à contre-jour 42-55 % sombres / 9-14 % clairs ; hutte sombre 90-97 % / 0 %)
GAMMA = 2.2         # éclaircissement de cette copie : à contre-jour, des débris passaient pour un « tissu noir »
VAGUE = re.compile(r"\b(objets?|sphères?|sphériques?|masses?|formes?|choses?|éléments?)\b", re.I)
MAX_SPEEDUP = 1.10
VOCAL_DB = 10.0     # dB au-dessus de la fuite de musique (90e centile hors parole) : vérité terrain de l'évaluation
BURST_DB = 8.0      # même critère sur la voix isolée par Demucs, pour le pipeline. Mesuré sur Sintel : à +8 dB,
                    # 2 descriptions sur 3 (VO) et 2 sur 5 (VF) qui touchaient un cri ou un souffle en sont protégées,
                    # pour 11 % et 4 % du silence utilisable ; à +10 dB, 2 sur 3 et 1 sur 5
MAX_CPS = 16.0      # car./s : au-delà, Qwen3-TTS a précipité la phrase et avalé un mot (mesuré : 19 à 20, contre 10 à 15)
TTS_TRIES = 3       # synthèses au plus par phrase ; si toutes sont précipitées, la plus lente est gardée
REMENTION_S = 15.0  # s : au-delà, un personnage est redésigné en entier plutôt que par « elle » ou « il »
REPEAT_RATIO = 0.6  # similarité au-delà de laquelle une description redit la précédente (redite mesurée : 0,75)
DUCK = 0.3          # gain de la bande-son sous la voix
RAMP = 0.25         # s : rampe d'atténuation
TTS_INSTRUCT = "Voix calme et neutre de narrateur, diction claire, débit régulier."
ASR_MODEL = "openai/whisper-large-v3"  # transcription des dialogues et retranscription de contrôle de la voix
PAGE = Path(__file__).with_name("relecture.html")  # modèle de la page de relecture
PAGE_DATA = '<script id="donnees" type="application/json">null</script>'  # remplacé par les données du run

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
VERIFY = """Tu vérifies, sur les images, la description d'un plan de film destinée à une audiodescription. Découpe \
la description en faits élémentaires : un personnage, une action, un objet, un lieu, un état, une couleur. Pour chaque \
fait, regarde les images : est-il clairement visible ? Un fait faux, douteux ou invisible n'est pas visible. N'ajoute \
aucun fait qui ne soit pas dans la description.
Réponds uniquement en JSON : {"faits": [{"fait": "...", "visible": true}]}"""
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
USAGE = {r: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0} for r in ("vlm", "writer")}  # metrics.json
MAX_TOKENS = 1500   # par réponse : une description fait 3 à 5 phrases, un registre quelques lignes de JSON
CONFIGS = Path(__file__).with_name("configs")  # profils small (RTX 5080) et large (GX10)


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


def plausible(d):
    """Segment Whisper crédible : débit plausible, sans boucle. Sur la musique, Whisper étire une phrase
    (15 mots sur 53 s) ou répète en boucle (« I'm sorry, I'm sorry… » sur 131 s de Sintel) ; un texte qui se
    compresse plus de 2,4 fois tourne en boucle, critère que Whisper applique lui-même."""
    text = d["text"].encode("utf-8")
    return (bool(text) and len(text) <= 2.4 * len(zlib.compress(text))
            and d["end"] - d["start"] <= WHISPER_S_PER_WORD * len(d["text"].split()) + 1.0)


def plausible_segments(dialogues):
    """Segments Whisper crédibles, élargis : ils rattrapent les chuchotements que la VAD manque."""
    return [[d["start"] - WHISPER_PAD, d["end"] + WHISPER_PAD] for d in dialogues if plausible(d)]


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


def frame_times(win, n_max):
    """De 3 à n_max images réparties sur la fenêtre, environ FRAME_STEP s d'écart dans les plans courts."""
    a, b = win
    n = min(n_max, max(3, round((b - a) / FRAME_STEP)))
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


def missing_words(expected, heard):
    """Mots de plus de 3 lettres du texte que la retranscription de la voix ne contient pas, même à peu près : les
    mots avalés. « À peu près » (similarité ≥ 0,75) tolère les finales muettes du français : ouverte, ouvertes."""
    def words(text):
        return re.findall(r"[a-z]+", unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode())
    got = words(heard)
    return [w for w in words(expected)
            if len(w) > 3 and not any(difflib.SequenceMatcher(None, w, g).ratio() >= 0.75 for g in got)]


def steady(text, generate, missing=lambda text, wav, sr: []):
    """Synthèse acceptable : refaite jusqu'à TTS_TRIES fois quand la voix précipite la phrase (plus de MAX_CPS
    caractères par seconde) ou avale des mots (absents de sa retranscription : « sur un toit » devenait « sur un C ») ;
    sinon la meilleure tentative, la moins incomplète puis la plus lente. Rend aussi le nombre d'essais."""
    best = key = None
    for tries in range(1, TTS_TRIES + 1):
        wav, sr = generate(text)
        rushed = len(text) * sr > MAX_CPS * len(wav)
        lost = [] if rushed else missing(text, wav, sr)  # une voix précipitée est refaite sans être retranscrite
        if best is None or (rushed, len(lost), -len(wav) / sr) < key:
            best, key = (wav, sr), (rushed, len(lost), -len(wav) / sr)
        if not rushed and not lost:
            break
        why = (f"précipitée ({len(text) * sr / max(len(wav), 1):.1f} car./s)" if rushed
               else f"incomplète (manque : {', '.join(lost)})")
        print(f"  voix {why}, essai {tries}/{TTS_TRIES} : {text}", flush=True)
    return (*best, tries)


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


def cut_hint(text, dur, avail):
    """Caractères à retirer d'un texte relu dont la voix dure dur s pour avail s disponibles (accélération comprise)."""
    return max(1, math.ceil(len(text) * (1 - avail * MAX_SPEEDUP / dur)))


def repeats(text, previous):
    """Vrai si text redit presque la description précédente (similarité des caractères ≥ REPEAT_RATIO)."""
    return difflib.SequenceMatcher(None, text.lower(), previous.lower()).ratio() >= REPEAT_RATIO


def free_span(win, speech):
    """Plus grand intervalle de la fenêtre sans parole. Une fenêtre décrite avant un changement de la détection (cris
    et souffles ajoutés, par exemple) peut en contenir : la description se cale alors dans la plus grande place libre."""
    a, b = win
    spans, t = [], a
    for s, e in merge([[max(s, a), min(e, b)] for s, e in speech if s < b and e > a]):
        if s > t:
            spans.append([t, s])
        t = max(t, e)
    if b > t:
        spans.append([t, b])
    return max(spans, key=lambda x: x[1] - x[0], default=[a, a])


def candidates(d, corrections, names, recent, gap, last_text):
    """Textes à essayer pour la fenêtre d, et s'ils viennent de la relecture. Le texte relu fait foi : ni allègement
    ni filtre des redites ; vide, il laisse la fenêtre silencieuse."""
    if d["id"] in corrections:
        text = corrections[d["id"]].strip()
        return ([text] if text else []), True
    # Fluidité : « elle » ou forme courte plutôt que la désignation complète répétée d'un plan à l'autre,
    # et pas de redite de la description qui vient d'être lue (le plan reste alors silencieux).
    variants = [lighten(v, names, recent, gap) for v in d["variants"]]
    if gap <= REMENTION_S:
        variants = [v for v in variants if not repeats(v, last_text)]
    return variants, False


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


def verified(raw):
    """Faits du JSON de vérification, et le texte de ceux qui sont visibles."""
    facts = [f for f in json_list(raw, "faits") if isinstance(f, dict) and isinstance(f.get("fait"), str) and f["fait"].strip()]
    return facts, [f["fait"].strip() for f in facts if f.get("visible") is True]


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
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, dtype=torch.float16, device="cuda:0")
    chunks = asr({"raw": audio, "sampling_rate": sr}, return_timestamps=True, chunk_length_s=30, batch_size=8,
                 generate_kwargs={"task": "transcribe"})["chunks"]
    del asr
    gc.collect()
    torch.cuda.empty_cache()  # libère la VRAM pour le VLM et la voix
    dialogues = [{"start": c["timestamp"][0], "end": c["timestamp"][1] or duration, "text": c["text"].strip()}
                 for c in chunks]
    return {"duration": duration, "speech": merge([[v["start"], v["end"]] for v in vad]), "dialogues": dialogues}


def energy_speech(audio, speech, db=VOCAL_DB, sr=16000):
    """Intervalles où l'énergie 300–3400 Hz d'une voix isolée dépasse de db la fuite de musique, mesurée au 90e
    centile hors parole : cris, souffles, chuchotements. Trames de 20 ms ; intervalles d'au moins 0,25 s. Sert au
    pipeline (voix isolée par Demucs) et à l'évaluation (résidu de la piste musique + effets)."""
    import numpy as np
    hop, win = 320, 512
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop] * np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / sr)
    band = 10 * np.log10((np.abs(np.fft.rfft(frames, axis=1)) ** 2)[:, (f >= 300) & (f <= 3400)].sum(axis=1) + 1e-12)
    t = np.arange(len(band)) * hop / sr
    calm = np.ones(len(t), dtype=bool)
    for s, e in speech:
        calm &= (t < s) | (t >= e)
    if not calm.any():
        return []
    on = band > np.percentile(band[calm], 90) + db
    return [iv for iv in merge([[x - 0.1, x + 0.12] for x in t[on].tolist()]) if iv[1] - iv[0] >= 0.25]


def separate_voice(wav48):
    """Voix isolée du mélange par Hybrid Demucs (torchaudio), en mono à 16 kHz."""
    import soundfile as sf
    import torch
    import torchaudio
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to("cuda").eval()
    mix, sr = sf.read(wav48, dtype="float32", always_2d=True)
    mix = torchaudio.functional.resample(torch.from_numpy(mix.T.copy()), sr, bundle.sample_rate).to("cuda")
    ref = mix.mean(0)
    mix = (mix - ref.mean()) / ref.std()
    n, seg, pad = mix.shape[1], 10 * bundle.sample_rate, bundle.sample_rate
    voice, vocals = torch.zeros(n, device="cuda"), model.sources.index("vocals")
    with torch.no_grad():  # tranches de 10 s avec 1 s de contexte jetée de chaque côté : pas de clic aux raccords
        for i in range(0, n, seg):
            a, b = max(0, i - pad), min(n, i + seg + pad)
            voice[i: min(n, i + seg)] = model(mix[None, :, a:b])[0, vocals].mean(0)[i - a: i - a + min(seg, n - i)]
    voice16 = torchaudio.functional.resample(voice * ref.std() + ref.mean(), bundle.sample_rate, 16000).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return voice16


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


def load_profile(name):
    """Profil small, large ou chemin d'un fichier TOML : serveur, modèle et extra_body de chaque rôle (vlm décrit,
    writer révise et rédige), images par fenêtre et requêtes simultanées."""
    import tomllib
    path = Path(name) if name.endswith(".toml") else CONFIGS / f"{name}.toml"
    try:
        profile = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"profil {path} illisible : {e}")
    for role in ("vlm", "writer"):
        if not {"base_url", "model"} <= profile.get(role, {}).keys():
            sys.exit(f"profil {path} : la section [{role}] doit donner base_url et model")
        # Sous Docker (scripts/docker.sh), les serveurs se joignent par leur nom sur le réseau interne.
        profile[role]["base_url"] = os.environ.get(f"AUDESIA_{role.upper()}_URL", profile[role]["base_url"])
    profile["name"] = path.stem
    profile.setdefault("run", {})
    return profile


def chat(llm, cfg, system, content, json_mode=False):
    """Une requête au serveur d'un rôle, à température 0, avec l'extra_body du profil. Rend le texte et l'usage."""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    # extra_body coupe la « réflexion » : sans elle, Gemma 4 sous Ollama réfléchit 2 à 3 min et rend un contenu vide.
    # max_tokens : à température 0, une boucle de répétition irait sinon jusqu'au bout du contexte.
    r = llm.chat.completions.create(model=cfg["model"], temperature=0, max_tokens=MAX_TOKENS,
                                    extra_body=cfg.get("extra_body", {}),
                                    messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                                    **extra)
    return (r.choices[0].message.content or "").strip(), r.usage


def describe(clip, segs, shots, profile, cps, out):
    """1. Par fenêtre, en parallèle : description de la suite d'images seule (vlm), puis détails agrandis si un objet
    reste vague. 2. Sur l'ensemble : registre des personnages (vlm). 3. Par fenêtre, dans l'ordre : révision sur les
    mêmes images avec le registre et les plans voisins, puis 3 variantes calées (writer)."""
    # ponytail: révision et rédaction en série, car chacune reprend la précédente ; le parallèle porte sur les
    # descriptions, les requêtes les plus lourdes (jusqu'à 16 images), et sur plusieurs vidéos à la fois.
    from openai import OpenAI
    clients = {role: OpenAI(base_url=profile[role]["base_url"], api_key="local") for role in ("vlm", "writer")}
    lock = threading.Lock()

    def ask(role, system, content, json_mode=False):
        text, usage = chat(clients[role], profile[role], system, content, json_mode)
        with lock:  # jetons envoyés et générés : débit du rapport, et preuve que les images comptent bien
            USAGE[role]["requests"] += 1
            USAGE[role]["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            USAGE[role]["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return text

    n_max, parallel = profile["run"].get("max_images", 4), profile["run"].get("parallel", 1)
    (out / "frames").mkdir(exist_ok=True)
    wins = [w for sil in segs["silences"] for w in windows(sil, shots)]

    def look(k, w):
        times, images, bright = frame_times(w, n_max), [], []
        for t in times:
            f = out / "frames" / f"{t:08.2f}.jpg"
            ff("-ss", t, "-i", clip, "-frames:v", 1, "-vf", "scale=1024:-2", "-q:v", 3, f)
            images.append(as_image(f))
            if backlit(f):
                bright.append(as_image(brighten(f)))
        extra = [{"type": "text", "text": "Versions éclaircies des images sombres :"}, *bright] if bright else []
        # Images seules : en contexte, les répliques (« Cette lame… ») et les descriptions précédentes
        # amorçaient des inventions qui se propageaient d'une fenêtre à l'autre.
        raw = ask("vlm", DESCRIBE, [{"type": "text", "text": "Décris cette suite d'images."}, *images, *extra])
        print(f"  {k + 1}/{len(wins)}  {w[0]:6.1f}–{w[1]:6.1f} s  décrit ({len(images)} images, "
              f"{len(bright)} éclaircies)", flush=True)
        return {"id": f"d_{k:04d}", "window": w, "frames": times, "brightened": len(bright),
                "raw_description": raw, "images": images, "extra": extra}

    def zoom(x):
        # Objet nommé vaguement : question directe sur des détails agrandis de l'image centrale, sans la
        # description (elle ancrait la réponse sur « objet sphérique »). Envoyés en vrac dans la révision,
        # ces détails ne suffisaient pas à la faire sortir de « objet ».
        if not VAGUE.search(x["raw_description"]):
            return None
        mid = x["frames"][len(x["frames"]) // 2]
        zoomed = [as_image(out / "frames" / f"{mid:08.2f}.jpg"), *[as_image(p) for p in tiles(clip, mid, out / "frames")]]
        return ask("vlm", ZOOM, [{"type": "text", "text": "Plan et détails agrandis :"}, *zoomed])

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        items = list(pool.map(look, range(len(wins)), wins))
        for x, z in zip(items, list(pool.map(zoom, items))):
            x["zoom"] = z

    # Une image par plan, pour que la même personne garde la même désignation tout au long de l'extrait.
    picks = items if len(items) <= MAX_CAST else [items[int(i * len(items) / MAX_CAST)] for i in range(MAX_CAST)]
    cast = [c for c in json_list(ask("vlm", CAST, [{"type": "text", "text": "Images, dans l'ordre :"},
                                                  *[x["images"][len(x["images"]) // 2] for x in picks]],
                                     json_mode=True), "personnages") if isinstance(c, dict)]
    registry = "\n".join(f"- {c.get('designation', '')} : {c.get('traits', '')}" for c in cast) or "aucun"
    (out / "personnages.json").write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  registre : {[c.get('designation') for c in cast]}", flush=True)

    said, previous = [], "aucun"
    for k, x in enumerate(items):
        w = x["window"]
        visuals = x.pop("images") + x.pop("extra")  # pas de base64 dans le cache JSON
        note = (f"\nPrécision sur l'objet, d'après des détails agrandis : {x['zoom']}"
                if x["zoom"] and "inconnu" not in x["zoom"].lower() else "")
        following = items[k + 1]["raw_description"] if k + 1 < len(items) else "aucun"
        # Révision sur les images, avec le registre et les plans voisins : désignations stables, actions rattachées au
        # bon personnage, rien de ce que les images ne montrent pas. Sur le GX10, le rédacteur est un second modèle,
        # d'une autre famille : il relit sur les images ce que le premier a décrit.
        desc = ask("writer", REVISE, [{"type": "text", "text": (
            f"Registre des personnages :\n{registry}\n\nPlan précédent : {previous}\n"
            f"Ce plan : {x['raw_description']}{note}\nPlan suivant : {following}")}, *visuals]) or x["raw_description"]
        previous = desc
        heard = " / ".join(d["text"] for d in segs["dialogues"] if d["end"] <= w[1] and plausible(d))[-600:] or "aucun"
        context = (f"Personnages :\n{registry}\nDéjà décrit : {' '.join(said[-3:]) or 'rien'}\n"
                   f"Dialogues entendus jusqu'ici : {heard}")
        b = budget(w, cps)
        limits = "\n".join(f"Variante {i + 1} : au plus {n} caractères (environ {max(n // 6, 1)} mots)."
                           for i, n in enumerate((b, b * 2 // 3, b // 2)))
        source, x["facts"] = desc, None
        if profile["run"].get("verify"):
            # Vérification fait par fait sur les images, par le rédacteur (sur le GX10, d'une autre famille que le modèle
            # de vision) : seuls les faits visibles sont rédigés ; s'ils sont tous écartés, le plan reste sans description.
            x["facts"], kept = verified(ask("writer", VERIFY, [{"type": "text", "text": f"Description : {desc}"}, *visuals],
                                            json_mode=True))
            if x["facts"]:  # sans fait lisible, la description révisée reste
                source = "Faits vérifiés sur les images : " + " ; ".join(kept) if kept else ""
        variants = parse_variants(ask("writer", WRITE, f"Description : {source}\n{context}\n{limits}", json_mode=True)
                                  ) if source else []
        x.update(description=desc, budget_chars=b, variants=variants)
        said += variants[:1]
        print(f"  {w[0]:6.1f}–{w[1]:6.1f} s  {'[zoom : ' + x['zoom'][:60] + '] ' if x['zoom'] else ''}{variants[:1]}",
              flush=True)
    return items


def outbound():
    """Vrai si une connexion sortante aboutit. Sous Docker, le réseau interne la refuse : metrics.json garde ainsi
    la preuve que le traitement est resté local."""
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


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
    """Synthétise la variante la plus longue qui tient dans chaque fenêtre, la place sur une piste et écrit la fiche
    de relecture. Un texte de corrections.json remplace les variantes de sa fenêtre."""
    import numpy as np
    import soundfile as sf

    cache, tts, asr, retries = out / "tts" / "cache", None, None, 0
    cache.mkdir(parents=True, exist_ok=True)

    def generate(text):
        nonlocal tts
        if tts is None:  # chargé seulement s'il manque une phrase au cache
            from types import SimpleNamespace
            import huggingface_hub
            import torch
            from qwen_tts import Qwen3TTSModel
            # transformers 4.57 demande à l'API du Hub (model_info), à chaque chargement du tokenizer de qwen-tts, si le
            # modèle dérive d'un Mistral : un appel réseau caché, que le réseau Docker sans sortie a révélé. On répond
            # localement ce que dirait le Hub (non), ce qui laisse la tokenisation inchangée.
            # ponytail: rustine sur cette version ; le run Docker hors ligne dira si une mise à jour la rend inutile.
            huggingface_hub.model_info = lambda *args, **kwargs: SimpleNamespace(tags=[])
            tts = Qwen3TTSModel.from_pretrained(a.tts_model, device_map="cuda:0", dtype=torch.bfloat16)
        wavs, sr = tts.generate_custom_voice(text=text, language="French", speaker=a.voice, instruct=TTS_INSTRUCT)
        return np.asarray(wavs[0], dtype=np.float32), sr

    def heard_missing(text, wav, sr):
        # Retranscription de contrôle par Whisper, chargé seulement s'il y a une phrase à synthétiser. Voix
        # rééchantillonnée à 16 kHz : à 24 ou 48 kHz, le pipeline Whisper rend du charabia.
        nonlocal asr
        import torch
        import torchaudio
        if asr is None:
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, dtype=torch.float16, device="cuda:0")
        audio = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
        heard = asr({"raw": audio, "sampling_rate": 16000}, generate_kwargs={"language": "french", "task": "transcribe"})
        return missing_words(text, heard["text"])

    def synth(text):
        # Cache par phrase : après une correction, seules les phrases nouvelles sont synthétisées.
        nonlocal retries
        f = cache / (hashlib.sha1(f"{a.tts_model}|{a.voice}|{TTS_INSTRUCT}|{text}".encode()).hexdigest()[:16] + ".wav")
        if f.exists():
            return sf.read(f, dtype="float32")
        wav, sr, tries = steady(text, generate, heard_missing)
        retries += tries - 1
        sf.write(f, wav, sr, subtype="FLOAT")
        return wav, sr

    fixes = out / "corrections.json"
    try:  # utf-8-sig : le Bloc-notes peut enregistrer avec un BOM
        corrections = json.loads(fixes.read_text(encoding="utf-8-sig")) if fixes.exists() else {}
    except ValueError as e:
        sys.exit(f"{fixes} illisible : {e}")
    if not isinstance(corrections, dict) or not all(isinstance(v, str) for v in corrections.values()):
        sys.exit(f'{fixes} : attendu {{"d_0004": "texte relu", "d_0007": ""}} (texte vide : fenêtre silencieuse)')
    unknown = sorted(set(corrections) - {d["id"] for d in descs})
    if unknown:
        print(f"  corrections ignorées, identifiants inconnus : {unknown}", flush=True)

    info = sf.info(out / "audio48k.wav")
    sr, n = info.samplerate, info.frames
    track = np.zeros(n, dtype=np.float32)
    registry = out / "personnages.json"
    names = [c["designation"].strip() for c in (json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else [])
             if isinstance(c, dict) and c.get("designation")]
    review, placed, dropped, recent, last_end, last_text = [], [], [], set(), -REMENTION_S, ""
    for d in descs:
        w0, w1 = free_span(d["window"], segs["speech"])
        start, avail = w0 + MARGIN, w1 - w0 - 2 * MARGIN
        variants, reviewed = candidates(d, corrections, names, recent, start - last_end, last_text)
        got = fit(variants, avail, synth) if avail > 0 else None
        horaire = f"{int(w0 // 60)}:{w0 % 60:04.1f} – {int(w1 // 60)}:{w1 % 60:04.1f}".replace(".", ",")
        row = {"id": d["id"], "horaire": horaire, "debut": round(w0, 2), "fin": round(w1, 2), "place_s": round(avail, 2),
               "place_caracteres": budget([w0, w1], a.cps), "statut": "relu" if reviewed else "automatique",
               "texte": got[0] if got else None, "voix_s": None, "acceleration": None,
               "description": d.get("description", ""), "variantes": d["variants"],
               "faits_ecartes": [f["fait"] for f in d.get("facts") or [] if f.get("visible") is not True]}
        review.append(row)
        if not got:
            dropped.append(d["id"])
            if not reviewed:
                row["statut"] = "sans description : aucune variante ne tient, ou redite de la précédente"
            elif variants:
                wav, tsr = synth(variants[0])  # déjà en cache : fit() vient de la synthétiser
                row["statut"] = f"relu, trop long : retirer au moins {cut_hint(variants[0], len(wav) / tsr, avail)} caractères"
                print(f"  {d['id']} {row['statut']}", flush=True)
            else:
                row["statut"] = "supprimé à la relecture"
            continue
        text, wav, tsr, speed = got
        raw, final = out / "tts" / f"{d['id']}_raw.wav", out / "tts" / f"{d['id']}.wav"
        sf.write(raw, wav, tsr)
        ff("-i", raw, "-af", f"atempo={speed}", "-ar", sr, "-ac", 1, final)  # accélération ≤ 10 % + 48 kHz
        clip_audio = sf.read(final, dtype="float32")[0]
        i = int(start * sr)
        clip_audio = clip_audio[: n - i]
        track[i: i + len(clip_audio)] += clip_audio
        row.update(voix_s=round(len(clip_audio) / sr, 2), acceleration=round(speed, 3))
        placed.append({"id": d["id"], "start": round(start, 3), "end": round(start + len(clip_audio) / sr, 3),
                       "text": text, "speedup": round(speed, 3), "chars_per_s": round(len(text) * tsr / len(wav), 1),
                       "reviewed": reviewed})
        recent, last_end, last_text = mentioned(text, names, recent), placed[-1]["end"], text
    for p in placed:  # invariant : aucune description ne chevauche la parole détectée
        assert not any(s < p["end"] and p["start"] < e for s, e in segs["speech"]), p
    sf.write(out / "ad_voice.wav", track, sr)
    (out / "relecture.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    return placed, dropped, retries


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


def review_page(out, a, metrics):
    """relecture.html : la page de relecture remplie avec les données du run, lisible hors ligne à côté des vidéos."""
    fixes = out / "corrections.json"
    data = {"titre": Path(a.video).stem, "cps": metrics["measured_chars_per_s"] or a.cps, "accel_max": MAX_SPEEDUP,
            "couverture": metrics["coverage"],
            "commande": f'python audesia_p0.py "{Path(a.video).as_posix()}" --start {a.start}'
                        + (f" --end {a.end}" if a.end else "") + f' --out "{out.as_posix()}"',
            "corrections": json.loads(fixes.read_text(encoding="utf-8-sig")) if fixes.exists() else {},
            "lignes": json.loads((out / "relecture.json").read_text(encoding="utf-8"))}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # un « </script> » dans un texte fermerait la balise
    (out / "relecture.html").write_text(PAGE.read_text(encoding="utf-8").replace(PAGE_DATA, PAGE_DATA.replace("null", blob)),
                                        encoding="utf-8")


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
        {"start": 0.0, "end": 131.3, "text": "IAEA " + "I'm sorry, " * 135},  # boucle sur la musique : ignorée
    ]) == [[113.7 - WHISPER_PAD, 115.3 + WHISPER_PAD]]
    assert frame_times([0, 3], 4) == [0.375, 1.125, 1.875, 2.625]  # 4 images réparties sur la fenêtre
    assert len(frame_times([0, 17], 4)) == 4 and len(frame_times([0, 17], 8)) == 8 and len(frame_times([0, 1], 8)) == 3
    for path in sorted(CONFIGS.glob("*.toml")):                  # profils livrés : lisibles et complets
        prof = load_profile(str(path))
        assert prof["run"]["parallel"] >= 1 and prof["run"]["max_images"] >= 3, path.name
    fake = lambda text: (np.zeros(len(text) * 10), 100)           # voix factice : 10 caractères/s
    assert fit(["x" * 30, "x" * 20], 2.5, fake)[0] == "x" * 20    # 3 s ne tient pas, 2 s oui
    assert fit(["x" * 30], 2.8, fake)[3] == 3.0 / 2.8             # accélération ≤ 10 %
    assert fit(["x" * 40], 3.0, fake) is None and fit([], 3.0, fake) is None
    gen = lambda text: (np.zeros(int(next(durations) * 100)), 100)
    with contextlib.redirect_stdout(io.StringIO()):               # sans l'affichage des essais rejetés
        durations = iter([2.4, 3.2])                              # 46 caractères : 19,2 puis 14,4 car./s
        wav, _, tries = steady("x" * 46, gen)
        assert (len(wav), tries) == (320, 2)                      # phrase précipitée refaite
        durations = iter([2.4, 2.5, 2.3])                         # toujours précipitée : la plus lente est gardée
        wav, _, tries = steady("x" * 46, gen)
        assert (len(wav), tries) == (250, 3)
        durations, lost = iter([3.0, 3.1]), iter([["toit"], []])  # débit normal, mais un mot avalé au 1er essai
        wav, _, tries = steady("x" * 46, gen, lambda text, w, sr: next(lost))
        assert (len(wav), tries) == (310, 2)
        durations, lost = iter([3.0, 3.1, 3.2]), iter([["a", "b"], ["a"], ["a", "b"]])  # la moins incomplète
        wav, _, tries = steady("x" * 46, gen, lambda text, w, sr: next(lost))
        assert (len(wav), tries) == (310, 3)
    assert missing_words("Elle grimpe sur un toit, un couteau à la main.",
                         "Elle grimpe sur un C, un couteau à la main.") == ["toit"]  # mot avalé
    assert missing_words("Le dragon gît, les ailes ouvertes.", "Le dragon gît, les ailes ouverte.") == []  # finale muette
    assert free_span([10, 20], [[0, 9]]) == [10, 20]                  # fenêtre libre
    assert free_span([10, 20], [[12, 13], [18, 30]]) == [13, 18]      # éclat ajouté : la plus grande place libre
    assert free_span([10, 20], [[5, 25]]) == [10, 10]                 # plus de place
    rng = np.random.default_rng(0)                    # 10 s de bruit de fond et un « cri » rare : 0,4 s à 1 kHz
    sound = rng.normal(0, 0.001, 16000 * 10)
    sound[80000:86400] += 0.1 * np.sin(2 * np.pi * 1000 * np.arange(6400) / 16000)
    bursts = energy_speech(sound, [])
    assert len(bursts) == 1 and 4.8 < bursts[0][0] < 5.0 and 5.4 < bursts[0][1] < 5.65, bursts
    assert missing_words("SINTEL s'affiche.", "Sintel s'affiche.") == []
    env = duck_envelope(1000, 100, [(2.0, 5.0)])
    assert env[350] == np.float32(DUCK) and env[0] == env[999] == 1 and DUCK < env[190] < 1
    assert parse_variants('Voici : {"variantes": ["Elle court.", "Elle court vers la porte.", "Elle court."]}') \
        == ["Elle court vers la porte.", "Elle court."]
    assert parse_variants("pas de JSON") == []
    facts, kept = verified('{"faits": [{"fait": "Elle grimpe.", "visible": true}, {"fait": "Un tissu noir.", '
                           '"visible": false}, {"fait": "", "visible": true}]}')
    assert len(facts) == 2 and kept == ["Elle grimpe."]  # fait vide ignoré, fait invisible écarté
    assert verified("pas de JSON") == ([], [])
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
    assert cut_hint("x" * 50, 6.0, 4.0) == 14  # 6 s de voix pour 4,4 s au plus (4 s + 10 %) : retirer 27 %
    d = {"id": "d_0001", "variants": ["La jeune fille aux cheveux roux grimpe."]}
    assert candidates(d, {}, [girl], {girl}, 4, "") == (["Elle grimpe."], False)
    assert candidates(d, {}, [girl], {girl}, 4, "Elle grimpe.") == ([], False)       # redite écartée
    assert candidates(d, {"d_0001": " Elle fouille sous les débris. "}, [girl], {girl}, 4, "Elle grimpe.") \
        == (["Elle fouille sous les débris."], True)                                 # le texte relu fait foi
    assert candidates(d, {"d_0001": ""}, [girl], {girl}, 4, "") == ([], True)        # vide : fenêtre silencieuse
    assert parse_variants('{"variantes": ["Bol.", "Elle tient un bol."]}') == ["Elle tient un bol."]  # un mot : refusé
    assert vtt([{"start": 3725.5, "end": 3727.25, "text": "Elle sourit."}]) \
        == "WEBVTT\n\n01:02:05.500 --> 01:02:07.250\nElle sourit.\n\n"
    assert not PAGE.exists() or PAGE.read_text(encoding="utf-8").count(PAGE_DATA) == 1  # emplacement des données
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description="Audesia P0 : audiodescription française d'un extrait vidéo.")
    p.add_argument("video", nargs="?")
    p.add_argument("--start", default="0", help="début de l'extrait (s ou mm:ss)")
    p.add_argument("--end", help="fin de l'extrait (par défaut : fin de la vidéo)")
    p.add_argument("--out", help="dossier de sortie (par défaut : out/<vidéo>_<début>-<fin>)")
    p.add_argument("--profile", default="small",
                   help="small (RTX 5080, Ollama), large (GX10, deux serveurs vLLM) ou chemin d'un fichier TOML")
    p.add_argument("--cps", type=float, default=13.0, help="débit de la voix en caractères/s (mesuré : 13 pour Vivian)")
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="…-0.6B-CustomVoice si la VRAM manque")
    p.add_argument("--voice", default="Vivian", help="Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee")
    p.add_argument("--precompute", action="store_true",
                   help="extrait, parole et plans seulement : à calculer sur la 5080, le GX10 n'a plus qu'à décrire")
    p.add_argument("--selftest", action="store_true", help="vérifie la logique de calage, sans GPU")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.video:
        p.error("indiquer la vidéo")
    profile = load_profile(a.profile)  # avant les étapes longues : un profil invalide arrête tout de suite

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
    # Cris, souffles, chuchotements : énergie de la voix isolée par Demucs, au-delà de la parole de la VAD.
    vocal = step("voix isolée", lambda: energy_speech(separate_voice(a48), segs["speech"], db=BURST_DB),
                 out / "vocal.json")
    # Recalculés à chaque lancement : changer ces réglages ne refait ni l'ASR ni la séparation.
    segs["speech"] = merge(segs["speech"] + plausible_segments(segs["dialogues"]) + vocal)
    segs["silences"] = silences(segs["speech"], segs["duration"])
    shots = step("plans", lambda: detect_shots(clip), out / "shots.json")
    if a.precompute:
        print(f"Précalcul prêt : {out}")
        return
    descs = step("description", lambda: describe(clip, segs, shots, profile, a.cps, out), out / "descriptions.json")
    for base_url, model in {(profile[r]["base_url"], profile[r]["model"]) for r in ("vlm", "writer")}:
        unload_ollama(base_url, model)
    placed, dropped, retries = step("voix", lambda: voice(descs, segs, a, out))
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
        "reviewed": sum(p["reviewed"] for p in placed),
        "tts_retries": retries,  # synthèses refaites : voix précipitée ou mot manquant à la retranscription
        "facts_checked": sum(len(d.get("facts") or []) for d in descs),  # vérification fait par fait (profil)
        "facts_rejected": sum(f.get("visible") is not True for d in descs for f in d.get("facts") or []),
        "profile": profile["name"],
        "models": {r: profile[r]["model"] for r in ("vlm", "writer")},
        "parallel_requests": profile["run"].get("parallel", 1),
        "llm_usage": USAGE,  # requêtes et jetons par rôle ; 0 si les descriptions venaient du cache
        "outbound_network": outbound(),  # False : aucun accès sortant possible pendant le traitement
        "coverage": round(sum(p["end"] - p["start"] for p in placed) / usable, 3) if usable else 0,
        "measured_chars_per_s": round(statistics.mean(p["chars_per_s"] for p in placed), 1) if placed else None,
        "peak_gpu_gib_this_process": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ad.json").write_text(json.dumps(placed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    if PAGE.exists():  # absente si audesia_p0.py a été copié seul
        review_page(out, a, metrics)
    print(f"Sans AD : {clip}\nAvec AD : {out / 'clip_ad.mp4'}\nMKV deux pistes : {out / 'clip_ad.mkv'}")
    print(f"Relecture (facultative) : ouvrir {out / 'relecture.html'} dans un navigateur, ou écrire les textes à "
          f"changer dans {out / 'corrections.json'}, puis relancer la même commande")


if __name__ == "__main__":
    main()
