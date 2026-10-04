#!/usr/bin/env python3
"""Taux de descriptions sans chevauchement des dialogues réels (AUDESIA.md, §7).

Vérité terrain : mixage original − piste musique + effets officielle, calée par corrélation croisée
et ajustée en gain. Compte comme parole ce que Silero VAD détecte dans ce résidu, plus toute énergie
de la bande vocale (300–3400 Hz) à plus de VOCAL_DB au-dessus de la fuite de musique : ce second
critère attrape les chuchotements et les vocalises, au prix de quelques bruitages mal annulés.

Usage :
  python eval/overlap.py out/Sintel.2010.1080p_1.35-3.35 --video out/Sintel.2010.1080p.mkv \
      --me out/sintel-m+e-st.flac --start 1:35 --end 3:35
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audesia_p0 import energy_speech, merge, plausible_segments, secs  # noqa: E402

SR = 16000
SEARCH = 2.0      # s : décalage maximal cherché entre le film et la piste musique + effets


def pcm(path, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32).copy()


def residual(film, me, pad=0.0):
    """Cale me sur film (me lu avec SEARCH s de marge de chaque côté, dont pad s de zéros quand la marge tombe avant le
    début de la piste), ajuste le gain, renvoie film − g·me, le décalage, le gain et les échantillons du film que la
    piste couvre : elle peut finir avant le film (générique ajouté au doublage)."""
    size = 2 * len(me)
    corr = np.fft.irfft(np.fft.rfft(me, size) * np.conj(np.fft.rfft(film, size)))[: int(2 * SEARCH * SR) + 1]
    lag = int(np.argmax(np.abs(corr)))
    covered = max(0, int(pad * SR) - lag), min(len(film), len(me) - lag)
    me = np.pad(me, (0, max(0, lag + len(film) - len(me))))[lag: lag + len(film)]
    g = float(np.dot(film, me) / np.dot(me, me))
    return film - g * me, lag / SR - SEARCH, g, covered


def overlap(a, b):
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def main():
    p = argparse.ArgumentParser(description="Chevauchement des descriptions avec les dialogues réels.")
    p.add_argument("out", help="dossier de sortie d'audesia_p0.py (ad.json, segments.json)")
    p.add_argument("--video", required=True, help="film original (même fichier que pour audesia_p0.py)")
    p.add_argument("--me", required=True, help="piste musique + effets sans dialogues")
    p.add_argument("--start", default="0")
    p.add_argument("--end", required=True)
    a = p.parse_args()

    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    out, start = Path(a.out), secs(a.start)
    dur = secs(a.end) - start
    film = pcm(a.video, start, dur)
    pad = max(0.0, SEARCH - start)  # au début du film, la marge de recherche tombe avant la piste : des zéros
    me = np.concatenate([np.zeros(int(pad * SR), dtype=np.float32), pcm(a.me, start - SEARCH + pad, dur + 2 * SEARCH - pad)])
    res, lag, gain, (c0, c1) = residual(film, me, pad)
    # Vérité terrain sur la seule partie couverte par la piste : ailleurs, le résidu est le film, musique comprise.
    part, t0, t1 = res[c0:c1], c0 / SR, c1 / SR
    vad = merge([[t["start"] + t0, t["end"] + t0] for t in get_speech_timestamps(
        torch.from_numpy(part), load_silero_vad(), sampling_rate=SR, return_seconds=True)])
    truth = merge(vad + [[s + t0, e + t0] for s, e in energy_speech(part, [[s - t0, e - t0] for s, e in vad])])

    calm = np.zeros(len(film), dtype=bool)
    calm[c0:c1] = True
    for s, e in truth:
        calm[int(s * SR): int(e * SR)] = False
    attenuation = 10 * np.log10(np.mean(film[calm] ** 2) / np.mean(res[calm] ** 2))

    everything = json.loads((out / "ad.json").read_text(encoding="utf-8"))
    placed = [d for d in everything if t0 <= d["start"] and d["end"] <= t1]  # mesurables contre la vérité terrain
    per = [{"id": d["id"], "start": d["start"], "end": d["end"], "text": d["text"],
            "overlap_s": round(overlap([[d["start"], d["end"]]], truth), 2),
            "overlap_words_s": round(overlap([[d["start"], d["end"]]], vad), 2)} for d in placed]
    clean = sum(d["overlap_s"] == 0 for d in per)
    clean_words = sum(d["overlap_words_s"] == 0 for d in per)
    # La parole que le pipeline évite : VAD, segments Whisper crédibles et éclats de la voix isolée (vocal.json).
    segs = json.loads((out / "segments.json").read_text(encoding="utf-8"))
    vocal = json.loads((out / "vocal.json").read_text(encoding="utf-8")) if (out / "vocal.json").exists() else []
    detected = merge(segs["speech"] + plausible_segments(segs["dialogues"]) + vocal)
    truth_s = sum(e - s for s, e in truth)
    result = {
        "lag_s": round(lag, 4), "gain": round(gain, 3), "music_attenuation_db": round(float(attenuation), 1),
        "truth_span_s": [round(t0, 2), round(t1, 2)], "outside_truth": len(everything) - len(placed),
        "truth_speech_s": round(truth_s, 1), "truth": [[round(s, 2), round(e, 2)] for s, e in truth],
        "descriptions": len(per), "clean": clean, "clean_rate": round(clean / len(per), 3) if per else None,
        "clean_words": clean_words, "clean_words_rate": round(clean_words / len(per), 3) if per else None,
        "max_overlap_s": max((d["overlap_s"] for d in per), default=0),
        "detector_recall": round(overlap(detected, truth) / truth_s, 3) if truth_s else None,
        "per_description": per,
    }
    (out / "overlap.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for d in per:
        print(f"{d['start']:6.1f}–{d['end']:6.1f}  {'ok' if not d['overlap_s'] else 'CHEVAUCHE ' + str(d['overlap_s']) + ' s':16} {d['text']}")
    print(f"\nCalage {lag:+.4f} s, gain {gain:.3f}, musique atténuée de {attenuation:.1f} dB dans le résidu")
    if result["outside_truth"]:
        print(f"Piste musique + effets de {t0:.1f} à {t1:.1f} s : {result['outside_truth']} descriptions hors de cette "
              f"partie, non mesurées")
    print(f"Parole réelle : {truth_s:.1f} s ; parole détectée par audesia_p0 qui la couvre : {result['detector_recall']:.0%}")
    print(f"Sans chevauchement des paroles (Silero sur le résidu) : {clean_words}/{len(per)} ({result['clean_words_rate']:.0%})")
    print(f"Sans chevauchement d'aucune voix (paroles + énergie vocale) : {clean}/{len(per)} ({result['clean_rate']:.0%}), "
          f"chevauchement maximal {result['max_overlap_s']} s")


if __name__ == "__main__":
    main()
