# Résultats archivés

Run de référence du profil `small` sur tout le corpus : 6 vidéos, 74 min, du 4 au 5 octobre 2026, sur une RTX 5080 (16 Go), sous Windows, sans Docker.

- **Vision et rédaction :** Gemma 4 26B-A4B en 4 bits (`gemma4:26b-a4b-it-qat`, Ollama), en partie sur le CPU.
- **Parole :** Silero VAD, Whisper large-v3 et voix isolée par Hybrid Demucs. **Voix :** Qwen3-TTS 1.7B.
- **Juge :** Qwen3.6 35B-A3B sous Ollama, absent de la chaîne `small`, en notation seule (`configs/judge-5080.toml`).

Les chiffres sont dans [rapport.md](rapport.md) ; le [README du dépôt](../README.md#résultats) en tire les constats.

| Fichier | Contenu |
| --- | --- |
| `rapport.md` | Rapport chiffré (`eval/report.py results`) : un tableau par vidéo, puis les notes du juge |
| `juge_small.md`, `juge_small.json` | Notes du juge, description par description, avec sa raison |
| `hallucinations_small.json` | Tirage des 100 silences à vérifier à la main ; il sert à compter les réponses |
| `run_small.log`, `juge_small.log` | Journaux du run et du juge |
| `small/<vidéo>/metrics.json` | Temps par étape, mémoire, requêtes et jetons, accès réseau |
| `small/<vidéo>/overlap.json` | Chevauchement contre la vérité terrain (Sintel VO et VF, Tears of Steel) |
| `small/<vidéo>/descriptions.json` | Descriptions brutes et révisées, détails agrandis, variantes de chaque fenêtre |
| `small/<vidéo>/ad.json`, `ad.vtt` | Descriptions placées, avec leurs horaires |
| `small/<vidéo>/relecture.json` | Fiche de relecture : place disponible, durée de la voix, statut |
| `small/<vidéo>/personnages.json` | Registre des personnages |

Aucune vidéo, aucun son ni aucune image n'est archivé. Pour tout refaire, après `python corpus/download.py` :

```bash
python eval/run_corpus.py --precompute
python eval/run_corpus.py --profile small
python eval/judge.py out/corpus/small --judge configs/judge-5080.toml
python eval/hallucination_sample.py out/corpus/small
python eval/archive.py small
```

Le journal du run garde deux erreurs corrigées en cours de route :
- la mesure du chevauchement plantait sur un film entier (commit `c16d2fa`) ;
- des fenêtres étaient placées après la fin de l'image de la VF de Sintel (commit `fc0e8a9`). Sintel VF a été refaite ensuite.

La colonne « Hors ligne » du rapport vaut « non » : ce run tournait hors du réseau Docker sans sortie. La preuve hors ligne vient du run sous Docker décrit dans le README du dépôt.

## Licences

Les descriptions sont des œuvres dérivées des films et suivent leur licence.

| Vidéo | Descriptions |
| --- | --- |
| *Sintel* © Blender Foundation | CC BY 3.0 |
| *Sintel*, doublage français de Touhoppai | CC BY 3.0 (Blender Foundation) et CC BY 4.0 (Touhoppai) |
| *Tears of Steel* © Blender Foundation | CC BY 3.0. Sa piste musique + effets (CC BY-ND 3.0) n'a servi qu'à mesurer : seuls les horaires de parole mesurés en dérivent |
| *Sprite Fright*, version française (Blender Foundation, Touhoppai) | CC BY 4.0 |
| *Pepper&Carrot*, épisode 6, version française | CC BY-SA 4.0 : les descriptions le sont aussi |
| *Le trésor de Sidiailles* (Kintésens) | Rien de publié, à cause des enfants à l'écran : ses descriptions sont masquées partout, seuls ses chiffres restent |

Sources des vidéos : [corpus/corpus.toml](../corpus/corpus.toml).
