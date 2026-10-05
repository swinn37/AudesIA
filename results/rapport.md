# Rapport du corpus (05/10/2026)

Calcul par minute de vidéo : durée des étapes de ce run, sans le précalcul (extrait, parole, plans) quand il était déjà fait. Chevauchement : contre la vérité terrain tirée de la piste musique + effets, pour les vidéos qui en ont une ; « aucune voix » compte aussi les cris et les souffles. Musique retirée du résidu : sous 3,0 dB, la piste ne retire presque rien du mixage, la mesure n'est pas probante et ne compte pas au total.

## Profil small : gemma4:26b-a4b-it-qat

| Vidéo | Durée | Placées | Couverture | Sans chevauchement des répliques | Sans chevauchement d'aucune voix | Musique retirée du résidu | Calcul par minute de vidéo | Jetons envoyés / générés | Pic GPU | Hors ligne |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pepper-carrot-6-vf | 7:37 | 26 sur 26 | 78 % | — | — | — | 4,0 min | 96 k / 7 k | 8,0 Gio | non |
| sidiailles | 11:56 | 79 sur 81 | 69 % | — | — | — | 6,9 min | 302 k / 16 k | 8,0 Gio | non |
| sintel-vf | 17:02 | 94 sur 97 | 72 % | 92 sur 94 (98 %) | 80 sur 94 (85 %) | 20,0 dB | 6,2 min | 369 k / 23 k | 8,0 Gio | non |
| sintel-vo | 14:48 | 113 sur 115 | 73 % | 110 sur 113 (97 %) | 101 sur 113 (89 %) | 6,8 dB | 9,2 min | 393 k / 24 k | 8,0 Gio | non |
| sprite-fright-vf | 10:26 | 32 sur 35 | 69 % | — | — | — | 3,4 min | 132 k / 8 k | 8,0 Gio | non |
| tears-of-steel | 12:14 | 75 sur 79 | 71 % | 75 sur 75 (100 %) (non probant) | 75 sur 75 (100 %) (non probant) | 1,6 dB | 5,3 min | 288 k / 18 k | 8,0 Gio | non |
| **Total** | 74:04 | 419 sur 433 | 72 % | 202 sur 207 (98 %) | 181 sur 207 (87 %) | | 6,1 min | | | |

## Juge qwen3.6:35b-a3b : small, notation seule

| Vidéo | exactitude | pertinence | coherence | concision | Exactitude ≤ 2 |
| --- | --- | --- | --- | --- | --- |
| pepper-carrot-6-vf | 4,46 (26) | 3,54 (26) | 4,88 (26) | 4,62 (26) | 3 |
| sidiailles | 4,13 (79) | 3,35 (79) | 4,54 (79) | 4,66 (79) | 11 |
| sintel-vf | 4,31 (94) | 3,60 (94) | 4,79 (94) | 4,72 (94) | 11 |
| sintel-vo | 4,32 (113) | 3,54 (113) | 4,82 (113) | 4,65 (113) | 11 |
| sprite-fright-vf | 3,91 (32) | 3,28 (32) | 4,69 (32) | 4,44 (32) | 5 |
| tears-of-steel | 4,52 (75) | 3,69 (75) | 4,83 (75) | 4,75 (75) | 4 |
| **Total** | 4,29 (419) | 3,53 (419) | 4,76 (419) | 4,67 (419) | 45 |

Descriptions les moins exactes :

| Vidéo | Moment | Description | Raison du juge |
| --- | --- | --- | --- |
| pepper-carrot-6-vf | 2:21 | Le mot Saffron s'affiche sur fond rouge. | La description est totalement fausse car elle décrit un plan qui n'est pas celui des images fournies. |
| sidiailles | 1:54 | (description non publiée : évaluation locale seulement) |  |
| sidiailles | 3:02 | (description non publiée : évaluation locale seulement) |  |
| sidiailles | 3:40 | (description non publiée : évaluation locale seulement) |  |
| sidiailles | 9:13 | (description non publiée : évaluation locale seulement) |  |
| sintel-vf | 6:30 | La jeune femme regarde vers la droite sous un ciel orangé. | La description précédente concerne une jeune femme dans un désert, mais les images montrent un homme avec une canne dans un paysage brumeux et verdoyant. |
| sintel-vo | 4:08 | Une poule. | La description est incorrecte car elle ne mentionne pas l'action principale de la poule qui court dans une ruelle en ruine. |
| sintel-vo | 6:36 | De dos, elle lève la main gauche face à l'objectif. | La description est totalement fausse car elle décrit une personne de dos levant la main gauche, alors que les images montrent clairement un personnage de face qui baisse et relève la tête. |
| sprite-fright-vf | 5:07 | Une cassette « VEEJAY » entre dans un lecteur. | La description est totalement fausse et ne correspond pas aux images qui montrent une fille avec une lampe torche et des champignons verts. |
| pepper-carrot-6-vf | 4:48 | Un oiseau jaune à perruque et perles crie sous une aura verte. | L'oiseau n'est plus vert mais jaune et l'aura verte a disparu. |
