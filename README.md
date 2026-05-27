# 9HPT Analiza — Avtomatska analiza Testa devetih zatičev

**Avtor:** Jernej Bartol  
**Predmet:** Robotski vid, FE UL — Izziv 2025/26  
**Repozitorij:** `JernejB-art/izziv-rv`

---

## Kaj sistem dela

Sistem avtomatsko analizira video posnetke **Testa devetih zatičev (9HPT)** brez posebne strojne opreme. Iz surovih `.mp4` datotek izračuna:

- skupni čas testa, čas vstavljanja in čas pospravljanja
- čase posameznih 9 zatičev (vstavljanje + pospravljanje)
- reakcijski čas (od signala do prvega pobira)
- kinematične parametre zapestja: pot `d(t)`, hitrost `v(t)`, pospešek `a(t)`
- katera roka je aktivna (leva / desna)
- primerjavo z referenčnimi CSV vrednostmi (MAE, Pearsonov r)

Izhod: **JSON** z vsemi parametri + **CSV** s kinematiko po frameih + **grafi** (`.png`).

---

## Struktura repozitorija

```
/workspace/
├── src/
│   ├── pipeline.py          ← glavna vstopna točka (TUKAJ ZAŽENEŠ)
│   ├── detect_combined.py   ← MediaPipe + homografija + FSM zaznava
│   ├── luknjice_led.py      ← zaznava LED luknjic, stroj stanj, veljavnost videa
│   ├── homografija.py       ← perspektivna normalizacija px → mm
│   ├── analizator_y.py      ← zaznava ciklov vstavljanja/pospravljanja iz Y-signala
│   ├── dva_grafi.py         ← kinematični grafi d/v/a, CSV izvoz
│   ├── evalvacija.py        ← primerjava z referenčnimi CSV vrednostmi
│   └── overlay.py           ← HUD anotacija izhodnega videa
├── results/                 ← sem se zapisujejo rezultati (avtomatsko ustvarjeno)
│   └── patient_XXX/
│       ├── patient_XXX_rezultati.json
│       ├── *_graf.png
│       ├── *_graf_y.png
│       ├── *_kinematika_multi.csv
│       └── evalvacija.png
└── README.md
```

Podatki (samo branje, ne spreminjaš):
```
/data/
└── Data/
    └── patient_001/
    └── patient_002/
    ...
```

---

## Zagon

### 1. Vstopi v Docker kontejner

Na strežniškem računalniku je moje delovno okolje znotraj: 

```bash 
/media/FastDataMama/jernejb
```

Odpremo ustrezno delovno mapo:

```bash 
cd izziv_rv
cd src
```

Zgradimo in poženemo docker:

```bash
docker build -t jernejb_rv /media/FastDataMama/jernejb/izziv_rv/
```

```bash
docker run --shm-size=16g -it \
  -v /media/FastDataMama/jernejb/izziv_rv:/workspace \
  -v /media/FastDataMama/data_rv_26:/data \
  -w /workspace \
  jernejb_rv bash
```

Ko je kontejner zagnan, boš v `/workspace`.

### 2. Zaženi pipeline

```bash
python3 src/pipeline.py
```

Program te interaktivno vodi:
1. Pokaže seznam razpoložljivih pacientov — vpiši številko ali ID (npr. `patient_024`)
2. Preveri veljavnost vseh posnetkov za izbranega pacienta
3. Pokaže seznam posnetkov z njihovim statusom (✓ veljaven / ✗ z razlogom)
4. Vpiši številko posnetka ali `0` za vse veljavne
5. Program obdela in izpiše rezultate v terminal + shrani v `results/`

---

## CLI način (brez interakcije)

Za avtomatsko obdelavo brez vprašanj:

```bash
# Določen pacient, vsi veljavni posnetki
python3 src/pipeline.py --pacient patient_024

# Določen pacient, določen video
python3 src/pipeline.py --pacient patient_024 --video /data/Data/patient_024/patient_024camP_1_20230511_14_11_19.mp4

# Drugačne poti do podatkov / rezultatov
python3 src/pipeline.py --pacient patient_024 --podatki /data/Data --izhod /workspace/results
```

| Argument | Privzeto | Opis |
|---|---|---|
| `--pacient` | interaktivno | ID pacienta, npr. `patient_024` |
| `--video` | vsi veljavni | Pot do enega specifičnega videa |
| `--podatki` | `/data/Data` | Mapa s podatki pacientov |
| `--izhod` | `/workspace/results` | Mapa za rezultate |

---

## Primer izpisa v terminalu

```
  ──────────────────────────────────────────────────────────
  9HPT ANALIZA  ·  pacient: patient_024
  ──────────────────────────────────────────────────────────
  Spol:        Female   Diagnoza: MS   Dom. roka: right
  Ref. časi:   P1=24.1s  P2=23.0s  S1=24.0s  S2=23.6s

  ── patient_024camP_1_20230511_14_11_19 ─────────────────
  ✓ Analiza zanesljiva
  Roka:                   desna
  Čas testa:              27.92 s
    ↳ vstavljanje:        18.44 s
    ↳ pospravljanje:      9.48 s
  Zatički (LED):          9
  Zaznano vstavljanj:     9/9  ✓
  Zaznano pospravljanj:   9/9  ✓
  Reakcijski čas:         0.760 s  (blink → 1. pobiranje)
  Časi vstavljanj [s]:    1.76  1.36  2.12  1.52  ...
  Časi pospravljanj [s]:  0.72  0.84  0.76  0.68  ...
  Kinematika zapestja (WRIST, 2D):
    ↳ pot skupaj:         4437 mm
    ↳ hitrost 95p:        440 mm/s
    ↳ pospešek 95p:       1633 mm/s²
  Primerjava CSV (P1):   izmerjeno 27.92s  ref 24.05s  (+16.1%)
```

---

## Izhodni rezultati

### JSON (`results/patient_XXX/patient_XXX_rezultati.json`)

Vsebuje vse izmerjene parametre za vsakega pacienta:
- metadata (datum analize, CSV OK/NOK)
- seznam obdelanih videov z vsemi vrednostmi
- povzetek (točnost zaznave vstavljanja/pospravljanja)

### Kinematični CSV (`*_kinematika_multi.csv`)

Stolpci po frameih za kazalec (TIP), palec (THUMB) in zapestje (CENTER):
```
frame, t_s, TIP_x_mm, TIP_y_mm, TIP_d_mm, TIP_v_mm_s, TIP_a_mm_s2,
             THUMB_x_mm, ..., CENTER_x_mm, ...
```

### Grafi (`.png`)

| Datoteka | Vsebina |
|---|---|
| `*_graf.png` | Homografija, trajektorija, kinematika d/v/a |
| `*_graf_y.png` | Y-signal z označenimi cikli vstavljanja/pospravljanja |
| `*_kinematika_multi.csv` + grafi | Kinematika za TIP / THUMB / CENTER |
| `evalvacija.png` | Bland-Altman diagram primerjave z referenco |

---

## Statusi veljavnosti posnetkov

| Status | Pomen |
|---|---|
| ✓ `[Xs]` | Veljavno, trajanje X sekund |
| ✗ `SAMO_TEMA` | Obe področji ugasnjeni — posnetek izven testa |
| ✗ `NI_BLINK_ZACETEK` | Video posnet sredi testa |
| ✗ `NI_KONEC` | Test se ni zaključil pred koncem videa |
| ✗ `KRATEK_VIDEO` | Trajanje pod 10 s |

---

## Zanesljivost analize

Analiza je označena kot **zanesljiva** (✓) samo kadar sta zaznana vsaj **7/9** zatičev v vsaki aktivni fazi. Ob nižji zaznavi sistem javi:

```
✗ Napaka v zaznavi, poskusi z drugim pogledom kamere.
  Pospravljanje: zaznano 3/9 (minimum 7)
```

---

## Odvisnosti (že v Docker sliki `jernejb_rv`)

```
mediapipe
opencv-python
scipy
numpy
matplotlib
scikit-learn
```

---

## Znane omejitve

- Sistem zanesljivo dela pri kotih kamere, kjer je Y-razlika med posodico in luknjicami dovolj velika (> ~80 mm v normaliziranem prostoru). Neugodni koti (frontalni pogled na roko) lahko povzročijo izpad zaznave.
- Časi testov so sistematično nekoliko previsoki (~+15 % v povprečju) — posledica zamude pri zaznavi konca LED signala.
- Zaznava pospravljanja je zanesljivejša pri počasnejšem gibanju; pri hitrejših pacientih je točnost nižja.

<img width="1915" height="1026" alt="image" src="https://github.com/user-attachments/assets/fb482f08-ad8b-40ea-ac9a-8a7254cae40c" />
