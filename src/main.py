#!/usr/bin/env python3
# main.py
# Osrednji program za analizo 9HPT testa.
#
# Uporaba:
#   python3 main.py 024
#   python3 main.py patient_024
#   python3 main.py 024 --podatki /data/Data --izhod /workspace/results
#
# Program:
#   1. Poišče vse datoteke pacienta v /data/Data/patient_XXX/
#   2. Razvrsti videoposnetke in CSV po poskusih (časovna oznaka v imenu)
#   3. Za vsak poskus požene analizo LED luknjic + kinematiko na vseh kamerah
#   4. Združi rezultate kamer (outlier robustna srednja vrednost)
#   5. Primerja z referenčnimi CSV vrednostmi
#   6. Izvozi grafe, debug video in poročilo

import os
import sys
import re
import glob
import argparse
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

# ── Uvozi lastnih modulov ────────────────────────────────────────────────────
# Predpostavka: main.py je v isti mapi kot ostali moduli,
# oz. so v src/ — poskusimo oba načina
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'src'))

try:
    from holes import (
        analiziraj_led_luknjice,
        preveri_veljavnost_videa,
        integriraj_v_detect,
        ROI_PARAMETRI,
        doloci_kamero,
    )
except ImportError as e:
    print(f"[NAPAKA] Ne morem uvoziti holes: {e}")
    sys.exit(1)

try:
    from kinematics import (
        izracun_center_roke,
        izracun_kazalec,
        izracun_kinematika,
        zaznava_faze_testa,
        zaznava_zaticev,
        filtriraj_skoke_kazalec,
        interpoliraj_manjkajoce,
        glajenje_signal,
    )
except ImportError as e:
    print(f"[NAPAKA] Ne morem uvoziti kinematics: {e}")
    sys.exit(1)

try:
    from csv_reader import preberi_csv_pacienta
except ImportError as e:
    print(f"[NAPAKA] Ne morem uvoziti csv_reader: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[OPOZORILO] MediaPipe ni nameščen — kinematična analiza ne bo možna.")

# ── Konstante ────────────────────────────────────────────────────────────────
PRIVZETA_POT_PODATKOV = '/data/Data'
PRIVZETA_POT_IZHODA   = '/workspace/results'

# Ime kamere → indeks za sortiranje (nižji indeks = boljši pogled)
KAMERA_PRIORITETA = {'camP_0': 0, 'camP_1': 1, 'camP_2': 2}

# Minimalni čas testa v sekundah (krajše = najverjetneje napaka)
MIN_CAS_TESTA = 5.0

# Prag za zavrnitev outlierja pri združevanju kamer
# Če vrednost ene kamere odstopa za več kot OUTLIER_K * std od median ostalih,
# se zavrne
OUTLIER_K = 2.0


# ════════════════════════════════════════════════════════════════════════════
# 1. ISKANJE IN RAZVRŠČANJE DATOTEK
# ════════════════════════════════════════════════════════════════════════════

def normaliziraj_id(id_pacienta):
    """
    Sprejme '24', '024', 'patient_024' — vedno vrne 'patient_024'.
    """
    id_pacienta = str(id_pacienta).strip()
    if id_pacienta.startswith('patient_'):
        stevka = id_pacienta.split('_')[1]
    else:
        stevka = id_pacienta
    return f"patient_{int(stevka):03d}"


def poisci_datoteke_pacienta(pot_podatkov, id_pacienta):
    """
    Vrne slovar:
      {
        'mapa':    str,
        'videi':   [(pot, ime_datoteke, kamera, cas_oznaka), ...],
        'csv':     [pot, ...],
      }
    """
    mapa = os.path.join(pot_podatkov, id_pacienta)
    if not os.path.isdir(mapa):
        raise FileNotFoundError(f"Mapa pacienta ne obstaja: {mapa}")

    # Videi — iščemo .mp4
    videi_raw = glob.glob(os.path.join(mapa, '*.mp4'))
    videi_raw += glob.glob(os.path.join(mapa, '*.MP4'))

    videi = []
    for pot in videi_raw:
        ime = os.path.basename(pot)
        kamera = doloci_kamero(ime)
        cas = _izvleci_cas_oznako(ime)
        videi.append((pot, ime, kamera, cas))

    # Sortiranje po časovni oznaki, potem po kameri
    videi.sort(key=lambda x: (x[3] or '0', KAMERA_PRIORITETA.get(x[2], 99)))

    # CSV datoteke
    csv_poti = glob.glob(os.path.join(mapa, '*.csv'))
    csv_poti += glob.glob(os.path.join(mapa, '*.CSV'))

    return {
        'mapa':  mapa,
        'videi': videi,
        'csv':   csv_poti,
    }


def _izvleci_cas_oznako(ime_datoteke):
    """
    Iz imena kot 'patient_024camP_0_20230511_14_12_59.mp4'
    izvleče '20230511_14_12_59' za sortiranje.
    """
    vzorec = r'(\d{8}_\d{2}_\d{2}_\d{2})'
    match = re.search(vzorec, ime_datoteke)
    return match.group(1) if match else None


def razvrsti_v_poskuse(videi):
    """
    Grupira videoposnetke po točno enaki časovni oznaki v imenu datoteke.
    Cam0/cam1/cam2 istega snemanja imajo identično oznako (npr. 20230511_14_11_19).
    Drugačna oznaka = drug poskus.

    Znotraj iste seanse (isti datum) so poskusi urejeni kronološko:
      1. poskus = dominantna roka
      2. poskus = nedominantna roka
      3. poskus = dominantna roka (ponavljanje)
      4. poskus = nedominantna roka (ponavljanje)

    Več CSV datotek = več seans (razberemo iz datuma v imenu CSV).

    Vrne seznam poskusov, vsak je seznam (pot, ime, kamera, cas).
    """
    if not videi:
        return []

    from collections import OrderedDict
    grupirano = OrderedDict()
    for v in videi:
        kljuc = v[3] or 'neznan'
        if kljuc not in grupirano:
            grupirano[kljuc] = []
        grupirano[kljuc].append(v)

    return list(grupirano.values())


# ════════════════════════════════════════════════════════════════════════════
# 2. KINEMATIČNA ANALIZA ENEGA VIDEA
# ════════════════════════════════════════════════════════════════════════════

def analiziraj_kinematiko_videa(pot_videa, cas_zacetka_s=None, cas_konca_s=None):
    """
    Požene MediaPipe na videu in izračuna kinematične parametre.
    Če sta podana cas_zacetka_s in cas_konca_s (iz LED analize),
    kinematiko izračuna samo za ta interval.

    Vrne slovar z vsemi parametri ali None ob napaki.
    """
    if not MP_AVAILABLE:
        return None

    cap = cv2.VideoCapture(pot_videa)
    if not cap.isOpened():
        print(f"  [KIN] Ne morem odpreti: {pot_videa}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skupaj = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Določi frame interval za analizo
    f_zacetek = int(cas_zacetka_s * fps) if cas_zacetka_s else 0
    f_konec   = int(cas_konca_s   * fps) if cas_konca_s   else skupaj

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pozicije_roke    = []
    pozicije_kazalec = []
    frame_indeksi    = []
    frame_idx        = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_zacetek <= frame_idx <= f_konec:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                cx, cy = izracun_center_roke(lm, sirina, visina)
                kx, ky = izracun_kazalec(lm, sirina, visina)
                pozicije_roke.append((cx, cy))
                pozicije_kazalec.append((kx, ky))
                frame_indeksi.append(frame_idx - f_zacetek)

        frame_idx += 1

    cap.release()
    hands.close()

    dolzina = f_konec - f_zacetek + 1

    if len(pozicije_roke) < fps * 2:
        print(f"  [KIN] Premalo zaznav roke ({len(pozicije_roke)} framov)")
        return None

    # Interpoliraj manjkajoče frame-e
    pos_roke_int = interpoliraj_manjkajoce(pozicije_roke, frame_indeksi, dolzina)
    pos_kaz_int  = interpoliraj_manjkajoce(pozicije_kazalec, frame_indeksi, dolzina)
    pos_kaz_int  = filtriraj_skoke_kazalec(pos_kaz_int, fps)

    # Kinematika
    kin = izracun_kinematika(pos_roke_int, fps)

    # Zaznava faz
    faze = zaznava_faze_testa(pos_roke_int, fps)

    # Zaznava zatičev
    zaticev = zaznava_zaticev(pos_kaz_int, fps)

    return {
        'fps':               fps,
        'kin':               kin,
        'faze':              faze,
        'zaticev':           zaticev,
        'pozicije_roke':     pos_roke_int,
        'pozicije_kazalec':  pos_kaz_int,
        'dolzina_framov':    dolzina,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. ANALIZA ENEGA POSKUSA (VEČ KAMER)
# ════════════════════════════════════════════════════════════════════════════

def analiziraj_poskus(videi_poskusa, izhod_mapa, indeks_poskusa):
    """
    Za en poskus (seznam posnetkov z različnih kamer) izvede:
      - LED analizo na vsakem posnetku
      - Kinematično analizo na vsakem posnetku
      - Združi rezultate kamer z outlier-robustno sredino

    Vrne slovar z združenimi rezultati.
    """
    os.makedirs(izhod_mapa, exist_ok=True)
    print(f"\n{'─'*60}")
    print(f"  POSKUS {indeks_poskusa+1} | {len(videi_poskusa)} kamer")
    print(f"{'─'*60}")

    rezultati_kamer = {}

    for pot, ime, kamera, cas in videi_poskusa:
        print(f"\n  [{kamera}] {ime}")
        izhod_prefix = os.path.join(izhod_mapa,
            f"poskus{indeks_poskusa+1:02d}_{kamera}")

        # ── LED analiza ──────────────────────────────────────────────────
        veljavno, razlog = preveri_veljavnost_videa(pot)
        if not veljavno:
            print(f"  [LED] PRESKOČI: {razlog}")
            continue

        led = analiziraj_led_luknjice(
            pot,
            izhod_video=izhod_prefix + '_led_debug.mp4',
            izhod_graf=izhod_prefix + '_led_graf.png',
        )

        if led is None or led['cas_testa'] is None:
            print(f"  [LED] Čas testa ni bil zaznán — preskočim")
            continue

        if led['cas_testa'] < MIN_CAS_TESTA:
            print(f"  [LED] Čas testa {led['cas_testa']:.1f}s je prekratek — preskočim")
            continue

        # ── Kinematična analiza ──────────────────────────────────────────
        kin_rez = None
        if MP_AVAILABLE:
            print(f"  [KIN] Začenjam kinematično analizo ...")
            kin_rez = analiziraj_kinematiko_videa(
                pot,
                cas_zacetka_s=led['cas_zacetka_s'],
                cas_konca_s=led['cas_konca_s'],
            )

        rezultati_kamer[kamera] = {
            'led':    led,
            'kin':    kin_rez,
            'pot':    pot,
        }

        # Izpis za to kamero
        print(f"  [LED] Čas: {led['cas_testa']:.2f}s | "
              f"Zatiči: {led['stevilo_zaticov']} | Roka: {led['roka']}")
        if kin_rez:
            st_zat_kin = kin_rez['zaticev'].get('stevilo_zaticev', '?')
            print(f"  [KIN] Zatičev (kinematika): {st_zat_kin}")

    if not rezultati_kamer:
        print("  [!] Noben posnetek tega poskusa ni bil uspešno analiziran.")
        return None

    # ── Združevanje rezultatov kamer ────────────────────────────────────
    zdruzen = zdruzi_kamere(rezultati_kamer, indeks_poskusa)
    return zdruzen


def zdruzi_kamere(rezultati_kamer, indeks_poskusa):
    """
    Iz rezultatov več kamer z outlier-robustno mediano izračuna
    skupne vrednosti za en poskus.

    Logika:
    - Zberi vrednosti za vsako metriko iz vseh kamer
    - Izloči kamero katere vrednost odstopa za >OUTLIER_K*std od mediane
    - Vrni mediano preostalih vrednosti in seznam zavrženih kamer
    """
    kamere    = list(rezultati_kamer.keys())
    n_kamer   = len(kamere)

    # ── Čas testa (LED) ──────────────────────────────────────────────────
    casi_led  = np.array([rezultati_kamer[k]['led']['cas_testa'] for k in kamere])
    cas_led, zavrzene_led = _robustna_sredina(casi_led, kamere)

    # ── Število zatičev (LED) ────────────────────────────────────────────
    zaticev_led = np.array([rezultati_kamer[k]['led']['stevilo_zaticov'] for k in kamere],
                           dtype=float)
    st_zat_led, _ = _robustna_sredina(zaticev_led, kamere)

    # ── Roka (glasovanje) ────────────────────────────────────────────────
    rok_glasovi = {}
    for k in kamere:
        r = rezultati_kamer[k]['led']['roka']
        rok_glasovi[r] = rok_glasovi.get(r, 0) + 1
    roka = max(rok_glasovi, key=rok_glasovi.get)

    # ── Kinematika ───────────────────────────────────────────────────────
    kin_skupaj = None
    kamere_z_kin = [k for k in kamere if rezultati_kamer[k]['kin'] is not None]
    if kamere_z_kin:
        # Skupna pot, max hitrost, povprečna hitrost, max pospešek
        skupne_poti    = np.array([rezultati_kamer[k]['kin']['kin']['pot'][-1]
                                   for k in kamere_z_kin])
        max_hitrosti   = np.array([np.max(rezultati_kamer[k]['kin']['kin']['hitrost'])
                                   for k in kamere_z_kin])
        pov_hitrosti   = np.array([np.mean(rezultati_kamer[k]['kin']['kin']['hitrost'])
                                   for k in kamere_z_kin])
        max_pospeski   = np.array([np.max(np.abs(rezultati_kamer[k]['kin']['kin']['pospesek']))
                                   for k in kamere_z_kin])
        zat_kin_arr    = np.array([rezultati_kamer[k]['kin']['zaticev']['stevilo_zaticev']
                                   for k in kamere_z_kin], dtype=float)

        kin_skupaj = {
            'skupna_pot_px':   float(_robustna_sredina(skupne_poti, kamere_z_kin)[0]),
            'max_hitrost_px_s':float(_robustna_sredina(max_hitrosti, kamere_z_kin)[0]),
            'pov_hitrost_px_s':float(_robustna_sredina(pov_hitrosti, kamere_z_kin)[0]),
            'max_pospesek':    float(_robustna_sredina(max_pospeski, kamere_z_kin)[0]),
            'stevilo_zaticev': float(_robustna_sredina(zat_kin_arr, kamere_z_kin)[0]),
            'kamere':          kamere_z_kin,
        }

    # ── Katera kamera je "najboljša" (za debug video) ────────────────────
    najboljsa_kamera = None
    for k in sorted(kamere, key=lambda x: KAMERA_PRIORITETA.get(x, 99)):
        if k not in zavrzene_led:
            najboljsa_kamera = k
            break
    if najboljsa_kamera is None and kamere:
        najboljsa_kamera = kamere[0]

    return {
        'indeks':           indeks_poskusa,
        'cas_testa_s':      float(cas_led),
        'stevilo_zaticov_led': int(round(st_zat_led)),
        'roka':             roka,
        'kin':              kin_skupaj,
        'kamere':           kamere,
        'zavrzene_kamere':  zavrzene_led,
        'najboljsa_kamera': najboljsa_kamera,
        'rezultati_kamer':  rezultati_kamer,
    }


def _robustna_sredina(vrednosti, kamere):
    """
    Iz array vrednosti vrne (robustna_mediana, seznam_zavrzenih_kamer).
    Zavrže vrednosti ki so oddaljene od mediane za >OUTLIER_K*std.
    Če so manj kot 2 vrednosti, vrne kar imamo brez zavrnitve.
    """
    vrednosti = np.array(vrednosti, dtype=float)
    kamere    = list(kamere)

    if len(vrednosti) <= 1:
        return float(vrednosti[0]) if len(vrednosti) else 0.0, []

    mediana = np.median(vrednosti)
    std     = np.std(vrednosti)

    if std < 1e-6:
        return float(mediana), []

    maska_ok = np.abs(vrednosti - mediana) <= OUTLIER_K * std
    zavrzene = [kamere[i] for i, ok in enumerate(maska_ok) if not ok]

    vrednosti_ok = vrednosti[maska_ok]
    if len(vrednosti_ok) == 0:
        vrednosti_ok = vrednosti  # ne zavrzi vsega

    return float(np.median(vrednosti_ok)), zavrzene


# ════════════════════════════════════════════════════════════════════════════
# 4. PRIMERJAVA Z REFERENČNIMI VREDNOSTMI (CSV)
# ════════════════════════════════════════════════════════════════════════════

def primerjaj_z_referenco(poskusi_rezultati, csv_podatki):
    """
    Primerja izmerjene čase testov z referenčnimi iz CSV.

    CSV ima 4 meritve: P1 (dominantna), P2 (nedominantna),
                       S1 (dominantna), S2 (nedominantna).

    Problem: ni nujno da imamo posnete vse 4 meritve. Rešitev:
    - Iz metadata CSV razberemo katera roka je dominantna
    - Vsak posnetek ima zaznano roko (leva/desna)
    - Znotraj seanse ločimo posnetke na dominantne in nedominantne
    - Dominantni posnetki (kronološko) → P1, S1
    - Nedominantni posnetki (kronološko) → P2, S2
    - Manjkajoče meritve dobijo None → jasno vidno v poročilu

    Vrne tabelo primerjav (vključno z None za manjkajoče meritve).
    """
    if csv_podatki is None:
        # Brez reference — samo izpišemo kar imamo
        primerjave = []
        for i, res in enumerate(poskusi_rezultati):
            if res is None:
                continue
            primerjave.append({
                'poskus':      i + 1,
                'oznaka':      f'P{i+1}',
                'izmerjeno_s': res['cas_testa_s'],
                'ref_s':       None,
                'napaka_s':    None,
                'napaka_pct':  None,
                'roka':        res['roka'],
                'zaticev_led': res['stevilo_zaticov_led'],
                'kin':         res['kin'],
            })
        return primerjave

    ref_casi = csv_podatki['skupni_casi']
    dominantna_roka = csv_podatki['metadata'].get('roka', 'desna').lower()
    # Normalizacija: CSV vrednost je npr. "D" ali "desna" ali "right"
    if dominantna_roka in ('d', 'desna', 'right', 'r'):
        dominantna_roka = 'desna'
    else:
        dominantna_roka = 'leva'

    # Razvrsti veljavne posnetke po dominantnosti
    # Vsak slot: (oznaka_csv, ref_cas, rezultat_ali_None)
    slots = {'P1': None, 'P2': None, 'S1': None, 'S2': None}
    stevci = {'dom': 0, 'nedom': 0}  # koliko smo že dodelili

    for res in poskusi_rezultati:
        if res is None:
            continue
        roka = res['roka']
        je_dominantna = (roka == dominantna_roka)

        if je_dominantna:
            stevci['dom'] += 1
            oznaka = 'P1' if stevci['dom'] == 1 else 'S1'
        else:
            stevci['nedom'] += 1
            oznaka = 'P2' if stevci['nedom'] == 1 else 'S2'

        if oznaka in slots:
            slots[oznaka] = res

    # Sestavi primerjave — vključno z None za manjkajoče
    primerjave = []
    for oznaka in ['P1', 'P2', 'S1', 'S2']:
        res = slots[oznaka]
        ref_cas = ref_casi.get(oznaka)

        if res is not None:
            izmerjeno = res['cas_testa_s']
            napaka_s   = (izmerjeno - ref_cas) if ref_cas else None
            napaka_pct = (100 * napaka_s / ref_cas) if ref_cas else None
            primerjave.append({
                'poskus':      oznaka,
                'oznaka':      oznaka,
                'izmerjeno_s': izmerjeno,
                'ref_s':       ref_cas,
                'napaka_s':    napaka_s,
                'napaka_pct':  napaka_pct,
                'roka':        res['roka'],
                'zaticev_led': res['stevilo_zaticov_led'],
                'kin':         res['kin'],
                'manjka':      False,
            })
        else:
            # Manjkajoča meritev — vnos z None vrednostmi
            primerjave.append({
                'poskus':      oznaka,
                'oznaka':      oznaka,
                'izmerjeno_s': None,
                'ref_s':       ref_cas,
                'napaka_s':    None,
                'napaka_pct':  None,
                'roka':        'dominantna' if oznaka in ('P1','S1') else 'nedominantna',
                'zaticev_led': None,
                'kin':         None,
                'manjka':      True,
            })

    return primerjave


# ════════════════════════════════════════════════════════════════════════════
# 5. VIZUALIZACIJA IN IZVOZ
# ════════════════════════════════════════════════════════════════════════════

def izvozi_porocilo(id_pacienta, primerjave, poskusi_rezultati, csv_podatki,
                    izhod_mapa):
    """
    Ustvari:
      - skupni_pregled.png  — tabela primerjav + tortni/bar grafi
      - kinematika_POSKUSXX.png — grafi d/v/a za vsak poskus
      - porocilo.json       — vsi rezultati v strojno berljivi obliki
    """
    os.makedirs(izhod_mapa, exist_ok=True)

    # ── Skupni pregled ────────────────────────────────────────────────────
    _narisi_skupni_pregled(id_pacienta, primerjave, izhod_mapa)

    # ── Kinematika za vsak poskus ─────────────────────────────────────────
    for res in poskusi_rezultati:
        if res is None:
            continue
        _narisi_kinematiko_poskusa(res, izhod_mapa)

    # ── JSON poročilo ─────────────────────────────────────────────────────
    _izvozi_json(id_pacienta, primerjave, izhod_mapa)

    print(f"\n[IZHOD] Rezultati shranjeni v: {izhod_mapa}")


def _narisi_skupni_pregled(id_pacienta, primerjave, izhod_mapa):
    if not primerjave:
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"9HPT Analiza — {id_pacienta}", fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Levo zgoraj: tabela primerjav ─────────────────────────────────────
    ax_tab = fig.add_subplot(gs[0, :])
    ax_tab.axis('off')

    vrstice_tabele = []
    for p in primerjave:
        nap_str = (f"{p['napaka_s']:+.2f}s ({p['napaka_pct']:+.1f}%)"
                   if p['napaka_s'] is not None else "—")
        kin_pot = (f"{p['kin']['skupna_pot_px']:.0f}px"
                   if p['kin'] else "—")
        kin_maxv = (f"{p['kin']['max_hitrost_px_s']:.0f}px/s"
                    if p['kin'] else "—")
        vrstice_tabele.append([
            p['oznaka'],
            p['roka'],
            f"{p['izmerjeno_s']:.2f}s",
            f"{p['ref_s']:.2f}s" if p['ref_s'] else "—",
            nap_str,
            str(p['zaticev_led']),
            kin_pot,
            kin_maxv,
        ])

    stolpci = ['Poskus', 'Roka', 'Izmerjeno', 'Referenca',
               'Napaka', 'Zatiči (LED)', 'Pot (kin)', 'Max v (kin)']

    tabela = ax_tab.table(
        cellText=vrstice_tabele,
        colLabels=stolpci,
        loc='center',
        cellLoc='center',
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 1.6)

    # Pobarvaj glavo
    for j in range(len(stolpci)):
        tabela[0, j].set_facecolor('#2c5f8a')
        tabela[0, j].set_text_props(color='white', fontweight='bold')

    # Pobarvaj vrstice ki imajo napako >15%
    for i, p in enumerate(primerjave):
        if p['napaka_pct'] is not None and abs(p['napaka_pct']) > 15:
            for j in range(len(stolpci)):
                tabela[i + 1, j].set_facecolor('#ffe0e0')

    ax_tab.set_title('Primerjava z referenčnimi vrednostmi', fontsize=11, pad=12)

    # ── Levo spodaj: barplot — izmerjeno vs. referenca ────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    oznake = [p['oznaka'] for p in primerjave]
    izm    = [p['izmerjeno_s'] for p in primerjave]
    ref    = [p['ref_s'] if p['ref_s'] else 0 for p in primerjave]
    x      = np.arange(len(oznake))
    sirina = 0.35

    bars1 = ax_bar.bar(x - sirina/2, ref, sirina, label='Referenca', color='steelblue', alpha=0.8)
    bars2 = ax_bar.bar(x + sirina/2, izm, sirina, label='Izmerjeno',  color='darkorange', alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(oznake)
    ax_bar.set_ylabel('Čas [s]')
    ax_bar.set_title('Časi testov')
    ax_bar.legend()
    ax_bar.grid(axis='y', alpha=0.3)

    # ── Desno spodaj: scatter — napaka po poskusih ────────────────────────
    ax_scat = fig.add_subplot(gs[1, 1])
    napake  = [p['napaka_pct'] if p['napaka_pct'] is not None else 0
               for p in primerjave]
    barve   = ['green' if abs(n) <= 10 else ('orange' if abs(n) <= 20 else 'red')
               for n in napake]
    ax_scat.bar(oznake, napake, color=barve, alpha=0.8)
    ax_scat.axhline(0,   color='black', linewidth=0.8)
    ax_scat.axhline( 10, color='green',  linestyle='--', alpha=0.5, label='±10%')
    ax_scat.axhline(-10, color='green',  linestyle='--', alpha=0.5)
    ax_scat.axhline( 20, color='orange', linestyle='--', alpha=0.5, label='±20%')
    ax_scat.axhline(-20, color='orange', linestyle='--', alpha=0.5)
    ax_scat.set_ylabel('Napaka [%]')
    ax_scat.set_title('Relativna napaka časa')
    ax_scat.legend(fontsize=8)
    ax_scat.grid(axis='y', alpha=0.3)

    izhod = os.path.join(izhod_mapa, 'skupni_pregled.png')
    plt.savefig(izhod, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IZHOD] skupni_pregled.png")


def _narisi_kinematiko_poskusa(res, izhod_mapa):
    """
    Nariše grafe d/v/a za najboljšo kamero poskusa.
    """
    best_kamera = res.get('najboljsa_kamera')
    if best_kamera is None:
        return

    kin_rez = res['rezultati_kamer'].get(best_kamera, {}).get('kin')
    if kin_rez is None:
        return

    kin = kin_rez['kin']
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Kinematika — Poskus {res['indeks']+1} | {res['roka']} roka | "
        f"Kamera: {best_kamera} | Čas: {res['cas_testa_s']:.2f}s",
        fontsize=12, fontweight='bold'
    )

    axs[0].plot(kin['cas_pot'],     kin['pot'],     color='steelblue')
    axs[0].set_ylabel('Pot [px]')
    axs[0].set_title('Kumulativna pot d(t)')
    axs[0].grid(alpha=0.3)

    axs[1].plot(kin['cas_hitrost'], kin['hitrost'], color='darkorange')
    axs[1].set_ylabel('Hitrost [px/s]')
    axs[1].set_title('Hitrost v(t)')
    axs[1].grid(alpha=0.3)

    axs[2].plot(kin['cas_pospesek'], kin['pospesek'], color='firebrick')
    axs[2].set_ylabel('Pospešek [px/s²]')
    axs[2].set_xlabel('Čas [s]')
    axs[2].set_title('Pospešek a(t)')
    axs[2].grid(alpha=0.3)

    # Označi faze (vrhovi = pobiranje, doline = vstavljanje)
    faze = kin_rez['faze']
    fps  = kin_rez['fps']
    for vi in faze.get('vrhovi_idx', []):
        t = vi / fps
        for ax in axs:
            ax.axvline(t, color='green', alpha=0.3, linewidth=0.8)
    for di in faze.get('doline_idx', []):
        t = di / fps
        for ax in axs:
            ax.axvline(t, color='red', alpha=0.3, linewidth=0.8)

    # Legenda za faze
    from matplotlib.lines import Line2D
    legenda = [
        Line2D([0], [0], color='green', alpha=0.6, label='Pobiranje'),
        Line2D([0], [0], color='red',   alpha=0.6, label='Vstavljanje'),
    ]
    axs[0].legend(handles=legenda, fontsize=8, loc='upper left')

    plt.tight_layout()
    izhod = os.path.join(izhod_mapa, f"kinematika_poskus{res['indeks']+1:02d}.png")
    plt.savefig(izhod, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IZHOD] kinematika_poskus{res['indeks']+1:02d}.png")


def _izvozi_json(id_pacienta, primerjave, izhod_mapa):
    """
    Shrani vse numerične rezultate v JSON za nadaljnjo obdelavo.
    """
    def serializiraj(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Neserializabilen tip: {type(obj)}")

    izhod = {
        'id_pacienta': id_pacienta,
        'cas_analize': datetime.now().isoformat(),
        'poskusi': []
    }

    for p in (primerjave or []):
        vnos = {
            'oznaka':      p['oznaka'],
            'roka':        p['roka'],
            'izmerjeno_s': p['izmerjeno_s'],
            'ref_s':       p['ref_s'],
            'napaka_s':    p['napaka_s'],
            'napaka_pct':  p['napaka_pct'],
            'zaticev_led': p['zaticev_led'],
        }
        if p['kin']:
            vnos['kin'] = {
                'skupna_pot_px':    p['kin']['skupna_pot_px'],
                'max_hitrost_px_s': p['kin']['max_hitrost_px_s'],
                'pov_hitrost_px_s': p['kin']['pov_hitrost_px_s'],
                'max_pospesek':     p['kin']['max_pospesek'],
                'stevilo_zaticev':  p['kin']['stevilo_zaticev'],
            }
        izhod['poskusi'].append(vnos)

    pot_json = os.path.join(izhod_mapa, 'porocilo.json')
    with open(pot_json, 'w', encoding='utf-8') as f:
        json.dump(izhod, f, ensure_ascii=False, indent=2, default=serializiraj)
    print(f"[IZHOD] porocilo.json")


# ════════════════════════════════════════════════════════════════════════════
# 6. IZPIS V KONZOLO
# ════════════════════════════════════════════════════════════════════════════

def izpisi_povzetek(id_pacienta, primerjave, csv_podatki):
    print(f"\n{'═'*70}")
    print(f"  POVZETEK — {id_pacienta}")
    if csv_podatki:
        meta = csv_podatki['metadata']
        print(f"  Spol: {meta['spol']}  |  Diagnoza: {meta['diagnoza']}  |  Roka: {meta['roka']}")
    print(f"{'═'*70}")

    if not primerjave:
        print("  Ni veljavnih rezultatov.")
        return

    skupna_napaka = []
    print(f"\n  {'Psk':>3} {'Oznaka':>7} {'Roka':>7} {'Izm [s]':>9} {'Ref [s]':>9} "
          f"{'Napaka':>10} {'Zatiči':>7}")
    print(f"  {'-'*60}")

    for p in primerjave:
        nap = (f"{p['napaka_pct']:+.1f}%"
               if p['napaka_pct'] is not None else "    —  ")
        ref = f"{p['ref_s']:.2f}" if p['ref_s'] else "   —  "
        izm = f"{p['izmerjeno_s']:.2f}" if p['izmerjeno_s'] is not None else "  MANJKA"
        zat = str(p['zaticev_led']) if p['zaticev_led'] is not None else "—"
        manjka_oznaka = " ⚠" if p.get('manjka') else "  "
        print(f"  {str(p['oznaka']):>3}{manjka_oznaka} {p['roka']:>12} "
              f"{izm:>9} {ref:>9} {nap:>10} {zat:>7}")
        if p['napaka_pct'] is not None:
            skupna_napaka.append(abs(p['napaka_pct']))

    if skupna_napaka:
        print(f"\n  Povprečna absolutna napaka časa: {np.mean(skupna_napaka):.1f}%")

    print(f"{'═'*70}\n")


# ════════════════════════════════════════════════════════════════════════════
# 7. GLAVNI VSTOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='9HPT analiza pacienta — LED + kinematika + primerjava z CSV'
    )
    parser.add_argument('id_pacienta',
        help="Številka ali ID pacienta (npr. '024' ali 'patient_024')")
    parser.add_argument('--podatki', default=PRIVZETA_POT_PODATKOV,
        help=f"Korenski imenik s podatki (privzeto: {PRIVZETA_POT_PODATKOV})")
    parser.add_argument('--izhod', default=PRIVZETA_POT_IZHODA,
        help=f"Izhodni imenik (privzeto: {PRIVZETA_POT_IZHODA})")
    parser.add_argument('--brez-kin', action='store_true',
        help="Preskoči kinematično analizo (samo LED)")
    parser.add_argument('--samo-poskus', type=int, default=None,
        help="Analiziraj samo en poskus (1-indeksirano)")
    args = parser.parse_args()

    id_pac = normaliziraj_id(args.id_pacienta)
    print(f"\n{'█'*70}")
    print(f"  9HPT ANALIZA — {id_pac}")
    print(f"{'█'*70}")
    print(f"  Podatki: {args.podatki}")
    print(f"  Izhod:   {args.izhod}/{id_pac}")

    if args.brez_kin:
        global MP_AVAILABLE
        MP_AVAILABLE = False

    # ── 1. Poišči datoteke ───────────────────────────────────────────────
    try:
        datoteke = poisci_datoteke_pacienta(args.podatki, id_pac)
    except FileNotFoundError as e:
        print(f"\n[NAPAKA] {e}")
        sys.exit(1)

    print(f"\n  Najdenih videov: {len(datoteke['videi'])}")
    print(f"  Najdenih CSV:    {len(datoteke['csv'])}")

    if not datoteke['videi']:
        print("[NAPAKA] Ni videoposnetkov v mapi pacienta!")
        sys.exit(1)

    # ── 2. Razvrsti v poskuse ────────────────────────────────────────────
    poskusi = razvrsti_v_poskuse(datoteke['videi'])
    print(f"\n  Zaznanih poskusov: {len(poskusi)}")
    for i, p in enumerate(poskusi):
        kamere = [v[2] for v in p]
        cas    = p[0][3] or '?'
        print(f"    Poskus {i+1}: {kamere} @ {cas}")

    # ── 3. Poveži CSV z seansami (po datumu) ────────────────────────────
    # Vsak CSV ustreza eni seansi. Datum razberemo iz imena datoteke.
    # Npr. patient_024MS202305111413.csv -> datum 20230511
    # Vsaka seansa ima 4 poskuse: P1, P2, S1, S2 (dom, nedom, dom, nedom).
    def datum_iz_imena(ime):
        m = re.search(r'(\d{8})', ime)
        return m.group(1) if m else 'neznan'

    def datum_iz_cas_oznake(oznaka):
        return oznaka[:8] if oznaka and len(oznaka) >= 8 else 'neznan'

    # Mapiraj CSV datoteke po datumu
    csv_po_datumu = {}
    for csv_pot in sorted(datoteke['csv']):
        datum = datum_iz_imena(os.path.basename(csv_pot))
        csv_po_datumu[datum] = csv_pot

    # Grupiramo poskuse po seansi (datumu)
    seanse = {}
    for p in poskusi:
        datum = datum_iz_cas_oznake(p[0][3])
        if datum not in seanse:
            seanse[datum] = []
        seanse[datum].append(p)

    print(f"\n  Seans: {len(seanse)}")
    for datum, ps in seanse.items():
        csv_ime = os.path.basename(csv_po_datumu.get(datum, '—'))
        print(f"    {datum}: {len(ps)} poskusi -> CSV: {csv_ime}")

    # Preberi vse CSV-je
    csv_po_datumu_podatki = {}
    for datum, csv_pot in csv_po_datumu.items():
        try:
            csv_pod = preberi_csv_pacienta(args.podatki, id_pac)
            csv_po_datumu_podatki[datum] = csv_pod
            casi = csv_pod['skupni_casi']
            print(f"  CSV {datum}: " +
                  "  ".join(f"{k}={v:.2f}s" for k, v in casi.items()))
        except Exception as e:
            print(f"  [OPOZORILO] Napaka pri branju CSV {datum}: {e}")

    csv_podatki = next(iter(csv_po_datumu_podatki.values()), None)

    # ── 4. Analiziraj poskuse ────────────────────────────────────────────
    izhod_pac = os.path.join(args.izhod, id_pac)
    os.makedirs(izhod_pac, exist_ok=True)

    poskusi_rezultati = []
    for i, videi_p in enumerate(poskusi):
        if args.samo_poskus is not None and (i + 1) != args.samo_poskus:
            poskusi_rezultati.append(None)
            continue
        izhod_p = os.path.join(izhod_pac, f"poskus{i+1:02d}")
        rez = analiziraj_poskus(videi_p, izhod_p, i)
        poskusi_rezultati.append(rez)

    # ── 5. Primerjaj z referenco (po seansi) ─────────────────────────────
    # Vsaka seansa ima 4 poskuse (P1..S2). Primerjamo znotraj vsake seanse.
    vse_primerjave = []
    for datum, videi_seanse in seanse.items():
        csv_seanse = csv_po_datumu_podatki.get(datum)
        indeksi = [poskusi.index(vp) for vp in videi_seanse]
        rezultati_seanse = [poskusi_rezultati[i] for i in indeksi]
        primerjave_seanse = primerjaj_z_referenco(rezultati_seanse, csv_seanse)
        if primerjave_seanse:
            vse_primerjave.extend(primerjave_seanse)
    primerjave = vse_primerjave


    # ── 6. Izpiši povzetek ───────────────────────────────────────────────
    izpisi_povzetek(id_pac, primerjave, csv_podatki)

    # ── 7. Izvozi grafe in JSON ──────────────────────────────────────────
    izvozi_porocilo(id_pac, primerjave, poskusi_rezultati, csv_podatki, izhod_pac)

    print(f"[KONEC] Analiza zaključena.\n")


if __name__ == '__main__':
    main()