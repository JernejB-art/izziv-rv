# stej_luknjice.py
# Prešteje število prenesenih zatičev z merjenjem svetlih luknjic.
#
# Algoritem (po učbeniku Špiclin, poglavje 4 — Preslikave sivin in barv):
#   1. CLAHE (adaptivno izenačevanje histograma) za izboljšanje kontrasta
#   2. Gama korekcija za osvetlitev temnih slik
#   3. Štejemo svetle luknjice na začetku (max) in skozi video (min)
#   4. stevilo_zatičev = max - min

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def izboljsaj_kontrast(frame):
    """
    Izboljša kontrast slike z metodami iz učbenika (pogl. 4).

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
        cv2.createCLAHE(clipLimit, tileGridSize) -> CLAHE objekt
        clahe.apply(siva_slika)                 -> izenačena slika
        clipLimit  = meja za rezanje histograma (prepreči pretirano ojačanje)
        tileGridSize = velikost oken za lokalno izenačevanje

    Gama korekcija (sivinska preslikava iz učbenika):
        f(x) = 255 * (x/255)^(1/gama)
        gama > 1 -> osvetli sliko (koristno za temne posnetke)
        gama < 1 -> potemni sliko
    """
    siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Korak 1: CLAHE — lokalno izenačevanje histograma (poglavje 4.3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva = clahe.apply(siva)

    # Korak 2: Gama korekcija — osvetli temne slike
    gama = 1.5
    tabela = np.array([
        255 * (i / 255) ** (1.0 / gama)
        for i in range(256)
    ], dtype=np.uint8)
    # cv2.LUT(slika, tabela) -> aplicira preslikavo vrednosti (Look-Up Table)
    siva = cv2.LUT(siva, tabela)

    return siva

def stej_luknjice_v_frame(siva):
    """
    Zazna svetle luknjice kot top-N najsvetlejših okroglih območij.
    """
    # Visok prag — samo resnično svetle točke
    _, maska = cv2.threshold(siva, 200, 255, cv2.THRESH_BINARY)
    jedro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, jedro)
    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, jedro)

    konture, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kandidati = []
    for k in konture:
        p = cv2.contourArea(k)
        if 8 < p < 3000:
            obseg = cv2.arcLength(k, True)
            if obseg == 0:
                continue
            if 4 * np.pi * p / (obseg ** 2) < 0.55:
                continue
            # Povprečna svetlost znotraj konture
            maska_k = np.zeros(siva.shape, dtype=np.uint8)
            cv2.drawContours(maska_k, [k], -1, 255, -1)
            srednja_svetlost = cv2.mean(siva, mask=maska_k)[0]
            kandidati.append((srednja_svetlost, k))

    # Razvrsti po svetlosti — najsvetlejše luknjice so prave
    kandidati.sort(key=lambda x: x[0], reverse=True)

    # Vzemi max 18 najsvetlejših (9 luknjic na vsaki strani)
    return len(kandidati[:18])

def analiziraj_luknjice(pot_videa, n_zacetnih_framov=75, n_koncnih_framov=75):
    """
    Analizira video in izračuna število prenesenih zatičev.

    n_zacetnih_framov -> prvih N framov za določitev max luknjic
    n_koncnih_framov  -> zadnjih N framov (alternativa če roka zakriva začetek)

    uniform_filter1d(signal, size) -> drseče povprečje dolžine size
        robustnejše od posameznih framov, zmanjša šum zaznave
    np.percentile(signal, p) -> vrednost pod katero leži p% podatkov
        percentil 5% za min in 85% za max je robustnejše od min/max
    """
    cap = cv2.VideoCapture(pot_videa)

    stevila = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        siva = izboljsaj_kontrast(frame)
        n = stej_luknjice_v_frame(siva)
        stevila.append(n)
        frame_idx += 1

    cap.release()

    stevila = np.array(stevila, dtype=float)

    # Glajenje signala z drsečim povprečjem (okno 15 framov)
    stevila_glajeno = uniform_filter1d(stevila, size=15)

    # Robustni max (percentil 85%) iz začetka in konca videa
    rob_zacetek = stevila_glajeno[:n_zacetnih_framov]
    rob_konec   = stevila_glajeno[-n_koncnih_framov:]
    max_luknjic = int(np.percentile(
        np.concatenate([rob_zacetek, rob_konec]), 85
    ))

    # Robustni min (percentil 5%) — izloči šumne outlierje
    min_luknjic = int(np.percentile(stevila_glajeno, 5))

    stevilo_zaticov = max_luknjic - min_luknjic

    print(f"Max luknjic (prazna plošča): {max_luknjic}")
    print(f"Min luknjic (med testom):    {min_luknjic}")
    print(f"Število prenesenih zatičev:  {stevilo_zaticov}")

    # Graf — surovi in glajeni signal
    plt.figure(figsize=(12, 4))
    plt.plot(stevila, color='lightblue', alpha=0.5, label='Surovi signal')
    plt.plot(stevila_glajeno, color='blue', label='Glajeni signal (okno=15)')
    plt.axhline(max_luknjic, color='green', linestyle='--', label=f'Max={max_luknjic}')
    plt.axhline(min_luknjic, color='red',   linestyle='--', label=f'Min={min_luknjic}')
    plt.xlabel('Frame')
    plt.ylabel('Število luknjic')
    plt.title(f'Zaznane luknjice — prenesenih zatičev: {stevilo_zaticov}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/workspace/results/luknjice_cas.png')
    print("Graf shranjen!")

    return stevilo_zaticov, stevila

def naredi_debug_video(pot_videa, izhod_video='/workspace/results/luknjice_debug.mp4'):
    """
    Naredi video z označenimi zaznanimi luknjicami za vizualni pregled.
    """
    cap = cv2.VideoCapture(pot_videa)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(izhod_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (sirina, visina))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        siva = izboljsaj_kontrast(frame)

        _, maska = cv2.threshold(siva, 150, 255, cv2.THRESH_BINARY)
        jedro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, jedro)
        konture, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stevilo = 0
        for k in konture:
            p = cv2.contourArea(k)
            if 8 < p < 2000:
                obseg = cv2.arcLength(k, True)
                if obseg == 0:
                    continue
                if 4 * np.pi * p / (obseg ** 2) >= 0.55:
                    stevilo += 1
                    # Nariši zaznan krog — zeleno
                    (cx, cy), r = cv2.minEnclosingCircle(k)
                    cv2.circle(frame, (int(cx), int(cy)), int(r) + 3, (0, 255, 0), 2)

        # Izpiši število na frame
        cv2.putText(frame, f'Luknjice: {stevilo}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Debug video shranjen: {izhod_video}")


if __name__ == "__main__":
    analiziraj_luknjice(
        "/data/Data/patient_167/patient_167camP_1_20241107_10_57_29.mp4"
    )
    naredi_debug_video(
        "/data/Data/patient_167/patient_167camP_1_20241107_10_57_29.mp4"
    )