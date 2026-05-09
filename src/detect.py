# detect.py
# Skripta s CV prebere video posnetek, z MediaPipe zazna roko v vsakem frame-u,
# nariše skelet roke (21 točk in povezave),
# izračuna kinematične parametre (d, v, a) in
# zazna faze testa (pobiranje/vstavljanje zatičev). Rezultate shrani kot video in graf.

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from kinematics import izracun_center_roke, izracun_kazalec, izracun_kinematika, zaznava_faze_testa, zaznava_zaticev, izracun_casov_zaticev, zaznava_prehodov, filtriraj_skoke_kazalec

# Orodja za zaznavo rok in risanje točk ter povezav
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# Orodje za barvni slog točk in povezav
mp_styles = mp.solutions.drawing_styles

# Minimalna zanesljivost zaznave roke — frame z nižjo zanesljivostjo se preskoči
PRAG_ZANESLJIVOSTI = 0.7

def doloci_posodico(frame, ime_datoteke=''):
    """
    Empirično določena lokacija posodice glede na kamero.
    Center in polmer so relativni glede na resolucijo slike.

    ime_datoteke -> string z imenom video datoteke (za zaznavo kamere)
    vrne -> (cx, cy, polmer) v pikslih
    """
    visina, sirina = frame.shape[:2]

    # Parametri: (relativni_x, relativni_y, relativni_polmer)
    parametri = {
        'camP_0': (0.561, 0.544, 0.090),
        'camP_1': (0.571, 0.456, 0.091),
        'camP_2': (0.461, 0.410, 0.090),
    }


    # Razpoznaj kamero iz imena datoteke
    match = re.search(r'(camP_\d+)', ime_datoteke)
    kamera = match.group(1) if match else 'camP_0'

    rx, ry, rr = parametri.get(kamera, parametri['camP_0'])
    cx = int(sirina * rx)
    cy = int(visina * ry)
    polmer = int(sirina * rr)
    return cx, cy, polmer

def analiza_video(vhod, izhod, izhod_graf):
    ime_datoteke = os.path.basename(vhod)
    # Odpri vhodni video
    cap = cv2.VideoCapture(vhod)

    # Preberi lastnosti videa (hitrost v sličicah na sekundo, širina in višina sličice v pikslih)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pripravi zapisovanje izhodnega videa
    # cv2.VideoWriter(pot, codec, fps, (sirina, visina)) -> objekt za pisanje videa
    # cv2.VideoWriter_fourcc(*'mp4v')                   -> mp4 codec
    writer = cv2.VideoWriter(izhod, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sirina, visina))

    # Seznami za shranjevanje pozicij skozi čas
    pozicije_roka = []    # center roke -> za kinematiko
    pozicije_kazalec = [] # konica kazalca -> za zaznavo faz
    posodica = None

    # Zaženi detektor rok z nastavitvami
    # static_image_mode=False      -> optimizirano za video (sledenje med framji)
    # max_num_hands=1              -> išče samo eno roko
    # model_complexity=1           -> bolj natančen model (0=hitrejši, 1=natančnejši)
    # min_detection_confidence=0.5 -> minimalna zanesljivost za zaznavo roke (0.0-1.0)
    # min_tracking_confidence=0.5  -> minimalna zanesljivost za sledenje roke (0.0-1.0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # Zanka se vrti do konca videa
        while cap.isOpened():

            # Preberi naslednji frame
            # cap.read() -> (ret, frame): ret=True če je frame uspešno prebran
            ret, frame = cap.read()
            # Določi posodico iz prvega frame-a (samo enkrat)
            if posodica is None:
                posodica = doloci_posodico(frame, ime_datoteke)
            if not ret:
                break  # Konec videa

            # Pretvori barve BGR -> RGB
            # MediaPipe zahteva RGB, OpenCV privzeto bere BGR
            # cv2.cvtColor(slika, cv2.COLOR_BGR2RGB) -> pretvori barvni prostor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Zazna roko v frameju
            # hands.process(rgb) -> rezultat z .multi_hand_landmarks (seznam zaznanih rok)
            rezultat = hands.process(rgb)

            # Če je roka zaznana, preveri zanesljivost in nariši skelet
            if rezultat.multi_hand_landmarks and rezultat.multi_handedness:

                # Izberi roko z največjo hitrostjo gibanja
                najboljsi_landmarks = None
                max_hitrost = -1


                for landmarks, handedness in zip(rezultat.multi_hand_landmarks, rezultat.multi_handedness):

                    # Preveri zanesljivost zaznave — preskoči nezanesljive frame-e
                    # handedness.classification[0].score -> verjetnost pravilne zaznave (0.0-1.0)
                    zanesljivost = handedness.classification[0].score
                    if zanesljivost < PRAG_ZANESLJIVOSTI:
                        continue

                    # Nariši skelet roke z barvnim slogom
                    # get_default_hand_landmarks_style()   -> barvne pike za točke roke
                    # get_default_hand_connections_style() -> barvne črte med točkami

                    # Izračunaj center roke iz zapestja in MCP sklepov
                    cx, cy = izracun_center_roke(landmarks, sirina, visina)
                    # Izračunaj hitrost glede na prejšnjo pozicijo
                    if len(pozicije_roka) > 0:
                        zadnja = pozicije_roka[-1]
                        hitrost = np.sqrt((cx - zadnja[0])**2 + (cy - zadnja[1])**2)
                    else:
                        hitrost = 0

                    if hitrost > max_hitrost:
                        max_hitrost = hitrost
                        najboljsi_landmarks = landmarks
                        najboljsi_cx, najboljsi_cy = cx, cy

                # Uporabi samo najboljšo roko
                if najboljsi_landmarks is not None:
                    mp_draw.draw_landmarks(
                        frame, najboljsi_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    pozicije_roka.append((najboljsi_cx, najboljsi_cy))
                    kx, ky = izracun_kazalec(najboljsi_landmarks, sirina, visina)
                    pozicije_kazalec.append((kx, ky))
                    cv2.circle(frame, (int(najboljsi_cx), int(najboljsi_cy)), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)

            # Izriši območje posodice na frame
            cx_p, cy_p, r_p = posodica
            cv2.circle(frame, (cx_p, cy_p), r_p, (255, 100, 0), 2)
            cv2.circle(frame, (cx_p, cy_p), 5, (0, 255, 255), -1)
            # Izriši prehode na video (samo če so podatki že na voljo)
            # To deluje samo za debug — v produkciji bi rabili online detekcijo
            frame_idx = len(pozicije_kazalec)  # trenutni frame
            if posodica is not None and len(pozicije_kazalec) > 0:
                kx_now = int(pozicije_kazalec[-1][0]) if pozicije_kazalec else 0
                ky_now = int(pozicije_kazalec[-1][1]) if pozicije_kazalec else 0
                razdalja_now = np.sqrt((kx_now - cx_p)**2 + (ky_now - cy_p)**2)
                
                if razdalja_now < r_p:
                    # Kazalec je ZNOTRAJ posodice — zelena barva kroga
                    cv2.circle(frame, (cx_p, cy_p), r_p, (0, 255, 0), 3)
                    cv2.putText(frame, "V POSODICI", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Kazalec je ZUNAJ posodice — modra barva kroga
                    cv2.circle(frame, (cx_p, cy_p), r_p, (255, 100, 0), 2)
            # Zapiši frame v izhodni video
            writer.write(frame)

    # Sprosti vire
    cap.release()
    writer.release()

    np.save('/workspace/results/pozicije_kazalec.npy', np.array(pozicije_kazalec))

    # Izračunaj kinematiko in zaznaj faze testa
    if len(pozicije_roka) > 20:
        kin = izracun_kinematika(pozicije_roka, fps)
        faze = zaznava_faze_testa(np.array(pozicije_kazalec), fps)

        rezultat_zaticev = zaznava_zaticev(pozicije_kazalec, fps)
        print(f"Število zatičev (Balaceanu metoda): {rezultat_zaticev['stevilo_zaticev']}")

        print(f"x min: {rezultat_zaticev.get('x_kazalec', [0]).min():.0f}, max: {rezultat_zaticev.get('x_kazalec', [0]).max():.0f}, sredina: {rezultat_zaticev.get('sredina', 0):.0f}")
        print(f"Surovi vrhovi: {len(rezultat_zaticev.get('vrhovi_idx', []))}, doline: {len(rezultat_zaticev.get('doline_idx', []))}")
        print(f"Število zatičev (Balaceanu metoda): {rezultat_zaticev['stevilo_zaticev']}")

        casi = izracun_casov_zaticev(faze, kin, fps)

        for i, t in enumerate(casi['casi_postavljanja']):
            print(f"  Zatič {i+1}: {t:.2f}s")


        print(f"Število pobiranje zatičev: {faze['stevilo_pobiranje']}")
        print(f"Število vstavljanje zatičev: {faze['stevilo_vstavljanje']}")

        if len(pozicije_kazalec) > 20 and posodica is not None:
            pozicije_kazalec_clean = filtriraj_skoke_kazalec(pozicije_kazalec, fps)
            prehodi = zaznava_prehodov(pozicije_kazalec_clean, posodica, fps)
            print(f"Število obiskov posodice: {prehodi['stevilo_zaticev']}")
            print(f"Časi pobiranje (v posodici):")
            for i, t in enumerate(prehodi['casi_pobiranje']):
                print(f"  Obisk {i+1}: {t:.2f}s")
            print(f"Časi transporta:")
            for i, t in enumerate(prehodi['casi_transport']):
                print(f"  Transport {i+1}: {t:.2f}s")

        csv_kumulativni = [1.691, 2.852, 3.957, 5.06, 6.503, 7.589, 10.275, 11.39, 13.436]
        csv_skupni = 17.95  # Total S1

        print(f"Skupni čas (algoritem): {casi['skupni_cas']:.2f}s")
        print(f"Referenčni čas CSV P1:  {csv_skupni:.2f}s")
        print(f"Napaka:                 {abs(casi['skupni_cas'] - csv_skupni):.2f}s ({abs(casi['skupni_cas'] - csv_skupni)/csv_skupni*100:.1f}%)")

        csv_posamezni = [csv_kumulativni[0]] + [csv_kumulativni[i] - csv_kumulativni[i-1] for i in range(1,9)]
        print("\nPrimerjava čas posameznega zatiča:")
        print(f"{'Zatič':>6} | {'CSV [s]':>8} | {'Algoritem [s]':>13} | {'Napaka [s]':>10}")
        print("-" * 45)
        for i in range(min(9, len(casi['casi_postavljanja']))):
            napaka = casi['casi_postavljanja'][i] - csv_posamezni[i]
            print(f"{i+1:>6} | {csv_posamezni[i]:>8.2f} | {casi['casi_postavljanja'][i]:>13.2f} | {napaka:>+10.2f}")

        # Diagnostični graf faz
        plt.figure()
        plt.plot(faze["gibanje_1d"])
        plt.axhline(faze["sredina"], color="red", linestyle="--", label="sredina")
        plt.scatter(faze["vrhovi_idx"], faze["gibanje_1d"][faze["vrhovi_idx"]], color="green", label="pobiranje")
        plt.scatter(faze["doline_idx"], faze["gibanje_1d"][faze["doline_idx"]], color="blue", label="vstavljanje")
        plt.legend()
        plt.savefig("/workspace/results/faze.png")
        plt.figure(figsize=(14, 4))
        plt.plot(faze['gibanje_1d'], color='blue', label='Gibanje 1D')
        plt.axhline(faze['sredina'], color='gray', linestyle='--', label='Sredina')
        plt.scatter(faze['vrhovi_idx'], faze['gibanje_1d'][faze['vrhovi_idx']], 
                    color='green', s=100, label='Pobiranje', zorder=5)
        plt.scatter(faze['doline_idx'], faze['gibanje_1d'][faze['doline_idx']], 
                    color='red', s=100, label='Vstavljanje', zorder=5)
        plt.legend()
        plt.xlabel('Frame')
        plt.title('Trajektorija in zaznani eventi')
        plt.tight_layout()
        plt.savefig('/workspace/results/faze_debug.png')
        # Nariši in shrani grafe d/v/a
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle("Kinematični parametri gibanja roke")

        axes[0].plot(kin["cas_pot"], kin["pot"], color="blue")
        axes[0].set_ylabel("Pot [px]")
        axes[0].set_title("Skupna prevožena pot d(t)")

        axes[1].plot(kin["cas_hitrost"], kin["hitrost"], color="green")
        axes[1].set_ylabel("Hitrost [px/s]")
        axes[1].set_title("Hitrost v(t)")

        axes[2].plot(kin["cas_pospesek"], kin["pospesek"], color="red")
        axes[2].set_ylabel("Pospešek [px/s²]")
        axes[2].set_xlabel("Čas [s]")
        axes[2].set_title("Pospešek a(t)")

        plt.tight_layout()
        plt.savefig(izhod_graf)
        print(f"Graf shranjen: {izhod_graf}")


        x = rezultat_zaticev.get('x_kazalec', None)
        if x is not None:
            plt.figure(figsize=(14, 4))
            plt.plot(x, color='blue', label='x kazalec')
            plt.axhline(rezultat_zaticev['sredina'], color='gray', linestyle='--', label='Sredina')
            
            vrhovi = rezultat_zaticev['vrhovi_idx']
            doline = rezultat_zaticev['doline_idx']
            
            if len(vrhovi): plt.scatter(vrhovi, x[vrhovi], color='green', s=100, label='Pobiranje', zorder=5)
            if len(doline): plt.scatter(doline, x[doline], color='red', s=100, label='Vstavljanje', zorder=5)
            
            plt.legend()
            plt.title('Balaceanu zaznava — x trajektorija kazalca')
            plt.savefig('/workspace/results/balaceanu_debug.png')

        # Graf prehodov čez posodico
        if posodica is not None:
            prehodi_x = np.array([d['frame'] for d in prehodi['dogodki']])
            prehodi_tip = np.array([d['tip'] for d in prehodi['dogodki']])
            gibanje = faze['gibanje_1d']
            
            vstopi = prehodi_x[prehodi_tip == 'vstop_posodica']
            izstopi = prehodi_x[prehodi_tip == 'izstop_posodica']
            
            plt.figure(figsize=(14, 4))
            plt.plot(gibanje, color='blue', label='Gibanje 1D')
            if len(vstopi): plt.scatter(vstopi, gibanje[vstopi], 
                                        color='green', s=100, marker='^', label='Vstop posodica', zorder=5)
            if len(izstopi): plt.scatter(izstopi, gibanje[izstopi], 
                                        color='red', s=100, marker='v', label='Izstop posodica', zorder=5)
            plt.legend()
            plt.xlabel('Frame')
            plt.title('Prehodi kazalca čez posodico')
            plt.tight_layout()
            plt.savefig('/workspace/results/prehodi_debug.png')

        for d in prehodi['dogodki']:
            print(f"Frame {d['frame']:4d} | t={d['cas']:.2f}s | {d['tip']}")

    print("Končano!")

if __name__ == "__main__":
    analiza_video(
        "/data/Data/patient_007/patient_007camP_1_20231109_09_35_10.mp4",
        "/workspace/results/output.mp4",
        "/workspace/results/kinematika.png"
    )