# detect.py
# Skripta s CV prebere video posnetek, z MediaPipe zazna roko v vsakem frame-u,
# nariše skelet roke (21 točk in povezave),
# izračuna kinematične parametre (d, v, a) in
# zazna faze testa (pobiranje/vstavljanje zatičev). Rezultate shrani kot video in graf.

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from kinematics import izracun_center_roke, izracun_kazalec, izracun_kinematika, zaznava_faze_testa

# Orodja za zaznavo rok in risanje točk ter povezav
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# Orodje za barvni slog točk in povezav
mp_styles = mp.solutions.drawing_styles

# Minimalna zanesljivost zaznave roke — frame z nižjo zanesljivostjo se preskoči
PRAG_ZANESLJIVOSTI = 0.7

def analiza_video(vhod, izhod, izhod_graf):
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

    # Zaženi detektor rok z nastavitvami
    # static_image_mode=False      -> optimizirano za video (sledenje med framji)
    # max_num_hands=1              -> išče samo eno roko
    # model_complexity=1           -> bolj natančen model (0=hitrejši, 1=natančnejši)
    # min_detection_confidence=0.5 -> minimalna zanesljivost za zaznavo roke (0.0-1.0)
    # min_tracking_confidence=0.5  -> minimalna zanesljivost za sledenje roke (0.0-1.0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # Zanka se vrti do konca videa
        while cap.isOpened():

            # Preberi naslednji frame
            # cap.read() -> (ret, frame): ret=True če je frame uspešno prebran
            ret, frame = cap.read()
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
                for landmarks, handedness in zip(rezultat.multi_hand_landmarks, rezultat.multi_handedness):

                    # Preveri zanesljivost zaznave — preskoči nezanesljive frame-e
                    # handedness.classification[0].score -> verjetnost pravilne zaznave (0.0-1.0)
                    zanesljivost = handedness.classification[0].score
                    if zanesljivost < PRAG_ZANESLJIVOSTI:
                        continue

                    # Nariši skelet roke z barvnim slogom
                    # get_default_hand_landmarks_style()   -> barvne pike za točke roke
                    # get_default_hand_connections_style() -> barvne črte med točkami
                    mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # Izračunaj center roke iz zapestja in MCP sklepov
                    cx, cy = izracun_center_roke(landmarks, sirina, visina)
                    pozicije_roka.append((cx, cy))

                    # Izračunaj konico kazalca za zaznavo faz
                    kx, ky = izracun_kazalec(landmarks, sirina, visina)
                    pozicije_kazalec.append((kx, ky))

                    # Nariši center roke (rumena) in konico kazalca (vijolična)
                    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)

            # Zapiši frame v izhodni video
            writer.write(frame)

    # Sprosti vire
    cap.release()
    writer.release()

    # Izračunaj kinematiko in zaznaj faze testa
    if len(pozicije_roka) > 20:
        kin = izracun_kinematika(pozicije_roka, fps)
        faze = zaznava_faze_testa(np.array(pozicije_kazalec), fps)

        print(f"Število pobiranje zatičev: {faze['stevilo_pobiranje']}")
        print(f"Število vstavljanje zatičev: {faze['stevilo_vstavljanje']}")

        # Diagnostični graf faz
        plt.figure()
        plt.plot(faze["gibanje_1d"])
        plt.axhline(faze["sredina"], color="red", linestyle="--", label="sredina")
        plt.scatter(faze["vrhovi_idx"], faze["gibanje_1d"][faze["vrhovi_idx"]], color="green", label="pobiranje")
        plt.scatter(faze["doline_idx"], faze["gibanje_1d"][faze["doline_idx"]], color="blue", label="vstavljanje")
        plt.legend()
        plt.savefig("/workspace/results/faze.png")

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

    print("Končano!")

if __name__ == "__main__":
    analiza_video(
        "/data/Data/patient_183/patient_183camP_0_20231116_14_10_55.mp4",
        "/workspace/results/output.mp4",
        "/workspace/results/kinematika.png"
    )