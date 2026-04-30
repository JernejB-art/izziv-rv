# detect.py
# Skripta s CV prebere video posnetek, z MediaPipe zazna roko v vsakem frame-u,
# nariše skelet roke (21 točk in povezave),
# izračuna kinematične parametre (d, v, a) in
# zazna faze testa (pobiranje/vstavljanje zatičev). Rezultate shrani kot video in graf.

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from kinematics import izracun_center_roke, izracun_kinematika, zaznava_faze_testa

# Orodja za zaznavo rok in risanje točk ter povezav
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# Orodje za barvni slog točk in povezav
mp_styles = mp.solutions.drawing_styles

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

    # Seznam za shranjevanje pozicij centra roke skozi čas
    pozicije = []

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

            # Če je roka zaznana, nariši skelet na frame
            # mp_draw.draw_landmarks(slika, landmarks, povezave)
            #   landmarks        -> 21 točk roke v normaliziranih koordinatah
            #   HAND_CONNECTIONS -> seznam parov točk, ki definirajo kosti roke
            if rezultat.multi_hand_landmarks:
                for landmarks in rezultat.multi_hand_landmarks:
                    # Nariši skelet roke z barvnim slogom
                    # get_default_hand_landmarks_style()  -> rdeče pike za posamezne točke roke
                    # get_default_hand_connections_style() -> zelene črte med točkami (kosti roke)
                    mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # Izračunaj center roke iz zapestja in MCP sklepov
                    cx, cy = izracun_center_roke(landmarks, sirina, visina)
                    pozicije.append((cx, cy))

                    # Nariši center roke na frame
                    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 255), -1)

            # Zapiši frame v izhodni video
            writer.write(frame)
    
    # Sprosti vire
    cap.release()
    writer.release()

    # Izračunaj kinematiko in zaznaj faze testa
    if len(pozicije) > 20:
        kin = izracun_kinematika(pozicije, fps)
        faze = zaznava_faze_testa(kin["pozicije"], fps)

        print(f"Število pobiranje zatičev: {faze['stevilo_pobiranje']}")
        print(f"Število vstavljanje zatičev: {faze['stevilo_vstavljanje']}")

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
    analiza_video("/data/Data/patient_183/patient_183camP_2_20231116_14_10_55.mp4",
        "/workspace/results/output.mp4",
        "/workspace/results/kinematika.png"
        )