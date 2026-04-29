# detect.py
# Skripta s CV prebere video posnetek, z MediaPipe zazna roko v vsakem frame-u,
# nariše skelet roke (21 točk in povezave) ter shrani rezultat kot nov video.

import cv2
import mediapipe as mp

# Orodja za zaznavo rok in risanje točk ter povezav
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def analiziraj_video(vhod, izhod):
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

    # Zaženi detektor rok in procesiraj frame po frame
    # mp_hands.Hands(static_image_mode, max_num_hands) -> detektor rok
    #   static_image_mode=False -> optimiziran za video (sledenje med framji)
    #   max_num_hands=1         -> išče največ eno roko
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
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
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Zapiši frame v izhodni video
            writer.write(frame)
    
    # Sprosti vire
    cap.release()
    writer.release()
    print("Končano!")

if __name__ == "__main__":
    analiziraj_video("/data/test.mp4", "/workspace/results/output.mp4")