import cv2, os

data_pot = '/data/Data/'
pacienti = sorted(os.listdir(data_pot))[:300]

for pacient in pacienti:
    mapa = os.path.join(data_pot, pacient)
    videi = [f for f in os.listdir(mapa) if f.endswith('.mp4') and 'camP_2' in f]
    if not videi:
        continue
    cap = cv2.VideoCapture(os.path.join(mapa, videi[0]))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    h, w = frame.shape[:2]
    cv2.imwrite(f'/workspace/results/ref_{pacient}.png', frame)
    print(f'{pacient}: {w}x{h}')