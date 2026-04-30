import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

frame = cv2.imread('/workspace/results/patient_088_posodica.png')
siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

_, maska = cv2.threshold(siva, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Otsu prag: {_:.0f}')
jedro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, jedro)
maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, jedro)

konture, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centri = []
for k in konture:
    p = cv2.contourArea(k)
    if 8 < p < 2000:
        obseg = cv2.arcLength(k, True)
        if obseg == 0:
            continue
        if 4 * np.pi * p / (obseg ** 2) < 0.55:
            continue
        M = cv2.moments(k)
        if M['m00'] > 0:
            centri.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])

print(f'Zaznanih: {len(centri)}')

if len(centri) >= 10:
    tocke = np.array(centri)

    db = DBSCAN(eps=50, min_samples=3).fit(tocke)
    # Ohrani samo gruče z vsaj 6 točkami (prave 3x3 skupini imata ~9)
    from collections import Counter
    stevilo = Counter(db.labels_)
    velike_gruce = {l for l, n in stevilo.items() if l != -1 and n >= 6}
    tocke = tocke[np.array([l in velike_gruce for l in db.labels_])]
    print(f'Po DBSCAN filtriranju: {len(tocke)} točk')

    if len(tocke) >= 10:
        pca = PCA(n_components=1)
        projekcije = pca.fit_transform(tocke).flatten()

        sorted_idx = np.argsort(projekcije)
        sorted_proj = projekcije[sorted_idx]
        vrzeli = np.diff(sorted_proj)

        n = len(sorted_proj)
        kandidati = [i for i in range(1, n) if abs(i - n//2) <= 2]
        rez = max(kandidati, key=lambda i: vrzeli[i-1])

        labels = np.zeros(n, dtype=int)
        labels[sorted_idx[rez:]] = 1

        print(f'Skupina 0: {np.sum(labels==0)} točk, Skupina 1: {np.sum(labels==1)} točk')

        vse_luknjice = []
        for skupina_id in [0, 1]:
            tocke_skupine = tocke[labels == skupina_id]
            centroid = tocke_skupine.mean(axis=0)
            razdalje = np.linalg.norm(tocke_skupine - centroid, axis=1)
            luknjice_skupine = tocke_skupine[np.argsort(razdalje)[:9]]
            vse_luknjice.append(luknjice_skupine)
            for t in luknjice_skupine:
                cv2.circle(frame, (int(t[0]), int(t[1])), 8, (0, 255, 0), 2)

        centroidi = np.array([l.mean(axis=0) for l in vse_luknjice])
        cx_pos = int(np.mean(centroidi[:, 0]))
        cy_pos = int(np.mean(centroidi[:, 1]))

        # Polmer = stranica 3x3 kvadrata / 2
        skupina0 = vse_luknjice[0]
        razpon_x = skupina0[:, 0].max() - skupina0[:, 0].min()
        razpon_y = skupina0[:, 1].max() - skupina0[:, 1].min()
        stranica = max(razpon_x, razpon_y)
        polmer = int(stranica * 0.75)

        print(f'Posodica: center=({cx_pos},{cy_pos}), polmer={polmer}')
        cv2.circle(frame, (cx_pos, cy_pos), polmer, (255, 100, 0), 3)
        cv2.circle(frame, (cx_pos, cy_pos), 6, (0, 255, 255), -1)

cv2.imwrite('/workspace/results/luknjice_bele.png', frame)
print('Shranjeno!')