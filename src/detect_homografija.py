# detect_homografija.py
# Integracija homografije v obstoječ pipeline detect.py + luknjice_led.py
#
# Nadgradnja obstoječega detect.py:
#   - Vse koordinate roke sedaj v mm (board prostor)
#   - ROI radiji v mm (neodvisni od kamere)
#   - FSM logika temelji na mm razdalji, ne px koordinatah
#   - Podpira vse kote kamere (cam_P0, cam_P1, cam_P2)
#
# Zahteve: OpenCV, MediaPipe, NumPy, SciPy
#   pip install mediapipe --break-system-packages
#
# Struktura izhoda za vsak peg cikel:
#   {
#     "pickup_start":   float (s),
#     "pickup_complete":float (s),
#     "insert_start":   float (s),
#     "insert_complete":float (s),
#     "movement_time":  float (s),
#     "pickup_duration":float (s),
#     "insert_duration":float (s),
#     "trajectory_mm":  [(x,y,t), ...],
#     "velocity_mm_per_s": [float, ...],
#   }

import cv2
import numpy as np
import time
from collections import deque

try:
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

from homografija import BoardHomografija, homografija_iz_luknjic_led
from luknjice_led import (
    doloci_kamero, izrezi_roi, stej_svetle_luknjice,
    ROI_PARAMETRI, je_obmocje_prizgano
)


# ===== PARAMETRI =====

# Razmik med luknjicami (mm) — nastavi glede na tvojo ploščico
RAZMIK_MM = 32  # ali 50

# Glajenje trajektorije (Savitzky-Golay)
SG_OKNO = 11    # okno v frame-ih (mora biti liho)
SG_RED  = 2     # red polinoma

# FSM pragovi (v mm)
PRAG_PICKUP_ENTRY_MM    = 45.0   # vstop v posodico (ROI = 40mm + buffer)
PRAG_PICKUP_EXIT_MM     = 50.0   # izhod iz posodice
PRAG_INSERT_ENTRY_MM    = 15.0   # vstop v ROI luknjice
PRAG_MIN_HITROST_MM_S   = 5.0    # "mirovanje" — zaznava dviga/vstavljanja

# Časovni okni za zaznavo lokalnega minimuma (s)
OKNO_MIN_S = 0.4   # 400ms okno za detekcijo obrata smeri


# ===== SLEDENJE ROKE Z MEDIAPIPE =====

class SledilecRoke:
    """
    Sledilec roke z MediaPipe.
    Vrača pozicijo konice kazalca ali centra roke v image koordinatah.
    """

    def __init__(self, min_zaupanje=0.7):
        if not MEDIAPIPE_OK:
            raise ImportError("MediaPipe ni nameščen. Zaženi: pip install mediapipe")

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_zaupanje,
            min_tracking_confidence=0.5,
        )

    def zazaj(self, frame):
        """
        Zazna roko v frame-u.
        Vrne (x_px, y_px) konice kazalca ali None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rezultat = self.hands.process(rgb)

        if not rezultat.multi_hand_landmarks:
            return None, None

        landmarks = rezultat.multi_hand_landmarks[0]
        h, w = frame.shape[:2]

        # Konica kazalca = landmark 8
        kazalec = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x_px = int(kazalec.x * w)
        y_px = int(kazalec.y * h)

        return (x_px, y_px), landmarks

    def narisi(self, frame, landmarks):
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        return frame

    def zapri(self):
        self.hands.close()


# ===== TRAJEKTORIJA =====

class TrajektorijaBuffer:
    """
    Hrani zgodovino pozicij roke v mm koordinatah.
    Aplicira Savitzky-Golay glajenje za zmanjšanje jitterja.
    """

    def __init__(self, max_dolzina=300):
        self.xs_mm = deque(maxlen=max_dolzina)
        self.ys_mm = deque(maxlen=max_dolzina)
        self.casi = deque(maxlen=max_dolzina)

    def dodaj(self, x_mm, y_mm, t_s):
        self.xs_mm.append(x_mm)
        self.ys_mm.append(y_mm)
        self.casi.append(t_s)

    def zadnja_mm(self):
        """Vrne zadnjo zaznano pozicijo (brez glajenja)."""
        if not self.xs_mm:
            return None
        return (self.xs_mm[-1], self.ys_mm[-1])

    def zglajena_mm(self):
        """Vrne zglajen signal (Savitzky-Golay ali moving average)."""
        n = len(self.xs_mm)
        if n < 5:
            return list(self.xs_mm), list(self.ys_mm)

        xs = np.array(self.xs_mm)
        ys = np.array(self.ys_mm)

        if SCIPY_OK and n >= SG_OKNO:
            okno = SG_OKNO if (SG_OKNO % 2 == 1) else SG_OKNO + 1
            try:
                xs_gl = savgol_filter(xs, okno, SG_RED)
                ys_gl = savgol_filter(ys, okno, SG_RED)
                return xs_gl.tolist(), ys_gl.tolist()
            except ValueError:
                pass

        # Fallback: drseče povprečje
        okno = min(7, n)
        kernel = np.ones(okno) / okno
        xs_gl = np.convolve(xs, kernel, mode='same')
        ys_gl = np.convolve(ys, kernel, mode='same')
        return xs_gl.tolist(), ys_gl.tolist()

    def hitrost_mm_s(self):
        """Izračuna hitrost (mm/s) iz zglajene trajektorije."""
        n = len(self.casi)
        if n < 3:
            return []

        xs_gl, ys_gl = self.zglajena_mm()
        casi = list(self.casi)

        hitrosti = [0.0]
        for i in range(1, n):
            dt = casi[i] - casi[i-1]
            if dt <= 0:
                hitrosti.append(0.0)
                continue
            dx = xs_gl[i] - xs_gl[i-1]
            dy = ys_gl[i] - ys_gl[i-1]
            v = np.sqrt(dx**2 + dy**2) / dt
            hitrosti.append(float(v))

        return hitrosti

    def lokalni_minimum_razdalje(self, center_mm, okno_s=OKNO_MIN_S, fps=30):
        """
        Preveri ali je bil v zadnjem oknu okno_s sekund lokalni minimum
        razdalje od center_mm (obrat smeri).

        Vrne True če je bil lokalni minimum (= dokončanje dviga ali vstavljanja).
        """
        n = len(self.xs_mm)
        okno_f = int(okno_s * fps)
        if n < okno_f + 2:
            return False

        xs = list(self.xs_mm)[-okno_f:]
        ys = list(self.ys_mm)[-okno_f:]
        razdalje = [np.sqrt((x - center_mm[0])**2 + (y - center_mm[1])**2)
                    for x, y in zip(xs, ys)]

        if not SCIPY_OK:
            # Ročna zaznava minimuma
            sr = len(razdalje) // 2
            return (razdalje[sr] < razdalje[0] and razdalje[sr] < razdalje[-1])

        vrhovi, _ = find_peaks([-r for r in razdalje], prominence=2.0)
        return len(vrhovi) > 0

    def get_zadnjih_n_mm(self, n):
        """Vrne zadnjih N točk kot seznam (x,y) tuples."""
        xs = list(self.xs_mm)[-n:]
        ys = list(self.ys_mm)[-n:]
        return list(zip(xs, ys))


# ===== FSM STANJE =====

class FSMStanje:
    IDLE             = "IDLE"
    MOV_POSODICA     = "MOVING_TO_CONTAINER"
    PICKUP           = "PICKUP_PHASE"
    MOV_LUKNJICE     = "MOVING_TO_HOLES"
    INSERT           = "INSERT_PHASE"


class PegCikel:
    """Podatki o enem ciklu (dvig + vstavitev zatiča)."""
    def __init__(self):
        self.pickup_start    = None
        self.pickup_complete = None
        self.insert_start    = None
        self.insert_complete = None
        self.target_hole_idx = None
        self.trajektorija_mm = []

    def zakljuci(self):
        return {
            "pickup_start":    self.pickup_start,
            "pickup_complete": self.pickup_complete,
            "insert_start":    self.insert_start,
            "insert_complete": self.insert_complete,
            "movement_time":   (self.insert_start - self.pickup_complete
                                if self.pickup_complete and self.insert_start else None),
            "pickup_duration": (self.pickup_complete - self.pickup_start
                                if self.pickup_start and self.pickup_complete else None),
            "insert_duration": (self.insert_complete - self.insert_start
                                if self.insert_start and self.insert_complete else None),
            "target_hole":     self.target_hole_idx,
        }


# ===== GLAVNI PROCESOR =====

class Procesor9HPT:
    """
    Glavni razred ki poveže vse module.

    Primer uporabe:
        p = Procesor9HPT("videa/test.mp4")
        rezultati = p.procesiraj()
    """

    def __init__(self, pot_videa, razmik_mm=RAZMIK_MM,
                 izhod_video=None, izhod_graf=None, verbose=True):
        self.pot_videa = pot_videa
        self.razmik_mm = razmik_mm
        self.izhod_video = izhod_video
        self.izhod_graf = izhod_graf
        self.verbose = verbose

        # Določi kamero iz imena datoteke
        import os
        self.kamera = doloci_kamero(os.path.basename(pot_videa))
        self.roi_param = ROI_PARAMETRI.get(self.kamera, ROI_PARAMETRI['camP_0'])

        # Objekti
        self.homografija = BoardHomografija(razmik_mm=razmik_mm)
        self.trajektorija = TrajektorijaBuffer()
        self.sledilec = SledilecRoke() if MEDIAPIPE_OK else None

        # FSM
        self.fsm = FSMStanje.IDLE
        self.cicli = []
        self.trenutni_cikel = None

        # Kalibracija posodice (določimo med prvim blink signalom)
        self._center_posodice_mm = None

        # Homografija se inicializira iz prve zaznave luknjic
        self._homografija_inicializirana = False

        if verbose:
            print(f"[Procesor] Kamera: {self.kamera}")
            print(f"[Procesor] Razmik: {razmik_mm}mm")
            print(f"[Procesor] MediaPipe: {'OK' if MEDIAPIPE_OK else 'NI NAMEŠČEN'}")

    def _posodobi_homografijo(self, centri_zg, centri_sp):
        """
        Poskusi inicializirati/posodobiti homografijo iz zaznanih luknjic.
        Prioriteta: aktivno področje (več luknjic).
        """
        # Vzamemo področje z največ točkami
        if len(centri_zg) >= len(centri_sp) and len(centri_zg) >= 4:
            ok = self.homografija.posodobi_iz_luknjic(centri_zg)
        elif len(centri_sp) >= 4:
            ok = self.homografija.posodobi_iz_luknjic(centri_sp)
        else:
            ok = False

        if ok and not self._homografija_inicializirana:
            self._homografija_inicializirana = True
            px_mm = self.homografija.px_per_mm
            if self.verbose:
                print(f"[Homografija] Inicializirana! px/mm = {px_mm:.3f}")

        return ok

    def _posodobi_center_posodice(self, centri_zg, centri_sp, aktivno):
        """
        Ko vemo katero področje je AKTIVNO (vstavljanje), je NEAKTIVNO področje posodica.
        Izračunamo center posodice v mm iz centrov neaktivnega področja.
        """
        if self._center_posodice_mm is not None:
            return  # Že inicializirana

        neaktivni_centri = centri_sp if aktivno == 'zgornje' else centri_zg
        if not neaktivni_centri or len(neaktivni_centri) < 3:
            return

        pts = np.array(neaktivni_centri, dtype=np.float32)
        center_img = pts.mean(axis=0)
        center_mm = self.homografija.v_mm(tuple(center_img))
        if center_mm:
            self._center_posodice_mm = center_mm
            if self.verbose:
                print(f"[Posodica] Center (mm): ({center_mm[0]:.1f}, {center_mm[1]:.1f})")

    def _posodobi_fsm(self, roka_mm, t_s):
        """
        Posodobi FSM glede na pozicijo roke v mm.
        """
        if roka_mm is None:
            return

        posodica = self._center_posodice_mm
        if posodica is None:
            return  # Ne moremo brez posodice

        d_posodica = np.linalg.norm(
            np.array(roka_mm) - np.array(posodica)
        )

        # Razdalja do najbližje luknjice
        luknjice = self.homografija.get_luknjice_mm()
        razdalje_luk = [
            np.linalg.norm(np.array(roka_mm) - luk)
            for luk in luknjice
        ]
        d_najbljizja_luk = min(razdalje_luk)
        idx_najbljizja = int(np.argmin(razdalje_luk))

        fps = 30.0  # bo prava vrednost iz videa

        if self.fsm == FSMStanje.IDLE:
            # Gibanje proti posodici
            if d_posodica < PRAG_PICKUP_ENTRY_MM:
                self.fsm = FSMStanje.PICKUP
                self.trenutni_cikel = PegCikel()
                self.trenutni_cikel.pickup_start = t_s
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → PICKUP_PHASE (d_pos={d_posodica:.1f}mm)")

        elif self.fsm == FSMStanje.PICKUP:
            if self.trenutni_cikel:
                self.trenutni_cikel.trajektorija_mm.append((*roka_mm, t_s))

            # Dokončanje dviga: lokalni minimum razdalje do posodice
            if self.trajektorija.lokalni_minimum_razdalje(posodica, fps=fps):
                self.fsm = FSMStanje.MOV_LUKNJICE
                self.trenutni_cikel.pickup_complete = t_s
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → MOVING_TO_HOLES (pickup done)")

            # Izhod iz posodice brez min — fallback
            elif d_posodica > PRAG_PICKUP_EXIT_MM:
                self.fsm = FSMStanje.MOV_LUKNJICE
                if self.trenutni_cikel.pickup_complete is None:
                    self.trenutni_cikel.pickup_complete = t_s
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → MOVING_TO_HOLES (exit container)")

        elif self.fsm == FSMStanje.MOV_LUKNJICE:
            if self.trenutni_cikel:
                self.trenutni_cikel.trajektorija_mm.append((*roka_mm, t_s))

            # Vstop v ROI luknjice
            if d_najbljizja_luk < PRAG_INSERT_ENTRY_MM:
                self.fsm = FSMStanje.INSERT
                self.trenutni_cikel.insert_start = t_s
                self.trenutni_cikel.target_hole_idx = idx_najbljizja
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → INSERT_PHASE (luknjica {idx_najbljizja}, "
                          f"d={d_najbljizja_luk:.1f}mm)")

        elif self.fsm == FSMStanje.INSERT:
            if self.trenutni_cikel:
                self.trenutni_cikel.trajektorija_mm.append((*roka_mm, t_s))

            luk_center = tuple(luknjice[idx_najbljizja])

            # Dokončanje vstavljanja: lokalni minimum razdalje do luknjice
            if self.trajektorija.lokalni_minimum_razdalje(luk_center, fps=fps):
                self.fsm = FSMStanje.IDLE
                self.trenutni_cikel.insert_complete = t_s
                self.cicli.append(self.trenutni_cikel.zakljuci())
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → IDLE (cikel {len(self.cicli)} zaključen)")
                self.trenutni_cikel = PegCikel()

            # Fallback: izhod iz luknjice ROI
            elif d_najbljizja_luk > PRAG_INSERT_ENTRY_MM * 2:
                self.fsm = FSMStanje.IDLE
                if self.trenutni_cikel.insert_complete is None:
                    self.trenutni_cikel.insert_complete = t_s
                self.cicli.append(self.trenutni_cikel.zakljuci())
                if self.verbose:
                    print(f"[FSM] t={t_s:.2f}s → IDLE (exit hole ROI)")
                self.trenutni_cikel = PegCikel()

    def procesiraj(self):
        """
        Glavni loop: procesira video, vrne strukturirane rezultate.

        vrne dict z:
          'cicli'        → seznam per-peg rezultatov
          'trajektorija' → celotna trajektorija [(x_mm, y_mm, t_s), ...]
          'hitrosti'     → hitrostni profil [mm/s]
          'px_per_mm'    → kalibracija
          'homografija'  → BoardHomografija objekt
        """
        cap = cv2.VideoCapture(self.pot_videa)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30.0

        writer = None
        if self.izhod_video:
            writer = cv2.VideoWriter(
                self.izhod_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (sirina, visina)
            )

        celotna_trajektorija = []
        frame_idx = 0
        hom_refresh_interval = 30   # Posodobi homografijo vsakih N frame-ov

        # Zaznava stanja LED (za določitev aktivnega področja)
        aktivno_obmocje = None

        if self.verbose:
            print(f"\n[Procesor] Začenjam procesiranje: {self.pot_videa}")
            print(f"[Procesor] FPS: {fps:.1f}, Ločljivost: {sirina}x{visina}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_s = frame_idx / fps

            # 1. Zaznava LED luknjic (vsak frame ali vsakih N frame-ov)
            roi_zg_slika, _ = izrezi_roi(frame, self.roi_param['zgornje'])
            roi_sp_slika, _ = izrezi_roi(frame, self.roi_param['spodnje'])
            n_zg, centri_zg_roi = stej_svetle_luknjice(roi_zg_slika)
            n_sp, centri_sp_roi = stej_svetle_luknjice(roi_sp_slika)

            # Pretvori ROI koordinate v frame koordinate
            x1z = int(sirina * self.roi_param['zgornje'][0])
            y1z = int(visina * self.roi_param['zgornje'][1])
            x1s = int(sirina * self.roi_param['spodnje'][0])
            y1s = int(visina * self.roi_param['spodnje'][1])

            centri_zg = [(x + x1z, y + y1z) for x, y in centri_zg_roi]
            centri_sp = [(x + x1s, y + y1s) for x, y in centri_sp_roi]

            # 2. Posodobi homografijo (iz področja z več luknjicami)
            if frame_idx % hom_refresh_interval == 0 or not self._homografija_inicializirana:
                self._posodobi_homografijo(centri_zg, centri_sp)

            # 3. Določi aktivno področje (za center posodice)
            zg_on = je_obmocje_prizgano(n_zg)
            sp_on = je_obmocje_prizgano(n_sp)
            if zg_on and not sp_on:
                aktivno_obmocje = 'zgornje'
            elif sp_on and not zg_on:
                aktivno_obmocje = 'spodnje'

            if aktivno_obmocje and self.homografija.veljavna:
                self._posodobi_center_posodice(centri_zg, centri_sp, aktivno_obmocje)

            # 4. Sledenje roke
            roka_img = None
            landmarks = None
            if self.sledilec:
                roka_img, landmarks = self.sledilec.zazaj(frame)

            roka_mm = None
            if roka_img and self.homografija.veljavna:
                roka_mm = self.homografija.v_mm(roka_img)
                if roka_mm:
                    self.trajektorija.dodaj(roka_mm[0], roka_mm[1], t_s)
                    celotna_trajektorija.append((*roka_mm, t_s))

            # 5. FSM posodobitev
            if roka_mm and self.homografija.veljavna:
                self._posodobi_fsm(roka_mm, t_s)

            # 6. Vizualni overlay
            if writer is not None:
                frame = self._narisi_overlay(
                    frame, roka_img, roka_mm, landmarks,
                    centri_zg, centri_sp, n_zg, n_sp, t_s
                )
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if self.sledilec:
            self.sledilec.zapri()

        # Postprocesiraj hitrosti
        hitrosti = self.trajektorija.hitrost_mm_s()

        if self.verbose:
            self._izpisi_rezultate()

        if self.izhod_graf:
            self._narisi_graf(celotna_trajektorija, hitrosti)

        return {
            'cicli':           self.cicli,
            'trajektorija_mm': celotna_trajektorija,
            'hitrosti_mm_s':   hitrosti,
            'px_per_mm':       self.homografija.px_per_mm,
            'homografija':     self.homografija,
            'center_posodice_mm': self._center_posodice_mm,
        }

    def _narisi_overlay(self, frame, roka_img, roka_mm, landmarks,
                        centri_zg, centri_sp, n_zg, n_sp, t_s):
        """Nariše debug overlay na frame."""
        out = frame.copy()

        # Skeleton roke
        if self.sledilec and landmarks:
            out = self.sledilec.narisi(out, landmarks)

        # Homografija overlay (luknjice, ROI, posodica)
        zadnjih_mm = self.trajektorija.get_zadnjih_n_mm(50)
        if self.homografija.veljavna:
            out = self.homografija.narisi_debug_overlay(
                out,
                roka_mm=roka_mm,
                pot_mm=zadnjih_mm
            )

        # FSM stanje
        barva_fsm = {
            FSMStanje.IDLE:         (150, 150, 150),
            FSMStanje.MOV_POSODICA: (0, 200, 100),
            FSMStanje.PICKUP:       (0, 180, 255),
            FSMStanje.MOV_LUKNJICE: (0, 200, 100),
            FSMStanje.INSERT:       (0, 80, 255),
        }.get(self.fsm, (255, 255, 255))

        cv2.putText(out, f"FSM: {self.fsm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, barva_fsm, 2)
        cv2.putText(out, f"Cikli: {len(self.cicli)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(out, f"t = {t_s:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if roka_mm:
            cv2.putText(out, f"roka: ({roka_mm[0]:.0f},{roka_mm[1]:.0f})mm",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 1)

        # LED luknjice - zaznane točke
        for cx, cy in centri_zg:
            cv2.circle(out, (cx, cy), 4, (0, 255, 100), -1)
        for cx, cy in centri_sp:
            cv2.circle(out, (cx, cy), 4, (100, 255, 0), -1)

        return out

    def _izpisi_rezultate(self):
        print(f"\n=== REZULTATI DETECT HOMOGRAFIJA ===")
        print(f"Skupaj ciklov: {len(self.cicli)}")
        print(f"px/mm: {self.homografija.px_per_mm}")
        for i, c in enumerate(self.cicli):
            print(f"\nCikel {i+1} (luknjica {c.get('target_hole')}):")
            print(f"  pickup:    {c.get('pickup_start', 0):.2f}s → "
                  f"{c.get('pickup_complete', 0):.2f}s "
                  f"(traja {c.get('pickup_duration', 0):.3f}s)")
            print(f"  premik:    {c.get('movement_time', 0):.3f}s")
            print(f"  vstavitev: {c.get('insert_start', 0):.2f}s → "
                  f"{c.get('insert_complete', 0):.2f}s "
                  f"(traja {c.get('insert_duration', 0):.3f}s)")
        print(f"=====================================\n")

    def _narisi_graf(self, trajektorija, hitrosti):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Graf] matplotlib ni nameščen.")
            return

        if not trajektorija:
            return

        xs = [pt[0] for pt in trajektorija]
        ys = [pt[1] for pt in trajektorija]
        ts = [pt[2] for pt in trajektorija]

        fig, axs = plt.subplots(3, 1, figsize=(14, 10))

        # Trajektorija X in Y
        axs[0].plot(ts, xs, color='steelblue', label='X (mm)')
        axs[0].plot(ts, ys, color='darkorange', label='Y (mm)')
        axs[0].set_title('Trajektorija roke v mm koordinatah')
        axs[0].set_xlabel('Čas [s]')
        axs[0].set_ylabel('Pozicija [mm]')
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # Hitrosti
        if hitrosti and len(hitrosti) == len(ts):
            axs[1].plot(ts, hitrosti, color='green', label='Hitrost (mm/s)')
            axs[1].set_title('Hitrostni profil')
            axs[1].set_xlabel('Čas [s]')
            axs[1].set_ylabel('Hitrost [mm/s]')
            axs[1].legend()
            axs[1].grid(alpha=0.3)

        # 2D trajektorija
        axs[2].plot(xs, ys, color='purple', alpha=0.6, linewidth=0.8)
        axs[2].scatter(xs[0], ys[0], color='green', s=50, zorder=5, label='Start')
        axs[2].scatter(xs[-1], ys[-1], color='red', s=50, zorder=5, label='End')

        # Nariši luknjice
        luknjice = self.homografija.get_luknjice_mm()
        axs[2].scatter(luknjice[:, 0], luknjice[:, 1],
                       color='gold', s=80, zorder=6, marker='o', label='Luknjice')

        # Označimo cicle
        for c in self.cicli:
            if c.get('insert_complete'):
                axs[0].axvline(c['insert_complete'], color='red', alpha=0.4, linewidth=1)
            if c.get('pickup_start'):
                axs[0].axvline(c['pickup_start'], color='green', alpha=0.4, linewidth=1)

        axs[2].set_title('2D trajektorija (board mm koordinate)')
        axs[2].set_xlabel('X [mm]')
        axs[2].set_ylabel('Y [mm]')
        axs[2].legend()
        axs[2].grid(alpha=0.3)
        axs[2].set_aspect('equal')
        axs[2].invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.izhod_graf, dpi=120)
        plt.close()
        print(f"[Graf] Shranjen: {self.izhod_graf}")


# ===== VSTOPNA TOČKA =====

if __name__ == "__main__":
    import sys, os

    pot = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/Data/patient_024/patient_024camP_1_20230511_14_11_19.mp4"

    p = Procesor9HPT(
        pot_videa=pot,
        razmik_mm=RAZMIK_MM,
        izhod_video='/workspace/results/detect_hom_debug.mp4',
        izhod_graf='/workspace/results/detect_hom_graf.png',
        verbose=True
    )

    rezultati = p.procesiraj()

    print("\n=== POVZETEK ===")
    print(f"Skupaj ciklov: {len(rezultati['cicli'])}")
    print(f"px/mm: {rezultati['px_per_mm']}")
    if rezultati['center_posodice_mm']:
        print(f"Center posodice: {rezultati['center_posodice_mm']}")