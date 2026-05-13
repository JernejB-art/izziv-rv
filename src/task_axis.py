# task_axis.py
# Samostojna verzija: zaznava dogodkov 9HPT z 1D task-axis projekcijo.
#
# IDEJA:
#   Namesto surovih x/y koordinat ali homografije vzamemo geometrijo naloge:
#   vektor od posodice do luknjic definira "os naloge". Gibanje roke
#   projeciramo na to os → dobimo 1D skalaren signal s(t).
#
#   Interpretation:
#     s malo  → roka blizu posodice
#     s veliko → roka blizu luknjic
#
#   Prednost: deluje pri KATEREMKOLI kotu kamere, brez kalibracije v mm.
#   Omejitev: 1D signal, ne vemo katera točna luknjica je tarča.
#
# INTEGRACIJA Z OBSTOJEČIMI KODAMI:
#   - container_center in holes_center dobimo iz centri_sp/centri_zg
#     (luknjice_led.py) — povprečje centrov obeh področij
#   - hand_pos dobimo iz MediaPipe (detect.py) ali DLC
#
# ARHITEKTURA:
#   TaskOsGeometrija  → definira os, projekcijo, ROI-je
#   TrajektorijaBuffer1D → zbira in gladi 1D signal
#   FSM1D             → zaznava dogodkov na 1D signalu
#   Procesor1D        → glavni loop za procesiranje videa
#
# Zahteve: opencv-python, numpy, scipy, mediapipe
#   pip install mediapipe --break-system-packages

import cv2
import numpy as np
from collections import deque

try:
    from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===== PARAMETRI =====

# ROI radiji (v enotah task-osi, tj. delež razdalje posodica→luknjice)
# 0.0 = posodica, 1.0 = luknjice
CONTAINER_ROI_DELEZ  = 0.25   # roka je "pri posodici" ko s < 0.25
HOLES_ROI_DELEZ      = 0.75   # roka je "pri luknjicah" ko s > 0.75

# Pragovi za FSM
PRAG_MIROVANJE_S_PER_S = 0.05  # "roka miruje" ko |v| < 5% razdalje/s

# Glajenje
SG_OKNO = 11    # Savitzky-Golay okno (liho)
SG_RED  = 2     # red polinoma

# Butterworth filter
BUTTER_CUTOFF_HZ = 3.0   # rezalna frekvenca (Hz)

# Zaznava lokalnega minimuma/maksimuma
PEAK_PROMINENCE = 0.05   # min prominenca vrha (v enotah task-osi)
PEAK_DISTANCE_S = 0.3    # min razdalja med vrhovi (s)


# ===== GEOMETRIJA OSI NALOGE =====

class TaskOsGeometrija:
    """
    Definira os naloge in izvaja projekcijo koordinat roke.

    Os je vektor od centra posodice do centra luknjic.
    Projekcija: s = dot(hand_pos - container_center, v_norm)

    Koordinate so v IMAGE pikslih — ni potrebe po kalibraciji v mm.
    Vse razdalje so normalizirane z dolžino osi (s ∈ [0, 1] med točkama).
    """

    def __init__(self, container_center, holes_center):
        """
        container_center → (x, y) centra posodice v image px
        holes_center     → (x, y) centra luknjic v image px
        """
        self.container_center = np.array(container_center, dtype=np.float64)
        self.holes_center = np.array(holes_center, dtype=np.float64)

        v = self.holes_center - self.container_center
        self.dolzina_px = float(np.linalg.norm(v))

        if self.dolzina_px < 1e-6:
            raise ValueError("container_center in holes_center sta enaki točki!")

        self.v_norm = v / self.dolzina_px   # normaliziran smerni vektor

    def projekcija(self, hand_pos):
        """
        Projecira pozicijo roke na os naloge.

        hand_pos → (x, y) v image px
        vrne → s: float, kjer 0.0 = posodica, 1.0 = luknjice
                  negativno = za posodico, >1.0 = za luknjicami
        """
        p = np.array(hand_pos, dtype=np.float64) - self.container_center
        s_px = float(np.dot(p, self.v_norm))
        return s_px / self.dolzina_px   # normaliziramo na [0, 1]

    def v_image_tocko(self, s):
        """
        Obrne projekcijo: iz s koordinate nazaj v image px.
        Koristno za vizualizacijo.
        """
        return self.container_center + s * self.dolzina_px * self.v_norm

    def je_pri_posodici(self, hand_pos):
        s = self.projekcija(hand_pos)
        return s < CONTAINER_ROI_DELEZ

    def je_pri_luknjicah(self, hand_pos):
        s = self.projekcija(hand_pos)
        return s > HOLES_ROI_DELEZ

    def razdalja_od_posodice_norm(self, hand_pos):
        """Razdalja od posodice (v normaliziranih enotah 0–1)."""
        s = self.projekcija(hand_pos)
        return abs(s)

    def razdalja_od_luknjic_norm(self, hand_pos):
        """Razdalja od luknjic (v normaliziranih enotah 0–1)."""
        s = self.projekcija(hand_pos)
        return abs(s - 1.0)

    def posodobi_iz_centrov(self, centri_posodice, centri_luknjic):
        """
        Posodobi os iz zaznanih centrov luknjic (iz luknjice_led.py).

        centri_posodice → seznam (x,y) centrov neaktivnega področja
        centri_luknjic  → seznam (x,y) centrov aktivnega področja
        """
        if centri_posodice and len(centri_posodice) >= 3:
            pts = np.array(centri_posodice, dtype=np.float64)
            self.container_center = pts.mean(axis=0)

        if centri_luknjic and len(centri_luknjic) >= 3:
            pts = np.array(centri_luknjic, dtype=np.float64)
            self.holes_center = pts.mean(axis=0)

        v = self.holes_center - self.container_center
        self.dolzina_px = float(np.linalg.norm(v))
        if self.dolzina_px > 1e-6:
            self.v_norm = v / self.dolzina_px

    def narisi_os(self, frame, barva=(0, 255, 100), debelina=2):
        """Nariše os naloge na frame."""
        p1 = tuple(self.container_center.astype(int))
        p2 = tuple(self.holes_center.astype(int))

        # Os
        cv2.arrowedLine(frame, p1, p2, barva, debelina, tipLength=0.05)

        # Oznake
        cv2.circle(frame, p1, 8, (255, 150, 0), -1)
        cv2.putText(frame, "posodica", (p1[0]+10, p1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
        cv2.circle(frame, p2, 8, (0, 200, 255), -1)
        cv2.putText(frame, "luknjice", (p2[0]+10, p2[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # ROI cone
        r_pos = int(CONTAINER_ROI_DELEZ * self.dolzina_px)
        r_luk = int((1.0 - HOLES_ROI_DELEZ) * self.dolzina_px)
        cv2.circle(frame, p1, r_pos, (255, 150, 0), 1)
        cv2.circle(frame, p2, r_luk, (0, 200, 255), 1)

        return frame


# ===== 1D TRAJEKTORIJA =====

class TrajektorijaBuffer1D:
    """
    Zbira in gladi 1D projekcijo s(t).
    Računa hitrost ds/dt in zaznava lokalne ekstreme.
    """

    def __init__(self, max_dolzina=600, fps=30.0):
        self.s_raw   = deque(maxlen=max_dolzina)
        self.casi    = deque(maxlen=max_dolzina)
        self.fps     = fps

        # Izračunamo po koncu ali sproti
        self._s_gl  = None   # zglajen signal (numpy array)
        self._vel   = None   # hitrost

    def dodaj(self, s, t):
        self.s_raw.append(float(s))
        self.casi.append(float(t))
        self._s_gl = None   # invalidate cache

    def zglajen(self):
        """Vrne zglajen 1D signal (numpy array)."""
        if self._s_gl is not None:
            return self._s_gl

        arr = np.array(self.s_raw)
        n = len(arr)

        if n < 5:
            self._s_gl = arr
            return arr

        if SCIPY_OK and n >= SG_OKNO:
            okno = SG_OKNO if SG_OKNO % 2 == 1 else SG_OKNO + 1
            try:
                self._s_gl = savgol_filter(arr, okno, SG_RED)
                return self._s_gl
            except ValueError:
                pass

        # Fallback: drseče povprečje
        k = min(7, n)
        self._s_gl = np.convolve(arr, np.ones(k)/k, mode='same')
        return self._s_gl

    def hitrost(self):
        """Vrne hitrost ds/dt (normalizirane enote / s)."""
        sg = self.zglajen()
        casi = np.array(self.casi)
        n = len(sg)
        if n < 3:
            return np.zeros(n)

        dt = np.diff(casi)
        dt[dt == 0] = 1e-6
        vel = np.diff(sg) / dt
        return np.append(vel, vel[-1])   # zadnji element ponovimo

    def zadnji_s(self):
        return self.s_raw[-1] if self.s_raw else None

    def zadnja_hitrost(self):
        v = self.hitrost()
        return float(v[-1]) if len(v) > 0 else 0.0

    def lokalni_min(self, zadnjih_n=None, prominenca=PEAK_PROMINENCE):
        """
        Poišče lokalne minimume v zglajenem signalu.
        Minimum pri posodici → pickup complete.
        """
        sg = self.zglajen()
        if zadnjih_n:
            sg = sg[-zadnjih_n:]

        if not SCIPY_OK or len(sg) < 5:
            return []

        min_razdalja = max(3, int(PEAK_DISTANCE_S * self.fps))
        vrhovi, props = find_peaks(-sg,
                                   prominence=prominenca,
                                   distance=min_razdalja)
        return vrhovi.tolist()

    def lokalni_max(self, zadnjih_n=None, prominenca=PEAK_PROMINENCE):
        """
        Poišče lokalne maksimume v zglajenem signalu.
        Maksimum pri luknjicah → insert complete.
        """
        sg = self.zglajen()
        if zadnjih_n:
            sg = sg[-zadnjih_n:]

        if not SCIPY_OK or len(sg) < 5:
            return []

        min_razdalja = max(3, int(PEAK_DISTANCE_S * self.fps))
        vrhovi, _ = find_peaks(sg,
                                prominence=prominenca,
                                distance=min_razdalja)
        return vrhovi.tolist()

    def zadnji_obrat_smeri(self, okno_n=20):
        """
        Preveri ali je bil v zadnjih okno_n frame-ih obrat smeri (predznak hitrosti).
        Bolj robustno od iskanja eksaktnega vrha.
        """
        v = self.hitrost()
        if len(v) < okno_n + 2:
            return False
        zadnje_v = v[-okno_n:]
        # Iščemo spremembo predznaka
        for i in range(1, len(zadnje_v)):
            if zadnje_v[i-1] > PRAG_MIROVANJE_S_PER_S and zadnje_v[i] < -PRAG_MIROVANJE_S_PER_S:
                return True   # max (positivno → negativno) = vstavitev
            if zadnje_v[i-1] < -PRAG_MIROVANJE_S_PER_S and zadnje_v[i] > PRAG_MIROVANJE_S_PER_S:
                return True   # min (negativno → pozitivno) = dvig
        return False

    def get_vse(self):
        """Vrne (casi, s_raw, s_gl, vel) kot numpy array-e."""
        casi = np.array(self.casi)
        raw  = np.array(self.s_raw)
        gl   = self.zglajen()
        vel  = self.hitrost()
        return casi, raw, gl, vel


# ===== FSM NA 1D SIGNALU =====

class FSMStanje:
    IDLE         = "IDLE"
    MOV_POSODICA = "MOVING_TO_CONTAINER"
    PICKUP       = "PICKUP_PHASE"
    MOV_LUKNJICE = "MOVING_TO_HOLES"
    INSERT       = "INSERT_PHASE"


class PegCikel1D:
    def __init__(self):
        self.pickup_start    = None
        self.pickup_complete = None
        self.insert_start    = None
        self.insert_complete = None
        self.s_pickup        = None   # vrednost s pri dviganju
        self.s_insert        = None   # vrednost s pri vstavljanju

    def zakljuci(self):
        pt = (self.pickup_complete - self.pickup_start
              if self.pickup_start and self.pickup_complete else None)
        it = (self.insert_complete - self.insert_start
              if self.insert_start and self.insert_complete else None)
        mt = (self.insert_start - self.pickup_complete
              if self.pickup_complete and self.insert_start else None)
        return {
            "pickup_start":    self.pickup_start,
            "pickup_complete": self.pickup_complete,
            "insert_start":    self.insert_start,
            "insert_complete": self.insert_complete,
            "pickup_duration": pt,
            "insert_duration": it,
            "movement_time":   mt,
            "s_pickup":        self.s_pickup,
            "s_insert":        self.s_insert,
        }


class FSM1D:
    """
    Končni avtomat za zaznavo pickup/insert dogodkov na 1D task-axis signalu.

    Logika:
      IDLE → ko roka vstopi v container ROI → PICKUP
      PICKUP → ko signal doseže minimum (obrat smeri) → MOV_LUKNJICE
      MOV_LUKNJICE → ko roka vstopi v holes ROI → INSERT
      INSERT → ko signal doseže maksimum (obrat smeri) → IDLE (cikel zaključen)
    """

    def __init__(self, fps=30.0, verbose=True):
        self.fps = fps
        self.verbose = verbose
        self.stanje = FSMStanje.IDLE
        self.cicli = []
        self.trenutni = PegCikel1D()

        # Debounce counterji
        self._pickup_counter    = 0
        self._insert_counter    = 0
        self._exit_pos_counter  = 0
        self._exit_hol_counter  = 0
        self._min_frames        = max(3, int(0.15 * fps))   # 150ms debounce

    def posodobi(self, s, t, vel, traj: TrajektorijaBuffer1D):
        """
        Posodobi FSM z novo vrednostjo projekcije.

        s   → trenutna normalizirana projekcija (0=posodica, 1=luknjice)
        t   → čas (s)
        vel → trenutna hitrost (normalizirane enote/s)
        traj → TrajektorijaBuffer1D za zaznavo obraten smeri
        """

        if self.stanje == FSMStanje.IDLE:
            if s < CONTAINER_ROI_DELEZ:
                self._pickup_counter += 1
            else:
                self._pickup_counter = 0

            if self._pickup_counter >= self._min_frames:
                self.stanje = FSMStanje.PICKUP
                self.trenutni = PegCikel1D()
                self.trenutni.pickup_start = t
                self.trenutni.s_pickup = s
                self._pickup_counter = 0
                self._log(t, "PICKUP_START", f"s={s:.3f}")

        elif self.stanje == FSMStanje.PICKUP:
            self.trenutni.s_pickup = min(self.trenutni.s_pickup or s, s)

            # Pickup complete: obrat smeri (negativno → pozitivno) ali lokalni min
            obrat = traj.zadnji_obrat_smeri(okno_n=int(0.5 * self.fps))
            min_idx = traj.lokalni_min(zadnjih_n=int(1.0 * self.fps))

            if (vel > PRAG_MIROVANJE_S_PER_S and obrat) or len(min_idx) > 0:
                self.stanje = FSMStanje.MOV_LUKNJICE
                self.trenutni.pickup_complete = t
                self._log(t, "PICKUP_COMPLETE", f"s={s:.3f}, vel={vel:.3f}")

            # Fallback: roka zapustila container ROI brez zaznave min
            elif s > CONTAINER_ROI_DELEZ * 1.5:
                self._exit_pos_counter += 1
                if self._exit_pos_counter >= self._min_frames:
                    self.stanje = FSMStanje.MOV_LUKNJICE
                    if self.trenutni.pickup_complete is None:
                        self.trenutni.pickup_complete = t
                    self._exit_pos_counter = 0
                    self._log(t, "PICKUP_COMPLETE (exit ROI)", f"s={s:.3f}")
            else:
                self._exit_pos_counter = 0

        elif self.stanje == FSMStanje.MOV_LUKNJICE:
            if s > HOLES_ROI_DELEZ:
                self._insert_counter += 1
            else:
                self._insert_counter = 0

            if self._insert_counter >= self._min_frames:
                self.stanje = FSMStanje.INSERT
                self.trenutni.insert_start = t
                self.trenutni.s_insert = s
                self._insert_counter = 0
                self._log(t, "INSERT_START", f"s={s:.3f}")

        elif self.stanje == FSMStanje.INSERT:
            self.trenutni.s_insert = max(self.trenutni.s_insert or s, s)

            # Insert complete: obrat smeri (pozitivno → negativno) ali lokalni max
            obrat = traj.zadnji_obrat_smeri(okno_n=int(0.5 * self.fps))
            max_idx = traj.lokalni_max(zadnjih_n=int(1.0 * self.fps))

            if (vel < -PRAG_MIROVANJE_S_PER_S and obrat) or len(max_idx) > 0:
                self.stanje = FSMStanje.IDLE
                self.trenutni.insert_complete = t
                self.cicli.append(self.trenutni.zakljuci())
                self._log(t, f"INSERT_COMPLETE → cikel {len(self.cicli)}", f"s={s:.3f}")
                self.trenutni = PegCikel1D()

            # Fallback: roka zapustila holes ROI
            elif s < HOLES_ROI_DELEZ * 0.8:
                self._exit_hol_counter += 1
                if self._exit_hol_counter >= self._min_frames:
                    self.stanje = FSMStanje.IDLE
                    if self.trenutni.insert_complete is None:
                        self.trenutni.insert_complete = t
                    self.cicli.append(self.trenutni.zakljuci())
                    self._exit_hol_counter = 0
                    self._log(t, f"INSERT_COMPLETE (exit ROI) → cikel {len(self.cicli)}", "")
                    self.trenutni = PegCikel1D()
            else:
                self._exit_hol_counter = 0

        return self.stanje

    def _log(self, t, tip, dodatno=""):
        if self.verbose:
            print(f"  [FSM1D] t={t:6.2f}s | {tip:35s} | {dodatno}")


# ===== SLEDILEC ROKE (MediaPipe) =====

class SledilecRoke1D:
    """Preprosta ovoj za MediaPipe Hands."""

    def __init__(self, min_zaupanje=0.7):
        if not MEDIAPIPE_OK:
            raise ImportError("Namesti: pip install mediapipe --break-system-packages")

        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_zaupanje,
            min_tracking_confidence=0.5,
        )

    def zazaj(self, frame):
        """Vrne (x_px, y_px) konice kazalca ali None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None, None

        lm = res.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        tip = lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return (int(tip.x * w), int(tip.y * h)), lm

    def narisi(self, frame, lm):
        if lm:
            self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def zapri(self):
        self.hands.close()


# ===== VIZUALIZACIJA =====

def narisi_graf_1D(casi, s_raw, s_gl, vel, cicli, fps, izhod_pot):
    """Shrani diagnostični graf: trajektorija, hitrost, dogodki."""
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Zgornji: 1D projekcija
    axs[0].plot(casi, s_raw, color='lightsteelblue', alpha=0.5,
                linewidth=0.8, label='s surovi')
    axs[0].plot(casi, s_gl,  color='steelblue', linewidth=1.5,
                label='s zglajen')
    axs[0].axhline(CONTAINER_ROI_DELEZ, color='orange',
                   linestyle='--', alpha=0.7, label='container ROI')
    axs[0].axhline(HOLES_ROI_DELEZ, color='cyan',
                   linestyle='--', alpha=0.7, label='holes ROI')
    axs[0].set_ylim(-0.2, 1.4)
    axs[0].set_ylabel('s (0=posodica, 1=luknjice)')
    axs[0].set_title('1D Task-Axis Projekcija')
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    # Označi dogodke
    barve = {
        'pickup_start':    ('green',  '^', 'pickup start'),
        'pickup_complete': ('lime',   'v', 'pickup done'),
        'insert_start':    ('red',    '^', 'insert start'),
        'insert_complete': ('orange', 'v', 'insert done'),
    }
    for i, cikel in enumerate(cicli):
        for kljuc, (barva, marker, _) in barve.items():
            t_ev = cikel.get(kljuc)
            if t_ev is not None:
                axs[0].axvline(t_ev, color=barva, alpha=0.6, linewidth=1)
                # Poiščemo s v bližini tega časa
                idx = np.argmin(np.abs(casi - t_ev))
                if idx < len(s_gl):
                    axs[0].plot(t_ev, s_gl[idx], marker=marker,
                                color=barva, markersize=7, zorder=5)

    # Spodnji: hitrost
    axs[1].plot(casi[:len(vel)], vel, color='purple', linewidth=1.0,
                label='hitrost ds/dt')
    axs[1].axhline(0, color='gray', linewidth=0.8)
    axs[1].axhline(PRAG_MIROVANJE_S_PER_S,  color='gray',
                   linestyle=':', alpha=0.6)
    axs[1].axhline(-PRAG_MIROVANJE_S_PER_S, color='gray',
                   linestyle=':', alpha=0.6)
    axs[1].set_ylabel('ds/dt (normalizirano/s)')
    axs[1].set_xlabel('Čas [s]')
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(izhod_pot, dpi=120)
    plt.close()
    print(f"[Graf] Shranjen: {izhod_pot}")


def narisi_overlay_1D(frame, os: TaskOsGeometrija, hand_pos,
                      s_val, fsm_stanje, cicli_n, t_s):
    """Nariše overlay na en frame."""
    out = frame.copy()

    # Os naloge
    os.narisi_os(out)

    # Projekcija roke na os
    if hand_pos is not None:
        hx, hy = int(hand_pos[0]), int(hand_pos[1])
        # Točka na osi
        proj_pt = os.v_image_tocko(s_val)
        px, py = int(proj_pt[0]), int(proj_pt[1])

        cv2.circle(out, (hx, hy), 7, (50, 50, 255), -1)
        cv2.line(out, (hx, hy), (px, py), (200, 200, 200), 1)
        cv2.circle(out, (px, py), 5, (255, 255, 0), -1)
        cv2.putText(out, f"s={s_val:.2f}", (hx+10, hy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # FSM stanje
    barve_fsm = {
        FSMStanje.IDLE:         (150, 150, 150),
        FSMStanje.MOV_POSODICA: (100, 200, 100),
        FSMStanje.PICKUP:       (0,   180, 255),
        FSMStanje.MOV_LUKNJICE: (100, 200, 100),
        FSMStanje.INSERT:       (0,   80,  255),
    }
    b = barve_fsm.get(fsm_stanje, (255, 255, 255))
    cv2.putText(out, f"FSM: {fsm_stanje}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, b, 2)
    cv2.putText(out, f"Cikli: {cicli_n}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(out, f"t={t_s:.1f}s", (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    return out


# ===== GLAVNI PROCESOR =====

class Procesor1D:
    """
    Procesira video z 1D task-axis projekcijo.

    Integrira se z luknjice_led.py za avtomatično inicializacijo osi:
    - Iz centrov LED luknjic določimo container_center in holes_center
    - Ni potrebe po ročni kalibraciji ali homografiji

    Primer:
        p = Procesor1D("video.mp4",
                       container_center=(300, 400),   # ali None za auto
                       holes_center=(800, 400))
        rezultati = p.procesiraj()
    """

    def __init__(self, pot_videa,
                 container_center=None,
                 holes_center=None,
                 izhod_video=None,
                 izhod_graf=None,
                 verbose=True):

        self.pot_videa = pot_videa
        self.izhod_video = izhod_video
        self.izhod_graf  = izhod_graf
        self.verbose     = verbose

        # Os naloge — inicializira se iz parametrov ali avtomatično iz LED
        self._container_center_init = container_center
        self._holes_center_init     = holes_center
        self.os = None   # TaskOsGeometrija, inicializira se med procesiranjem

        self.sledilec = SledilecRoke1D() if MEDIAPIPE_OK else None

        # Bufferji in FSM
        self.traj  = None   # TrajektorijaBuffer1D
        self.fsm   = None   # FSM1D

        if verbose:
            print(f"[Procesor1D] MediaPipe: {'OK' if MEDIAPIPE_OK else 'NI'}")
            print(f"[Procesor1D] SciPy:     {'OK' if SCIPY_OK else 'NI'}")

    def _inicializiraj_os_iz_led(self, centri_zg, centri_sp, aktivno):
        """
        Ko vemo katero področje je aktivno (vstavljanje), inicializiramo os:
        - aktivno področje   = luknjice
        - neaktivno področje = posodica
        """
        if aktivno == 'zgornje':
            centri_luk = centri_zg
            centri_pos = centri_sp
        else:
            centri_luk = centri_sp
            centri_pos = centri_zg

        if len(centri_luk) < 3 or len(centri_pos) < 3:
            return False

        holes_c = np.mean(centri_luk, axis=0)
        cont_c  = np.mean(centri_pos, axis=0)

        try:
            self.os = TaskOsGeometrija(
                container_center=tuple(cont_c),
                holes_center=tuple(holes_c)
            )
            if self.verbose:
                print(f"[Os] Inicializirana iz LED: "
                      f"posodica={tuple(cont_c.astype(int))}, "
                      f"luknjice={tuple(holes_c.astype(int))}")
            return True
        except ValueError:
            return False

    def procesiraj(self):
        """
        Glavni loop.

        Vrne dict:
          'cicli'     → seznam per-peg rezultatov
          'casi'      → numpy array časov
          's_raw'     → numpy array surovih projekcij
          's_gl'      → numpy array zglajenih projekcij
          'vel'       → numpy array hitrosti
          'os'        → TaskOsGeometrija objekt
        """
        # Uvoz luknjice_led samo tu (opcijsko)
        try:
            from luknjice_led import (
                doloci_kamero, izrezi_roi, stej_svetle_luknjice,
                ROI_PARAMETRI, je_obmocje_prizgano
            )
            import os as _os
            kamera = doloci_kamero(_os.path.basename(self.pot_videa))
            roi_param = ROI_PARAMETRI.get(kamera, ROI_PARAMETRI['camP_0'])
            led_ok = True
            if self.verbose:
                print(f"[Procesor1D] LED modul OK, kamera={kamera}")
        except ImportError:
            led_ok = False
            if self.verbose:
                print("[Procesor1D] luknjice_led.py ni na volji — "
                      "os mora biti podana ročno")

        cap = cv2.VideoCapture(self.pot_videa)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inicializiraj os iz ročnih parametrov (fallback)
        if (self._container_center_init is not None and
                self._holes_center_init is not None):
            self.os = TaskOsGeometrija(
                self._container_center_init,
                self._holes_center_init
            )

        self.traj = TrajektorijaBuffer1D(fps=fps)
        self.fsm  = FSM1D(fps=fps, verbose=self.verbose)

        writer = None
        if self.izhod_video:
            writer = cv2.VideoWriter(
                self.izhod_video,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (sirina, visina)
            )

        aktivno_obmocje = None
        os_inicializirana = (self.os is not None)
        frame_idx = 0

        if self.verbose:
            print(f"\n[Procesor1D] Začetek: fps={fps:.1f}, {sirina}x{visina}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_s = frame_idx / fps

            # --- LED zaznava (za inicializacijo osi) ---
            if led_ok and not os_inicializirana:
                roi_zg, _ = izrezi_roi(frame, roi_param['zgornje'])
                roi_sp, _ = izrezi_roi(frame, roi_param['spodnje'])
                n_zg, c_zg_roi = stej_svetle_luknjice(roi_zg)
                n_sp, c_sp_roi = stej_svetle_luknjice(roi_sp)

                x1z = int(sirina * roi_param['zgornje'][0])
                y1z = int(visina * roi_param['zgornje'][1])
                x1s = int(sirina * roi_param['spodnje'][0])
                y1s = int(visina * roi_param['spodnje'][1])
                c_zg = [(x+x1z, y+y1z) for x,y in c_zg_roi]
                c_sp = [(x+x1s, y+y1s) for x,y in c_sp_roi]

                zg_on = je_obmocje_prizgano(n_zg)
                sp_on = je_obmocje_prizgano(n_sp)

                if zg_on and not sp_on:
                    aktivno_obmocje = 'zgornje'
                elif sp_on and not zg_on:
                    aktivno_obmocje = 'spodnje'

                if aktivno_obmocje:
                    ok = self._inicializiraj_os_iz_led(c_zg, c_sp, aktivno_obmocje)
                    if ok:
                        os_inicializirana = True

            # --- Sledenje roke ---
            hand_pos = None
            lm = None
            s_val = 0.0

            if self.sledilec:
                hand_pos, lm = self.sledilec.zazaj(frame)

            if hand_pos and self.os:
                s_val = self.os.projekcija(hand_pos)
                self.traj.dodaj(s_val, t_s)
                vel = self.traj.zadnja_hitrost()
                self.fsm.posodobi(s_val, t_s, vel, self.traj)

            # --- Overlay ---
            if writer is not None:
                if self.sledilec and lm:
                    frame = self.sledilec.narisi(frame, lm)
                if self.os:
                    frame = narisi_overlay_1D(
                        frame, self.os, hand_pos,
                        s_val, self.fsm.stanje,
                        len(self.fsm.cicli), t_s
                    )
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if self.sledilec:
            self.sledilec.zapri()

        casi, s_raw, s_gl, vel = self.traj.get_vse()

        if self.verbose:
            self._izpis(casi, s_raw, s_gl, vel)

        if self.izhod_graf and len(casi) > 0:
            narisi_graf_1D(casi, s_raw, s_gl, vel,
                           self.fsm.cicli, fps, self.izhod_graf)

        return {
            'cicli':  self.fsm.cicli,
            'casi':   casi,
            's_raw':  s_raw,
            's_gl':   s_gl,
            'vel':    vel,
            'os':     self.os,
        }

    def _izpis(self, casi, s_raw, s_gl, vel):
        print(f"\n=== REZULTATI 1D TASK-AXIS ===")
        print(f"Skupaj ciklov: {len(self.fsm.cicli)}")
        for i, c in enumerate(self.fsm.cicli):
            print(f"\nCikel {i+1}:")
            print(f"  pickup:    {c.get('pickup_start',0):.2f}s → "
                  f"{c.get('pickup_complete',0):.2f}s "
                  f"({c.get('pickup_duration',0):.3f}s)")
            print(f"  premik:    {c.get('movement_time',0):.3f}s")
            print(f"  vstavitev: {c.get('insert_start',0):.2f}s → "
                  f"{c.get('insert_complete',0):.2f}s "
                  f"({c.get('insert_duration',0):.3f}s)")
        print(f"==============================\n")


# ===== POST-PROCESIRANJE CELOTNEGA VIDEA =====

def analiziraj_z_osjo(casi_arr, s_arr, fps, verbose=True):
    """
    Post-procesna analiza že zbranega 1D signala.
    Koristno ko imaš trajektorijo že shranjeno (npr. iz DLC ali CSV).

    casi_arr → numpy array časov (s)
    s_arr    → numpy array projekcij (normaliziran na [0,1])
    fps      → FPS videa
    vrne → dict z cicli in signali
    """
    traj = TrajektorijaBuffer1D(fps=fps)
    fsm  = FSM1D(fps=fps, verbose=verbose)

    for t, s in zip(casi_arr, s_arr):
        traj.dodaj(float(s), float(t))
        vel = traj.zadnja_hitrost()
        fsm.posodobi(float(s), float(t), vel, traj)

    casi, s_raw, s_gl, vel = traj.get_vse()
    return {
        'cicli': fsm.cicli,
        'casi':  casi,
        's_raw': s_raw,
        's_gl':  s_gl,
        'vel':   vel,
    }


# ===== VSTOPNA TOČKA =====

if __name__ == "__main__":
    import sys

    pot = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/Data/patient_024/patient_024camP_0_20230511_14_11_19.mp4"

    p = Procesor1D(
        pot_videa=pot,
        # container_center in holes_center = None → avtomatično iz LED luknjic
        izhod_video='/workspace/results/task_axis_debug.mp4',
        izhod_graf='/workspace/results/task_axis_graf.png',
        verbose=True,
    )

    rezultati = p.procesiraj()

    print(f"\nSkupaj ciklov: {len(rezultati['cicli'])}")
    for i, c in enumerate(rezultati['cicli']):
        print(f"  Cikel {i+1}: "
              f"pickup={c.get('pickup_duration',0):.3f}s, "
              f"premik={c.get('movement_time',0):.3f}s, "
              f"insert={c.get('insert_duration',0):.3f}s")