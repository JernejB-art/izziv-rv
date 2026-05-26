# detect_combined.py — v3
# Kljucna novost v3:
#   - Integracija StrojStanj9HPT iz luknjice_led.py
#   - FSM pozna fazo testa: VSTAVLJANJE / POSPRAVLJANJE
#   - PICKUP pri posodici v fazi VSTAVLJANJE = pobiranje zatiča za vstavljanje
#   - PICKUP pri luknjicah v fazi POSPRAVLJANJE = pobiranje vstavljenega zatiča
#   - INSERT pri luknjicah v fazi VSTAVLJANJE = vstavitev
#   - INSERT pri posodici v fazi POSPRAVLJANJE = odlaganje nazaj
#   - Rezultati ločeni: cicli_vstavljanje / cicli_pospravljanje

import cv2
import numpy as np
from collections import deque
from enum import Enum

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homografija import BoardHomografija, ustvari_world_tocke_posodica
from task_axis import (
    TaskOsGeometrija, TrajektorijaBuffer1D, FSM1D, FSMStanje,
    CONTAINER_ROI_DELEZ, HOLES_ROI_DELEZ, PRAG_MIROVANJE_S_PER_S,
    narisi_graf_1D
)
from luknjice_led import (
    doloci_kamero, izrezi_roi, stej_svetle_luknjice,
    ROI_PARAMETRI, je_obmocje_prizgano, StrojStanj9HPT,
    postprocesiraj_casovnico
)
from analizator_y import analiziraj_iz_rezultatov
try:
    from overlay import OverlayRenderer
    OVERLAY_OK = True
except ImportError:
    OVERLAY_OK = False

# ===== STRATEGIJA =====

class Strategija(str, Enum):
    HOMOGRAFIJA = "homografija"
    TASK_AXIS   = "task_axis"
    AUTO        = "auto"


# ===== PARAMETRI =====

RAZMIK_MM     = 32
SG_OKNO       = 11
SG_RED        = 2

# ROI pragovi (mm)
PICKUP_ROI_MM = 60.0   # polmer posodice za pobiranje
INSERT_ROI_MM = 20.0   # polmer luknjice za vstavljanje
EXIT_ROI_MM   = 75.0   # izhod iz posodice
EXIT_HOL_MM   = 25.0   # izhod iz luknjice

# Minimalni časi faz (prepreči lažne kratke cikle)
MIN_PICKUP_S  = 0.20
MIN_INSERT_S  = 0.12

DEBOUNCE_S    = 0.12
BLINK_POTRDITEV_FRAMOV = 5

# Landmark: "TIP" = konica kazalca, "MCP" = koren kazalca, "WRIST" = zapestje
LANDMARK_NACIN = "TIP"


# ===== SLEDILEC ROKE =====

# ── ROI MASKE PO KAMERI ──────────────────────────────────────────────────
# Definira izključena območja za vsako kamero (normalizirane koordinate 0-1).
# MediaPipe ne zaznava roke v teh območjih → prepreči detekcijo napačne roke.
# Format: { "ime_kamere": (x_min, y_min, x_max, y_max) }
# Vrednost None = brez omejitve za to kamero.
#
# Območja so določena iz vizualnega pregleda videoposnetkov:
#   camP_0, camS_0 → desnih ~25% slike (roka zunaj plošče)
#   camP_1, camS_1 → desnih ~25% slike
#   camP_2, camS_2 → desnih ~28% slike
#
# Za kalibracijo: nastavi x_max_roi na vrednost kjer se začne nezaželena roka.
# Koordinate so normalizirane (0.0 = levo, 1.0 = desno).

ROI_KAMERE: dict = {
    # x_max     = meja na ZGORNJEM robu slike (normalizirano 0.0-1.0)
    # x_max_bot = meja na SPODNJEM robu slike (privzeto = x_max → navpična črta)
    # Primer poševne črte: x_max=0.75, x_max_bot=0.85 → nagib desno
    "camP_0": {"x_max": 0.95, "x_max_bot": 0.75},
    "camP_1": {"x_max": 0.85, "x_max_bot": 0.85},
    "camP_2": {"x_max": 0.60, "x_max_bot": 0.85},
    "camS_0": {"x_max": 0.92, "x_max_bot": 0.67},
    "camS_1": {"x_max": 0.75, "x_max_bot": 0.75},
    "camS_2": {"x_max": 0.72, "x_max_bot": 0.72},
}

# Privzeta vrednost za nedefinirane kamere (brez omejitve)
ROI_PRIVZETO = {"x_max": 1.0}


class SledilecRoke:
    def __init__(self, min_zaupanje=0.6, ime_kamere=None):
        if not MEDIAPIPE_OK:
            raise ImportError("pip install mediapipe --break-system-packages")
        self.mp_h  = mp.solutions.hands
        self.mp_dr = mp.solutions.drawing_utils
        self.hands = self.mp_h.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=min_zaupanje,
            min_tracking_confidence=0.5)

        # ROI maska za to kamero
        roi = ROI_KAMERE.get(ime_kamere or "", ROI_PRIVZETO)
        self._roi_x_max     = roi.get("x_max", 1.0)
        self._roi_x_max_bot = roi.get("x_max_bot", self._roi_x_max)
        self._roi_x_min     = roi.get("x_min", 0.0)
        self._roi_y_max     = roi.get("y_max", 1.0)
        self._roi_y_min     = roi.get("y_min", 0.0)
        self._ime_kamere    = ime_kamere

    def zazaj(self, frame, nacin=None):
        if nacin is None:
            nacin = LANDMARK_NACIN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm  = res.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        if nacin == "WRIST":
            pt = lm.landmark[self.mp_h.HandLandmark.WRIST]
        elif nacin == "MCP":
            pt = lm.landmark[self.mp_h.HandLandmark.INDEX_FINGER_MCP]
        else:
            pt = lm.landmark[self.mp_h.HandLandmark.INDEX_FINGER_TIP]
        return (int(pt.x * w), int(pt.y * h)), lm

    def _maskiraj_frame(self, frame):
        """
        Počrni izključeno območje slike (desna cona = napačna roka).
        MediaPipe potem ne zaznava rok v tem območju.
        Vrne maskiran frame in originalne dimenzije.
        """
        h, w = frame.shape[:2]
        x_min_px    = int(self._roi_x_min * w)
        x_max_top   = int(self._roi_x_max     * w)
        x_max_bot   = int(self._roi_x_max_bot * w)
        y_min_px    = int(self._roi_y_min * h)
        y_max_px    = int(self._roi_y_max * h)

        # Samo če je ROI < celotna slika
        if (x_max_top >= w and x_max_bot >= w and
                x_min_px <= 0 and y_max_px >= h and y_min_px <= 0):
            return frame, False

        masked = frame.copy()

        # Poševna meja — za vsako vrstico izračunaj x_max
        if x_max_top != x_max_bot:
            for y_px in range(h):
                t = y_px / max(h - 1, 1)
                x_meja_y = int(x_max_top + (x_max_bot - x_max_top) * t)
                if x_meja_y < w:
                    masked[y_px, x_meja_y:] = 0
        else:
            # Navpična meja (hitrejše)
            if x_max_top < w:
                masked[:, x_max_top:] = 0

        if x_min_px > 0:
            masked[:, :x_min_px] = 0
        if y_min_px > 0:
            masked[:y_min_px, :] = 0
        if y_max_px < h:
            masked[y_max_px:, :] = 0
        return masked, True

    def _tocka_v_roi(self, px, py, w, h):
        """Preveri ali je zaznana točka znotraj veljavnega ROI (upošteva poševno mejo)."""
        rel_y = py / h
        # Interpoliraj x_max za to Y pozicijo
        x_max_y = self._roi_x_max + (self._roi_x_max_bot - self._roi_x_max) * rel_y
        rel_x = px / w
        return (self._roi_x_min <= rel_x <= x_max_y and
                self._roi_y_min <= rel_y <= self._roi_y_max)

    def zazaj_multi(self, frame):
        """
        Zaznava vseh treh točk hkrati: kazalec (TIP), palec (THUMB), center (WRIST).
        Pred procesiranjem počrni izključeno območje (ROI maska).
        Vrne (dict { "TIP": (px,py), "THUMB": (px,py), "CENTER": (px,py) }, lm)
        ali (None, None) če roka ni zaznana ali je izven ROI.
        """
        frame_mask, maskirano = self._maskiraj_frame(frame)
        rgb = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        def pt(idx):
            p = lm.landmark[idx]
            return (int(p.x * w), int(p.y * h))
        tocke = {
            "TIP":    pt(8),   # INDEX_FINGER_TIP
            "THUMB":  pt(4),   # THUMB_TIP
            "CENTER": pt(0),   # WRIST
        }
        # Dvojna varnost: zavrni če je WRIST izven ROI
        wrist = tocke["CENTER"]
        if not self._tocka_v_roi(wrist[0], wrist[1], w, h):
            return None, None
        return tocke, lm

    def narisi(self, frame, lm):
        if lm:
            self.mp_dr.draw_landmarks(frame, lm, self.mp_h.HAND_CONNECTIONS)
        return frame

    def zapri(self):
        self.hands.close()


# ===== 2D TRAJEKTORIJA =====

class TrajektorijaBuffer2D:
    def __init__(self, max_dolzina=600, fps=30.0):
        self.xs  = deque(maxlen=max_dolzina)
        self.ys  = deque(maxlen=max_dolzina)
        self.ts  = deque(maxlen=max_dolzina)
        self.fps = fps

    def dodaj(self, x, y, t):
        self.xs.append(float(x))
        self.ys.append(float(y))
        self.ts.append(float(t))

    def zadnja_mm(self):
        return (self.xs[-1], self.ys[-1]) if self.xs else None

    def hitrost_mm_s(self):
        n = len(self.ts)
        if n < 3:
            return 0.0
        xs = np.array(list(self.xs)[-min(n, 15):])
        ys = np.array(list(self.ys)[-min(n, 15):])
        ts = np.array(list(self.ts)[-min(n, 15):])
        if SCIPY_OK and len(xs) >= SG_OKNO:
            try:
                okno = SG_OKNO if SG_OKNO % 2 == 1 else SG_OKNO + 1
                xs = savgol_filter(xs, okno, SG_RED)
                ys = savgol_filter(ys, okno, SG_RED)
            except Exception:
                pass
        dt = ts[-1] - ts[-2] if len(ts) >= 2 else 1 / self.fps
        if dt <= 0:
            return 0.0
        return float(np.sqrt((xs[-1] - xs[-2])**2 + (ys[-1] - ys[-2])**2) / dt)

    def lokalni_min_razdalje(self, center_mm, okno_s=0.8):
        okno_n = max(5, int(okno_s * self.fps))
        n = len(self.xs)
        if n < okno_n + 2:
            return False
        xs = list(self.xs)[-okno_n:]
        ys = list(self.ys)[-okno_n:]
        razd = [np.sqrt((x - center_mm[0])**2 + (y - center_mm[1])**2)
                for x, y in zip(xs, ys)]
        if not SCIPY_OK:
            sr = len(razd) // 2
            return razd[sr] < razd[0] and razd[sr] < razd[-1]
        vrhovi, _ = find_peaks([-r for r in razd], prominence=1.0, distance=3)
        return len(vrhovi) > 0

    def get_zadnjih_n(self, n):
        return list(zip(list(self.xs)[-n:], list(self.ys)[-n:]))

    def get_vse(self):
        return (np.array(list(self.ts)),
                np.array(list(self.xs)),
                np.array(list(self.ys)))


# ===== PEG CIKEL =====

class PegCikel2D:
    def __init__(self, faza="VSTAVLJANJE"):
        self.pickup_start    = None
        self.pickup_complete = None
        self.insert_start    = None
        self.insert_complete = None
        self.target_hole_idx = None
        self.faza            = faza   # "VSTAVLJANJE" ali "POSPRAVLJANJE"

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
            "target_hole":     self.target_hole_idx,
            "faza":            self.faza,
            "metoda":          "homografija",
        }


# ===== FSM HOMOGRAFIJA =====

class FSMHomografija:
    """
    FSM ki pozna fazo testa (VSTAVLJANJE / POSPRAVLJANJE) iz LED stroja.

    V fazi VSTAVLJANJE:
      IDLE → PICKUP (pri posodici) → MOV_LUKNJICE → INSERT (pri luknjici)

    V fazi POSPRAVLJANJE:
      IDLE → PICKUP (pri luknjici) → MOV_POSODICA → INSERT (pri posodici)

    Obe fazi sta simetrični — samo vlogi posodice in luknjic sta zamenjani.
    """

    def __init__(self, hom, fps=30.0, verbose=True):
        self.hom     = hom
        self.fps     = fps
        self.verbose = verbose

        self.stanje  = FSMStanje.IDLE
        self.cicli   = []
        self.trenutni = PegCikel2D()

        self._center_posodice = None
        self._debounce = max(3, int(DEBOUNCE_S * fps))
        self._cnt = 0

        # Faza testa — posodoblja se iz LED stroja
        self.faza_testa = "VSTAVLJANJE"

        # Minimalni časi (prepreči lažne cikle)
        self._pickup_t_start = None
        self._insert_t_start = None

        self._cakajoca_faza = None

    def nastavi_posodico(self, center_mm):
        self._center_posodice = center_mm
        # if self.verbose:
        #     print(f"  [FSM-H] Center posodice: ({center_mm[0]:.1f},{center_mm[1]:.1f})mm")

    def nastavi_fazo(self, faza):
        if faza != self.faza_testa:
            # Preklopi samo ko je FSM v IDLE — ne prekinjaj aktivnega cikla
            if self.stanje == FSMStanje.IDLE:
                # if self.verbose:
                #     print(f"  [FSM-H] Faza: {self.faza_testa} → {faza}")
                self.faza_testa = faza
                self._cnt = 0
            # Sicer shrani zahtevano fazo in preklopi ko dosežemo IDLE
            else:
                self._cakajoca_faza = faza

    def posodobi(self, roka_mm, t_s, traj2d):
        if roka_mm is None or self._center_posodice is None:
            return self.stanje

        cp       = self._center_posodice
        luknjice = self.hom.get_luknjice_mm()

        d_pos = np.linalg.norm(np.array(roka_mm) - np.array(cp))
        dists = [np.linalg.norm(np.array(roka_mm) - l) for l in luknjice]
        d_luk = min(dists)
        idx_luk = int(np.argmin(dists))

        # Glede na fazo določimo kaj je "izvor" (pobiranje) in "cilj" (vstavljanje)
        if self.faza_testa == "VSTAVLJANJE":
            # Pobiranje iz posodice, vstavljanje v luknjico
            d_izvor = d_pos
            d_cilj  = d_luk
            roi_izvor  = PICKUP_ROI_MM
            roi_cilj   = INSERT_ROI_MM
            exit_izvor = EXIT_ROI_MM
            exit_cilj  = EXIT_HOL_MM
            ime_izvor  = "posodica"
            ime_cilj   = f"luk={idx_luk}"
        else:
            # POSPRAVLJANJE: pobiranje iz luknjice, vstavljanje v posodico
            d_izvor = d_luk
            d_cilj  = d_pos
            roi_izvor  = INSERT_ROI_MM   # luknjice so manjše
            roi_cilj   = PICKUP_ROI_MM
            exit_izvor = EXIT_HOL_MM
            exit_cilj  = EXIT_ROI_MM
            ime_izvor  = f"luk={idx_luk}"
            ime_cilj   = "posodica"

        if self.stanje == FSMStanje.IDLE:
            if d_izvor < roi_izvor:
                self._cnt += 1
                if self._cnt >= self._debounce:
                    self.stanje = FSMStanje.PICKUP
                    self.trenutni = PegCikel2D(faza=self.faza_testa)
                    self.trenutni.pickup_start = t_s
                    self._pickup_t_start = t_s
                    self._cnt = 0
                    self._log(t_s, "PICKUP_START", f"d={d_izvor:.1f}mm {ime_izvor}")
            else:
                self._cnt = 0

        elif self.stanje == FSMStanje.PICKUP:
            # Minimalni čas dviga
            if self._pickup_t_start and (t_s - self._pickup_t_start) < MIN_PICKUP_S:
                return self.stanje
            center_izvor = cp if self.faza_testa == "VSTAVLJANJE" else tuple(luknjice[idx_luk])
            if traj2d.lokalni_min_razdalje(center_izvor) or d_izvor > exit_izvor:
                self.stanje = FSMStanje.MOV_LUKNJICE
                self.trenutni.pickup_complete = t_s
                self._log(t_s, "PICKUP_COMPLETE", f"d={d_izvor:.1f}mm")

        elif self.stanje == FSMStanje.MOV_LUKNJICE:
            if d_cilj < roi_cilj:
                self._cnt += 1
                if self._cnt >= self._debounce:
                    self.stanje = FSMStanje.INSERT
                    self.trenutni.insert_start = t_s
                    self._insert_t_start = t_s
                    self.trenutni.target_hole_idx = idx_luk
                    self._cnt = 0
                    self._log(t_s, "INSERT_START", f"d={d_cilj:.1f}mm {ime_cilj}")
            else:
                self._cnt = 0

        elif self.stanje == FSMStanje.INSERT:
            # Minimalni čas vstavljanja
            if self._insert_t_start and (t_s - self._insert_t_start) < MIN_INSERT_S:
                return self.stanje
            center_cilj = tuple(luknjice[idx_luk]) if self.faza_testa == "VSTAVLJANJE" else cp
            if traj2d.lokalni_min_razdalje(center_cilj) or d_cilj > exit_cilj:
                self.stanje = FSMStanje.IDLE
                self.trenutni.insert_complete = t_s
                self.cicli.append(self.trenutni.zakljuci())
                self._log(t_s,
                          f"INSERT_COMPLETE → cikel {len(self.cicli)} [{self.faza_testa}]",
                          f"d={d_cilj:.1f}mm")
                self.trenutni = PegCikel2D(faza=self.faza_testa)

        # Preklopi na čakajočo fazo ko dosežemo IDLE
        if self.stanje == FSMStanje.IDLE and hasattr(self, '_cakajoca_faza') \
                and self._cakajoca_faza and self._cakajoca_faza != self.faza_testa:
            # if self.verbose:
            #     print(f"  [FSM-H] Faza (zakasnjena): {self.faza_testa} → {self._cakajoca_faza}")
            # self.faza_testa = self._cakajoca_faza
            self._cakajoca_faza = None
            self._cnt = 0

        return self.stanje

    def _log(self, t, tip, d=""):
        if self.verbose:
            print(f"  [FSM-H] t={t:6.2f}s | {tip:50s} | {d}")

    @property
    def cicli_vstavljanje(self):
        return [c for c in self.cicli if c.get("faza") == "VSTAVLJANJE"]

    @property
    def cicli_pospravljanje(self):
        return [c for c in self.cicli if c.get("faza") == "POSPRAVLJANJE"]


# ===== PROCESOR COMBINED =====

class ProcesorCombined:
    """
    Procesira video z:
      A) Homografija pipeline  → 2D mm, per-luknjica, faza-ločeno
      B) Task-axis pipeline    → 1D projekcija, fallback
      C) LED stroj             → faza testa (VSTAVLJANJE/POSPRAVLJANJE)
    """

    def __init__(self, pot_videa, razmik_mm=RAZMIK_MM,
                 strategija=Strategija.AUTO,
                 izhod_video=None, izhod_graf=None, verbose=True):
        self.pot_videa   = pot_videa
        self.razmik_mm   = razmik_mm
        self.strategija  = strategija
        self.izhod_video = izhod_video
        self.izhod_graf  = izhod_graf
        self.verbose     = verbose

        import os as _os
        self.kamera    = doloci_kamero(_os.path.basename(pot_videa))
        self.roi_param = ROI_PARAMETRI.get(self.kamera, ROI_PARAMETRI["camP_0"])

        self.hom      = BoardHomografija(razmik_mm=razmik_mm)
        # Določi ime kamere iz poti videa za ROI kalibracijo
        _ime_vid = pot_videa.replace("\\", "/").split("/")[-1].replace(".mp4", "")
        _ime_kam = None
        for _k in ["camP_0","camP_1","camP_2","camS_0","camS_1","camS_2"]:
            if _k in _ime_vid:
                _ime_kam = _k; break
        self.sledilec = SledilecRoke(ime_kamere=_ime_kam) if MEDIAPIPE_OK else None

        self.traj2d  = None
        self.traj1d  = None
        self.fsm_hom = None
        self.fsm_1d  = None
        self.os      = None

        # Logerji za multi-točkovne trajektorije (TIP, THUMB, CENTER)
        self._log_multi = {
            "TIP":    {"casi": [], "traj": []},
            "THUMB":  {"casi": [], "traj": []},
            "CENTER": {"casi": [], "traj": []},
        }

        # LED stroj stanj — za zaznavo faze
        self.led_stroj       = None
        self._aktivno        = None   # 'zgornje' ali 'spodnje'
        self._ref_frame_zg   = None   # referenčni frame za luknjice
        self._ref_frame_sp   = None

        # Stanje inicializacije
        self._hom_ok                  = False
        self._os_ok                   = False
        self._blink_counter           = 0
        self._pospravljanje_t_start   = None   # zamuda pri preklopu faze
        self._overlay_renderer        = None   # inicializira se v procesiraj()

        if verbose:
            print(f"[Combined] kamera={self.kamera}, strategija={strategija}")
            print(f"[Combined] MediaPipe={'OK' if MEDIAPIPE_OK else 'NI'}")

    # ── LED BRANJE ──────────────────────────────────────────────────────

    def _preberi_luknjice(self, frame, sirina, visina):
        """Prebere LED luknjice iz obeh ROI-jev."""
        roi_zg, _ = izrezi_roi(frame, self.roi_param["zgornje"])
        roi_sp, _ = izrezi_roi(frame, self.roi_param["spodnje"])
        n_zg, c_zg_r = stej_svetle_luknjice(roi_zg)
        n_sp, c_sp_r = stej_svetle_luknjice(roi_sp)
        x1z = int(sirina * self.roi_param["zgornje"][0])
        y1z = int(visina * self.roi_param["zgornje"][1])
        x1s = int(sirina * self.roi_param["spodnje"][0])
        y1s = int(visina * self.roi_param["spodnje"][1])
        c_zg = [(x + x1z, y + y1z) for x, y in c_zg_r]
        c_sp = [(x + x1s, y + y1s) for x, y in c_sp_r]
        return n_zg, n_sp, c_zg, c_sp

    # ── BLINK INICIALIZACIJA ─────────────────────────────────────────────

    def _poskusi_blink_init(self, n_zg, n_sp, c_zg, c_sp):
        """Inicializira homografijo iz blink faze (obe področji ON)."""
        if self._hom_ok:
            return True

        if n_zg >= 7 and n_sp >= 7:
            self._blink_counter += 1
        else:
            self._blink_counter = 0
            return False

        if self._blink_counter < BLINK_POTRDITEV_FRAMOV:
            return False

        ok = self.hom.inicializiraj_iz_blinka(c_zg, c_sp)
        if not ok:
            return False

        self._hom_ok = True
        if self.verbose:
            print(f"[Combined] Homografija FIKSIRANA iz blink faze!")
            print(f"[Combined] px/mm = {self.hom.px_per_mm:.3f}")
            if self.hom.center_posodice_mm:
                print(f"[Combined] Center posodice = "
                      f"({self.hom.center_posodice_mm[0]:.1f},"
                      f"{self.hom.center_posodice_mm[1]:.1f})mm")
            if self.hom.razdalja_med_mm:
                print(f"[Combined] Razdalja med mrezama = {self.hom.razdalja_med_mm:.1f}mm")

        if self.hom.center_posodice_mm:
            self.fsm_hom.nastavi_posodico(self.hom.center_posodice_mm)

        # Task-axis os
        if not self._os_ok and len(c_zg) >= 3 and len(c_sp) >= 3:
            c_zg_c = np.mean(np.array(c_zg), axis=0)
            c_sp_c = np.mean(np.array(c_sp), axis=0)
            try:
                self.os = TaskOsGeometrija(tuple(c_sp_c), tuple(c_zg_c))
                self._os_ok = True
                if self.verbose:
                    print(f"[Combined] Task-os OK")
            except ValueError:
                pass

        return True

    # ── LED STROJ POSODOBITEV ────────────────────────────────────────────

    def _posodobi_led_stroj(self, frame_idx, n_zg, n_sp):
        """
        Posodobi LED stroj stanj in vrne trenutno fazo testa.

        LED stroj iz luknjice_led.py že zaznava:
          VSTAVLJANJE  → n_aktivno pada (zatiči se vstavljajo)
          CAKANJE      → n_aktivno = 0 (vsi vstavljeni)
          POSPRAVLJANJE → n_aktivno narašča (zatiči se jemljejo)

        Vrne: "VSTAVLJANJE", "POSPRAVLJANJE" ali "NEZNANO"
        """
        if self.led_stroj is None:
            return "NEZNANO"

        # Določi aktivno področje
        zg_on = je_obmocje_prizgano(n_zg)
        sp_on = je_obmocje_prizgano(n_sp)
        if zg_on and not sp_on:
            self._aktivno = "zgornje"
        elif sp_on and not zg_on:
            self._aktivno = "spodnje"

        # Posodobi LED stroj
        self.led_stroj.posodobi(frame_idx, n_zg, n_sp)

        # Preberi fazo
        faza = self.led_stroj.faza  # 'VSTAVLJANJE', 'CAKANJE', 'POSPRAVLJANJE'
        if faza == "CAKANJE":
            return "VSTAVLJANJE"
        # if faza == "POSPRAVLJANJE":
        #     if self.verbose:
        #         print(f"  [LED] t={frame_idx/self.led_stroj.fps:.2f}s → POSPRAVLJANJE")
        return faza

    # ── OVERLAY ─────────────────────────────────────────────────────────

    def _narisi_overlay(self, frame, hand_img, roka_mm, s_val, lm, t_s, faza):
        out = frame.copy()

        if self.sledilec and lm:
            out = self.sledilec.narisi(out, lm)

        # Task-axis os
        if self._os_ok and self.os:
            self.os.narisi_os(out, barva=(0, 200, 80))
            if hand_img and s_val is not None:
                proj = self.os.v_image_tocko(s_val)
                hx, hy = int(hand_img[0]), int(hand_img[1])
                cv2.line(out, (hx, hy), (int(proj[0]), int(proj[1])), (180, 180, 180), 1)
                cv2.circle(out, (int(proj[0]), int(proj[1])), 4, (255, 255, 0), -1)

        # Homografija overlay
        if self._hom_ok:
            out = self.hom.narisi_debug_overlay(
                out, roka_mm=roka_mm,
                pot_mm=self.traj2d.get_zadnjih_n(60) if self.traj2d else None)

        # Roka
        if hand_img:
            hx, hy = int(hand_img[0]), int(hand_img[1])
            cv2.circle(out, (hx, hy), 8, (50, 50, 230), -1)
            parts = []
            if roka_mm:
                parts.append(f"mm:({roka_mm[0]:.0f},{roka_mm[1]:.0f})")
            if s_val is not None:
                parts.append(f"s={s_val:.2f}")
            cv2.putText(out, " | ".join(parts), (hx + 10, hy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 230, 50), 1)

        # HUD
        hst = self.fsm_hom.stanje if self.fsm_hom else "-"
        ost = self.fsm_1d.stanje  if self.fsm_1d  else "-"
        n_v = len(self.fsm_hom.cicli_vstavljanje)   if self.fsm_hom else 0
        n_p = len(self.fsm_hom.cicli_pospravljanje) if self.fsm_hom else 0

        # Barva FSM glede na fazo
        barva_faza = (0, 200, 255) if faza == "VSTAVLJANJE" else (255, 150, 0)
        cv2.putText(out, f"FAZA: {faza}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, barva_faza, 2)
        cv2.putText(out, f"H-FSM: {hst}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(out, f"V:{n_v} P:{n_p}", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(out, f"t={t_s:.1f}s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)
        status = "FIKSIRANA" if self._hom_ok else f"blink:{self._blink_counter}"
        cv2.putText(out, f"HOM: {status}", (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 50), 1)
        return out

    # ── GLAVNI LOOP ──────────────────────────────────────────────────────

    def procesiraj(self):
        cap    = cv2.VideoCapture(self.pot_videa)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inicializiraj vse
        self.traj2d  = TrajektorijaBuffer2D(fps=fps)
        self.traj1d  = TrajektorijaBuffer1D(fps=fps)
        self.fsm_hom = FSMHomografija(self.hom, fps=fps, verbose=self.verbose)
        self.fsm_1d  = FSM1D(fps=fps, verbose=self.verbose)
        self.led_stroj = StrojStanj9HPT(fps=fps)

        # Overlay renderer
        if OVERLAY_OK:
            self._overlay_renderer = OverlayRenderer(fps=fps)

        writer = None
        if self.izhod_video:
            writer = cv2.VideoWriter(
                self.izhod_video, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (sirina, visina))

        log_casi, log_xs, log_ys, log_vel, log_dpos = [], [], [], [], []
        frame_idx = 0
        faza_trenutna = "NEZNANO"

        if self.verbose:
            print(f"\n[Combined] Start: fps={fps:.1f}, {sirina}x{visina}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t_s = frame_idx / fps

            # LED branje za zaznavo luknjic (homografija):
            #   vsak frame pred init, nato vsakih 5 frame-ov
            # LED STROJ pa mora dobiti n_zg/n_sp VSAK frame
            #   (trend za fazo zahteva kontinuiran signal)
            # LED zaznava VSAK frame — LED stroj potrebuje kontinuiran signal
            n_zg, n_sp, c_zg, c_sp = self._preberi_luknjice(frame, sirina, visina)
            # Po inicializaciji homografije preskoči računanje centrov (performance)
            if self._hom_ok:
                c_zg, c_sp = [], []

            # Blink inicializacija homografije
            if not self._hom_ok:
                self._poskusi_blink_init(n_zg, n_sp, c_zg, c_sp)

            # Posodobi LED stroj vsak frame in preberi fazo
            faza_trenutna = self._posodobi_led_stroj(frame_idx, n_zg, n_sp)

            # ── LIVE ŠTETJE ZATIČEV ─────────────────────────────────────
            # Sproti sledimo padcem (vstavljanje) in vzponom (pospravljanje)
            # signala n_luknjic — ne čakamo na postprocesiraj_casovnico.
            if self.led_stroj and self.led_stroj.stanje == 'ACTIVE':
                akt   = self.led_stroj.aktivno_obmocje
                n_akt = n_zg if akt == 'zgornje' else n_sp
                faza_led = self.led_stroj.faza

                if not hasattr(self, '_lv_init'):
                    self._lv_init        = True
                    self._lv_n_vst       = 0
                    self._lv_n_posp      = 0
                    self._lv_n_max_vst   = float(n_akt)  # max pri začetku vst.
                    self._lv_n_min_posp  = float(n_akt)  # min pri začetku posp.
                    self._lv_faza_zadnja = faza_led

                # Faza se je spremenila VST→CAKANJE→POSP
                if faza_led != self._lv_faza_zadnja:
                    if faza_led == 'POSPRAVLJANJE':
                        self._lv_n_min_posp = float(n_akt)
                    self._lv_faza_zadnja = faza_led

                if faza_led == 'VSTAVLJANJE':
                    if n_akt > self._lv_n_max_vst:
                        self._lv_n_max_vst = float(n_akt)
                    vst = int(round(self._lv_n_max_vst - n_akt))
                    self._lv_n_vst = max(0, min(vst, 9))

                elif faza_led == 'POSPRAVLJANJE':
                    if n_akt < self._lv_n_min_posp:
                        self._lv_n_min_posp = float(n_akt)
                    posp = int(round(n_akt - self._lv_n_min_posp))
                    self._lv_n_posp = max(0, min(posp, 9))

            # Posreduj fazo FSM-u
            # Po posodobitvi LED stroja, vedno preveri fazo
            # (ne samo ko je roka zaznana)
            if self.fsm_hom and faza_trenutna != "NEZNANO":
                self.fsm_hom.nastavi_fazo(faza_trenutna)
                # Če FSM čaka na IDLE za preklop, ga prisilno sprosti po 2s
                if (self.fsm_hom._cakajoca_faza is not None and
                        self.fsm_hom.trenutni.pickup_start is not None and
                        (t_s - self.fsm_hom.trenutni.pickup_start) > 2.0):
                    self.fsm_hom.stanje = FSMStanje.IDLE
                    self.fsm_hom._cnt = 0

            # Sledenje roke — zaznava vseh treh točk
            hand_img, lm = None, None
            tocke_px_multi = None
            if self.sledilec:
                tocke_px_multi, lm = self.sledilec.zazaj_multi(frame)
                # Osnovna točka za FSM in obstoječo logiko ostane TIP (kazalec)
                if tocke_px_multi:
                    hand_img = tocke_px_multi["TIP"]

            roka_mm = None
            s_val   = None

            if hand_img:
                if self._hom_ok:
                    roka_mm = self.hom.v_mm(hand_img)
                    # Za trajektorijo in kinematiko: WRIST (stabilnejši center roke)
                    # TIP ostane za FSM detekcijo vstavljanja
                    if roka_mm:
                        self.traj2d.dodaj(roka_mm[0], roka_mm[1], t_s)
                        vel = self.traj2d.hitrost_mm_s()

                        # Trajektorijo logiramo SAMO med aktivnim testom.
                        # Primarno: led_stroj.stanje == 'ACTIVE'
                        # Rezervno: cas_zacetka/cas_konca sta znana (po DONE)
                        #   → logiramo samo frame-e v tem časovnem oknu
                        test_aktiven = False
                        if self.led_stroj:
                            st = self.led_stroj.stanje
                            t_zac = (self.led_stroj.cas_zacetka / fps
                                     if self.led_stroj.cas_zacetka else None)
                            t_kon = (self.led_stroj.cas_konca / fps
                                     if self.led_stroj.cas_konca else None)
                            if st == 'ACTIVE' and t_zac is not None:
                                # Logiramo od blinka naprej
                                # (cas_konca ni znan dokler DONE ne pride)
                                test_aktiven = (t_s >= t_zac)
                            elif st == 'DONE' and t_zac is not None:
                                t_kon_eff = t_kon if t_kon else t_s
                                test_aktiven = (t_zac <= t_s <= t_kon_eff)

                        if test_aktiven:
                            log_casi.append(t_s)
                            # traj_mm = TIP (konica kazalca) — analizator_y
                            # in FSM pricakujeta TIP signal
                            log_xs.append(roka_mm[0])
                            log_ys.append(roka_mm[1])
                            log_vel.append(vel)
                            if self.hom.center_posodice_mm:
                                d_pos_val = np.linalg.norm(
                                    np.array(roka_mm) - np.array(self.hom.center_posodice_mm))
                                log_dpos.append(d_pos_val)
                            else:
                                log_dpos.append(np.nan)

                            # Logiranje vseh treh točk v mm
                            if tocke_px_multi:
                                for ime_t, px_t in tocke_px_multi.items():
                                    mm_t = self.hom.v_mm(px_t)
                                    if mm_t:
                                        self._log_multi[ime_t]["casi"].append(t_s)
                                        self._log_multi[ime_t]["traj"].append(mm_t)

                if self._os_ok and self.os:
                    s_val = self.os.projekcija(hand_img)
                    self.traj1d.dodaj(s_val, t_s)

            # FSM posodobitev
            if roka_mm and self._hom_ok:
                self.fsm_hom.posodobi(roka_mm, t_s, self.traj2d)
            if s_val is not None and self._os_ok:
                self.fsm_1d.posodobi(
                    s_val, t_s, self.traj1d.zadnja_hitrost(), self.traj1d)

            # Diagnostika vsakih 75 frame-ov
            # if self.verbose and frame_idx % 75 == 0 and frame_idx > 0:
            #     hst = self.fsm_hom.stanje if self.fsm_hom else "-"
            #     mm_str = f"({roka_mm[0]:.0f},{roka_mm[1]:.0f})mm" if roka_mm else "-"
            #     d_pos  = "-"
            #     if roka_mm and self.hom.center_posodice_mm:
            #         d_pos = f"{np.linalg.norm(np.array(roka_mm) - np.array(self.hom.center_posodice_mm)):.0f}mm"
            #     n_v = len(self.fsm_hom.cicli_vstavljanje)   if self.fsm_hom else 0
            #     n_p = len(self.fsm_hom.cicli_pospravljanje) if self.fsm_hom else 0
            #     print(f"  [t={t_s:.1f}s] {faza_trenutna:15s} roka={mm_str} "
            #           f"d_pos={d_pos} H={hst} V={n_v} P={n_p}")

            if writer is not None:
                if self._overlay_renderer:
                    ctx = {
                        "roka_px":    hand_img,
                        "roka_mm":    roka_mm,
                        "lm":         lm,
                        "t_s":        t_s,
                        "faza":       faza_trenutna,
                        # Live štetje iz LED signala (posodoblja se vsak frame)
                        "n_vst_led":  getattr(self, '_lv_n_vst',  0),
                        "n_posp_led": getattr(self, '_lv_n_posp', 0),
                        # Čas začetka testa (od blinka) za HUD uro
                        "t_blink": (self.led_stroj.cas_zacetka / fps
                                    if self.led_stroj and self.led_stroj.cas_zacetka
                                    else None),
                        # Čas konca testa (zamrzne uro)
                        "t_konec": (self.led_stroj.cas_konca / fps
                                    if self.led_stroj and self.led_stroj.cas_konca
                                    else None),
                        "hom":        self.hom,
                        "hom_ok":     self._hom_ok,
                        "roi_x_max":     self.sledilec._roi_x_max     if self.sledilec else 1.0,
                        "roi_x_max_bot": self.sledilec._roi_x_max_bot if self.sledilec else 1.0,
                    }
                    frame = self._overlay_renderer.narisi(frame, ctx)
                else:
                    frame = self._narisi_overlay(
                        frame, hand_img, roka_mm, s_val, lm, t_s, faza_trenutna)
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if self.sledilec:
            self.sledilec.zapri()

        # Post-procesiranje LED stroja
        postprocesiraj_casovnico(self.led_stroj, fps)

        # Zberi rezultate
        cicli_v = self.fsm_hom.cicli_vstavljanje   if self.fsm_hom else []
        cicli_p = self.fsm_hom.cicli_pospravljanje if self.fsm_hom else []
        cicli_vse = self.fsm_hom.cicli if self.fsm_hom else []

        casi_1d, s_raw, s_gl, vel1d = self.traj1d.get_vse()
        casi_hom = np.array(log_casi)
        xs_arr   = np.array(log_xs)
        ys_arr   = np.array(log_ys)
        vel_arr  = np.array(log_vel)

        if self.verbose:
            self._izpis(cicli_v, cicli_p)

        if self.izhod_graf and len(casi_1d) > 0:
            self._narisi_grafe(casi_hom, xs_arr, ys_arr, vel_arr,
                               casi_1d, s_raw, s_gl, vel1d,
                               cicli_v, cicli_p)

        # Kinematični grafi se kličejo iz pipeline.py (po koncu procesiraj)

        return {
            "cicli":                cicli_vse,
            "cicli_vstavljanje":    cicli_v,
            "cicli_pospravljanje":  cicli_p,
            "cicli_1d":             self.fsm_1d.cicli if self.fsm_1d else [],
            "px_per_mm":            self.hom.px_per_mm,
            "homografija":          self.hom,
            "os":                   self.os,
            "hom_ok":               self._hom_ok,
            "os_ok":                self._os_ok,
            "led_stroj":            self.led_stroj,
            "casi_1d":              casi_1d,
            "s_raw":                s_raw,
            "s_gl":                 s_gl,
            "vel_1d":               vel1d,
            "casi_hom":             casi_hom,
            "traj_mm":              list(zip(xs_arr, ys_arr)) if len(xs_arr) > 0 else [],
            "vel_mm_s":             vel_arr,
            "cicli_vstavljanje_y":   [],
            "cicli_pospravljanje_y": [],
            "log_multi":             self._log_multi,
        }

    # ── IZPIS ───────────────────────────────────────────────────────────

    def _izpis(self, cicli_v, cicli_p):
        print(f"\n=== COMBINED REZULTATI v3 ===")
        print(f"Vstavljanje ciklov:    {len(cicli_v)}  (pricakovano: 9)")
        print(f"Pospravljanje ciklov:  {len(cicli_p)}  (pricakovano: 9)")
        print(f"Skupaj:                {len(cicli_v) + len(cicli_p)}")
        print(f"px/mm:                 {self.hom.px_per_mm}")
        print(f"Center posodice:       {self.hom.center_posodice_mm}")

        # if cicli_v:
        #     print(f"\n--- VSTAVLJANJE ---")
        #     for i, c in enumerate(cicli_v):
        #         print(f"  V{i+1}: pickup={c.get('pickup_duration',0) or 0:.3f}s "
        #               f"premik={c.get('movement_time',0) or 0:.3f}s "
        #               f"insert={c.get('insert_duration',0) or 0:.3f}s "
        #               f"luk={c.get('target_hole','?')}")

        # if cicli_p:
        #     print(f"\n--- POSPRAVLJANJE ---")
        #     for i, c in enumerate(cicli_p):
        #         print(f"  P{i+1}: pickup={c.get('pickup_duration',0) or 0:.3f}s "
        #               f"premik={c.get('movement_time',0) or 0:.3f}s "
        #               f"insert={c.get('insert_duration',0) or 0:.3f}s "
        #               f"luk={c.get('target_hole','?')}")

        # LED stroj rezultati
        if self.led_stroj and self.led_stroj.cas_testa_sekunde:
            print(f"\n--- LED STROJ ---")
            print(f"  Skupni cas testa:   {self.led_stroj.cas_testa_sekunde:.2f}s")
            print(f"  Cas vstavljanja:    {self.led_stroj.cas_vstavljanja_sekunde or 0:.2f}s")
            print(f"  Cas pospravljanja:  {self.led_stroj.cas_pospravljanja_sekunde or 0:.2f}s")
            print(f"  Stevilo zaticev:    {self.led_stroj.stevilo_zatičev}")

        print(f"=============================\n")

    # ── GRAFI ───────────────────────────────────────────────────────────

    def _narisi_grafe(self, casi_hom, xs, ys, vel2d,
                      casi_1d, s_raw, s_gl, vel1d,
                      cicli_v, cicli_p):
        fig, axs = plt.subplots(3, 2, figsize=(18, 12))

        # Barve za vstavljanje/pospravljanje
        def _oznaci_cicle(ax, cicli, barva_start, barva_end):
            for c in cicli:
                for k, col in [("pickup_start", barva_start),
                                ("insert_complete", barva_end)]:
                    tv = c.get(k)
                    if tv:
                        ax.axvline(tv, color=col, alpha=0.6, lw=1.2)

        # X, Y koordinate
        axs[0, 0].plot(casi_hom, xs, color="steelblue",  label="X (mm)")
        axs[0, 0].plot(casi_hom, ys, color="darkorange", label="Y (mm)")
        _oznaci_cicle(axs[0, 0], cicli_v, "green",  "lime")
        _oznaci_cicle(axs[0, 0], cicli_p, "tomato", "orange")
        # Legenda faz
        axs[0, 0].axvline(0, color="green",  alpha=0, label="▏ vstavljanje start")
        axs[0, 0].axvline(0, color="tomato", alpha=0, label="▏ pospravljanje start")
        axs[0, 0].set_title("Homografija: X,Y (mm)")
        axs[0, 0].set_ylabel("mm")
        axs[0, 0].legend(fontsize=7)
        axs[0, 0].grid(alpha=0.3)

        # Hitrost
        axs[1, 0].plot(casi_hom[:len(vel2d)], vel2d, color="purple", label="v (mm/s)")
        axs[1, 0].set_title("Homografija: hitrost")
        axs[1, 0].set_ylabel("mm/s")
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].grid(alpha=0.3)

        # 2D trajektorija
        if len(xs) > 1:
            axs[2, 0].plot(xs, ys, color="purple", alpha=0.4, lw=0.7)
            axs[2, 0].scatter(xs[0],  ys[0],  color="green", s=40, zorder=5, label="Start")
            axs[2, 0].scatter(xs[-1], ys[-1], color="red",   s=40, zorder=5, label="End")
            # Vstavljanja — zelene pike
            for c in cicli_v:
                ti = c.get("insert_complete")
                if ti and len(casi_hom) > 0:
                    idx = np.argmin(np.abs(casi_hom - ti))
                    if idx < len(xs):
                        axs[2, 0].scatter(xs[idx], ys[idx], color="green",
                                          s=60, marker="^", zorder=6)
            # Pospravljanja — rdeče pike
            for c in cicli_p:
                ti = c.get("insert_complete")
                if ti and len(casi_hom) > 0:
                    idx = np.argmin(np.abs(casi_hom - ti))
                    if idx < len(xs):
                        axs[2, 0].scatter(xs[idx], ys[idx], color="tomato",
                                          s=60, marker="v", zorder=6)
            # Luknjice
            luk_A = self.hom.get_luknjice_mm()
            axs[2, 0].scatter(luk_A[:, 0], luk_A[:, 1],
                              color="gold", s=60, marker="o", label="luknjice (A)")
            for lx, ly in luk_A:
                axs[2, 0].add_patch(plt.Circle(
                    (lx, ly), self.hom.luknjica_roi_mm,
                    color="gold", fill=False, lw=0.8, alpha=0.5))
            if self.hom.world_B is not None:
                luk_B = self.hom.world_B
                axs[2, 0].scatter(luk_B[:, 0], luk_B[:, 1],
                                  color="lightsteelblue", s=40,
                                  marker="s", label="luknjice (B)")
                for lx, ly in luk_B:
                    axs[2, 0].add_patch(plt.Circle(
                        (lx, ly), self.hom.luknjica_roi_mm,
                        color="lightsteelblue", fill=False, lw=0.8, alpha=0.5))
            if self.hom.center_posodice_mm:
                cx, cy = self.hom.center_posodice_mm
                axs[2, 0].add_patch(plt.Circle(
                    (cx, cy), self.hom.polmer_posodice_mm,
                    color="darkorange", fill=False, lw=2,
                    label=f"posodica (d={int(self.hom.polmer_posodice_mm*2)}mm)"))
                axs[2, 0].plot(cx, cy, "+", color="darkorange", ms=10, markeredgewidth=2)
            axs[2, 0].set_aspect("equal")
            axs[2, 0].invert_yaxis()
        axs[2, 0].set_title("2D trajektorija (mm) | ▲=vstavitev ▼=pospravljanje")
        axs[2, 0].set_xlabel("X [mm]")
        axs[2, 0].set_ylabel("Y [mm]")
        axs[2, 0].legend(fontsize=7)
        axs[2, 0].grid(alpha=0.3)

        # Task-axis s(t)
        axs[0, 1].plot(casi_1d, s_raw, color="lightsteelblue",
                       alpha=0.4, lw=0.8, label="s surovi")
        axs[0, 1].plot(casi_1d, s_gl,  color="steelblue",
                       lw=1.5, label="s zglajen")
        axs[0, 1].axhline(CONTAINER_ROI_DELEZ, color="orange",
                           linestyle="--", alpha=0.7)
        axs[0, 1].axhline(HOLES_ROI_DELEZ, color="cyan",
                           linestyle="--", alpha=0.7)
        axs[0, 1].set_title("Task-axis: s(t)")
        axs[0, 1].set_ylabel("s")
        axs[0, 1].set_ylim(-0.2, 1.4)
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].grid(alpha=0.3)

        # Task-axis hitrost
        axs[1, 1].plot(casi_1d[:len(vel1d)], vel1d,
                       color="darkgreen", label="ds/dt")
        axs[1, 1].axhline(0, color="gray", lw=0.8)
        axs[1, 1].set_title("Task-axis: hitrost")
        axs[1, 1].set_ylabel("ds/dt")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(alpha=0.3)

        # Movement time primerjava — vstavljanje vs pospravljanje
        n_v = len(cicli_v)
        n_p = len(cicli_p)
        n_max = max(n_v, n_p, 1)
        x_v = np.arange(n_v)
        x_p = np.arange(n_p)
        mt_v = [c.get("movement_time") or 0 for c in cicli_v]
        mt_p = [c.get("movement_time") or 0 for c in cicli_p]
        if mt_v:
            axs[2, 1].bar(x_v - 0.2, mt_v, 0.35,
                          label="Vstavljanje", color="steelblue", alpha=0.8)
        if mt_p:
            axs[2, 1].bar(x_p + 0.2, mt_p, 0.35,
                          label="Pospravljanje", color="tomato", alpha=0.8)
        axs[2, 1].set_title("Movement time po ciklih [s]")
        axs[2, 1].set_ylabel("s")
        axs[2, 1].set_xlabel("Cikel #")
        axs[2, 1].legend(fontsize=8)
        axs[2, 1].grid(alpha=0.3, axis="y")

        fig.suptitle(
            f"Combined v3 | V:{n_v} P:{n_p} | "
            f"px/mm={self.hom.px_per_mm:.2f} | "
            f"strategija={self.strategija}",
            fontsize=11)
        plt.tight_layout()
        plt.savefig(self.izhod_graf, dpi=120)
        plt.close()
        print(f"[Graf] Shranjen: {self.izhod_graf}")


# ===== VSTOPNA TOČKA =====

if __name__ == "__main__":
    import sys
    pot = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/Data/patient_078/patient_078camP_0_20231116_11_14_00.mp4"
    p = ProcesorCombined(
        pot_videa=pot,
        razmik_mm=32,
        strategija=Strategija.AUTO,
        izhod_video="/workspace/results/combined_debug.mp4",
        izhod_graf="/workspace/results/combined_graf.png",
        verbose=True)
    r = p.procesiraj()
    print(f"\nVstavljanje: {len(r['cicli_vstavljanje'])} / 9")
    print(f"Pospravljanje: {len(r['cicli_pospravljanje'])} / 9")
    print(f"px/mm={r['px_per_mm']:.3f} | hom_ok={r['hom_ok']}")