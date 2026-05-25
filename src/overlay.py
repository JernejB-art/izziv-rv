#!/usr/bin/env python3
# overlay.py — čist HUD overlay za 9HPT output video
#
# Spremembe v tej verziji:
#   - Dead-zone filter za kinematiko: MediaPipe šum < DEAD_ZONE_MM se ignorira
#   - Velocity ceiling: v > V_MAX_FIZICNO se zavrže (nerealne hitrosti)
#   - Progres zatičev temelji na led_stroj (casi_vstavljanj / casi_pospravljanj)
#     in ne na analizator_y (ki da rezultate šele po koncu videa)
#   - Barviti MediaPipe skeleton: vsak prst svoja barva, različne debeline

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# ──────────────────────────────────────────────────────────────────────────
# FIZIKALNI PRAGI (nastavljivo)
# ──────────────────────────────────────────────────────────────────────────

# Minimalni premik ki se šteje kot gibanje [mm]
# MediaPipe šum na mirni roki je tipično ±1-3 mm v homografiranem prostoru
DEAD_ZONE_MM     = 2.5

# Maksimalna fizično možna hitrost roke pri 9HPT [mm/s]
# Najhitrejši pacienti dosežejo ~400 mm/s; nad tem je zagotovo šum
V_MAX_FIZICNO    = 500.0

# Minimalno število zaporednih frameov nad dead-zone za šteti kot gibanje
GIBANJE_MIN_N    = 3

# ──────────────────────────────────────────────────────────────────────────
# BARVE (BGR)
# ──────────────────────────────────────────────────────────────────────────

B_ZELENA   = (60, 220, 80)
B_RUMENA   = (30, 220, 230)
B_RDECA    = (60, 60, 230)
B_MODRA    = (220, 130, 40)
B_SIVA     = (160, 160, 160)
B_BELA     = (240, 240, 240)
B_CYAN     = (220, 210, 60)
B_LUKNJICE = (40, 220, 140)
B_POSODICA = (40, 160, 255)

# Barve prstov za MediaPipe skeleton (BGR)
# Vrstni red: palec, kazalec, sredinec, prstanec, mezinec
PRST_BARVE = [
    (80,  180, 255),   # palec    — zlato-oranžna
    (60,  220, 80),    # kazalec  — zelena
    (220, 200, 50),    # sredinec — cyan
    (220, 80,  180),   # prstanec — vijolična
    (80,  80,  230),   # mezinec  — rdeča
]

# Povezave MediaPipe (prst_idx, [(start_lm, end_lm), ...])
PRST_POVEZAVE = [
    (0, [(0,1),(1,2),(2,3),(3,4)]),          # palec
    (1, [(0,5),(5,6),(6,7),(7,8)]),          # kazalec
    (2, [(0,9),(9,10),(10,11),(11,12)]),     # sredinec
    (3, [(0,13),(13,14),(14,15),(15,16)]),   # prstanec
    (4, [(0,17),(17,18),(18,19),(19,20)]),   # mezinec
]
# Dlani
DLAN_POVEZAVE = [(0,5),(5,9),(9,13),(13,17),(0,17)]

# ──────────────────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────────────────

HUD_X       = 14
HUD_Y_START = 32
HUD_VRSTICA = 28
KIN_BUFFER  = 12       # število frameov za glajenje v/a
TRAJ_DOLZINA_S = 2.5
TRAJ_DEBELINA  = 2
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────────
# POMOŽNE FUNKCIJE
# ──────────────────────────────────────────────────────────────────────────

def hitrost_v_barvo(v_mm_s, v_max=300.0):
    t   = float(np.clip(v_mm_s / max(v_max, 1.0), 0.0, 1.0))
    hue = int((1.0 - t) * 90)
    hsv = np.uint8([[[hue, 230, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (int(bgr[0,0,0]), int(bgr[0,0,1]), int(bgr[0,0,2]))


def besedilo_senca(frame, tekst, pos, scale, barva, debelina=1, senca=True):
    if senca:
        cv2.putText(frame, tekst, (pos[0]+1, pos[1]+1),
                    FONT_SM, scale, (0,0,0), debelina+1, cv2.LINE_AA)
    cv2.putText(frame, tekst, pos, FONT_SM, scale, barva, debelina, cv2.LINE_AA)


def narisi_zapolnjen_pravokotnik(frame, x1, y1, x2, y2, barva, alpha=0.45):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(overlay, (0,0), (x2-x1, y2-y1), barva, -1)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def narisi_progres_pike(frame, n_zaznano, n_skupaj, x, y, barva_poln, barva_prazn):
    r, razmik = 6, 16
    for i in range(n_skupaj):
        barva = barva_poln if i < n_zaznano else barva_prazn
        cv2.circle(frame, (x + i*razmik, y), r, barva, -1)
        cv2.circle(frame, (x + i*razmik, y), r, (0,0,0), 1)


def format_cas(sekunde):
    if sekunde is None:
        return "--:--.--"
    s = float(sekunde)
    m = int(s // 60)
    return f"{m:02d}:{s - m*60:04.1f}"


# ──────────────────────────────────────────────────────────────────────────
# BARVITI MEDIAPIPE SKELETON
# ──────────────────────────────────────────────────────────────────────────

def narisi_skeleton_barvit(frame, lm, sirina, visina):
    """
    Nariše MediaPipe hand skeleton z barvami po prstih.
    Vsak prst ima svojo barvo, sklepi so obarvani z večjo piko.
    """
    if lm is None:
        return frame

    # Pretvori v pikselske koordinate
    def pt(idx):
        p = lm.landmark[idx]
        return (int(p.x * sirina), int(p.y * visina))

    # 1. Dlanski okvir (siva)
    for s, e in DLAN_POVEZAVE:
        cv2.line(frame, pt(s), pt(e), (120,120,120), 1, cv2.LINE_AA)

    # 2. Prsti (vsak svoja barva)
    for prst_idx, povezave in PRST_POVEZAVE:
        barva = PRST_BARVE[prst_idx]
        barva_temna = tuple(int(c * 0.55) for c in barva)

        for s, e in povezave:
            cv2.line(frame, pt(s), pt(e), barva, 2, cv2.LINE_AA)

        # Sklepi tega prsta
        for s, e in povezave:
            # manjša pika na začetku segmenta
            cv2.circle(frame, pt(s), 4, barva_temna, -1)
            cv2.circle(frame, pt(s), 4, barva, 1, cv2.LINE_AA)

    # 3. Konica kazalca — večja pika (najpomembnejša točka)
    cv2.circle(frame, pt(8), 7, PRST_BARVE[1], -1)
    cv2.circle(frame, pt(8), 7, B_BELA, 1, cv2.LINE_AA)

    # 4. Konica palca
    cv2.circle(frame, pt(4), 6, PRST_BARVE[0], -1)
    cv2.circle(frame, pt(4), 6, B_BELA, 1, cv2.LINE_AA)

    # 5. Zapestje
    cv2.circle(frame, pt(0), 5, (200,200,200), -1)

    return frame


# ──────────────────────────────────────────────────────────────────────────
# OVERLAY RENDERER
# ──────────────────────────────────────────────────────────────────────────

class _OneEuroRT:
    """OneEuro filter za realčasno glajenje (frame-po-frame)."""
    def __init__(self, fps, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.fps=fps; self.min_cutoff=min_cutoff
        self.beta=beta; self.d_cutoff=d_cutoff
        self.x_prev=None; self.dx_prev=0.0
    def _alpha(self, c):
        return 1.0/(1.0+1.0/(2.0*np.pi*c*(1.0/self.fps)))
    def filter(self, x):
        if self.x_prev is None: self.x_prev=x; return x
        dx=(x-self.x_prev)*self.fps
        dx_hat=self._alpha(self.d_cutoff)*dx+(1-self._alpha(self.d_cutoff))*self.dx_prev
        c=self.min_cutoff+self.beta*abs(dx_hat)
        a=self._alpha(c)
        xf=a*x+(1-a)*self.x_prev
        self.x_prev=xf; self.dx_prev=dx_hat; return xf



class OverlayRenderer:
    """
    Skrbi za risanje vseh elementov overlaya na vsak frame.

    ctx ki se poda v narisi():
      roka_px        : (x, y) v pikslih ali None  — konica kazalca
      roka_mm        : (x, y) v mm ali None
      lm             : MediaPipe NormalizedLandmarkList ali None
      t_s            : čas [s]
      faza           : "VSTAVLJANJE" | "POSPRAVLJANJE" | "NEZNANO" | "BLINK"
      n_vst_led      : int — vstavljeni zatički po LED stroju (live)
      n_posp_led     : int — pospravlienih zatičev po LED stroju (live)
      hom            : BoardHomografija ali None
      hom_ok         : bool
    """

    def __init__(self, fps=25.0):
        self.fps        = fps
        self.traj_maxn  = int(TRAJ_DOLZINA_S * fps)

        # Trajektorija v pikslih
        self._traj_px = deque(maxlen=self.traj_maxn)

        # OneEuro filter za glajenje landmark koordinat v realnem času
        self._oe_x = _OneEuroRT(fps, min_cutoff=1.0, beta=0.007)
        self._oe_y = _OneEuroRT(fps, min_cutoff=1.0, beta=0.007)

        # Kinematika bufferji
        self._v_buf = deque(maxlen=KIN_BUFFER)
        self._a_buf = deque(maxlen=KIN_BUFFER)
        self._d_kum = 0.0

        # Dead-zone štetje
        self._gibanje_n    = 0   # zaporedni frameji nad dead-zone
        self._v_prejsnja   = 0.0

        self._prejsnja_mm  = None
        self._t_blink      = None
        self._t_konec      = None   # zamrzne uro ob koncu testa
        self._v_max_opaz   = 80.0

    # ── KINEMATIKA Z DEAD-ZONE FILTROM ───────────────────────────────────

    def posodobi_kinematiko(self, roka_mm, t_s):
        """
        Posodobi d/v/a.
        - Premiki < DEAD_ZONE_MM se ignorirajo (MediaPipe šum)
        - Hitrosti > V_MAX_FIZICNO se zavržejo (nerealne vrednosti)
        - Zahtevamo GIBANJE_MIN_N zaporednih frameov nad dead-zone
          preden štejemo gibanje kot realno
        """
        if roka_mm is None:
            return
        # Filtriraj z OneEuro pred primerjavo
        x_f = self._oe_x.filter(roka_mm[0])
        y_f = self._oe_y.filter(roka_mm[1])
        roka_mm = (x_f, y_f)

        if self._prejsnja_mm is None:
            self._prejsnja_mm = roka_mm
            return

        dx   = roka_mm[0] - self._prejsnja_mm[0]
        dy   = roka_mm[1] - self._prejsnja_mm[1]
        razd = float(np.sqrt(dx**2 + dy**2))
        dt   = 1.0 / self.fps

        # Dead-zone filter
        if razd < DEAD_ZONE_MM:
            self._gibanje_n = 0
            self._v_buf.append(0.0)
            self._a_buf.append(0.0)
            self._prejsnja_mm = roka_mm
            self._v_prejsnja  = 0.0
            return

        self._gibanje_n += 1

        # Zahtevamo vsaj N zaporednih frameov gibanja
        if self._gibanje_n < GIBANJE_MIN_N:
            self._prejsnja_mm = roka_mm
            return

        v_inst = razd / dt

        # Velocity ceiling — zavrzi nerealne hitrosti
        if v_inst > V_MAX_FIZICNO:
            self._prejsnja_mm = roka_mm
            return

        # Sprejmi gibanje
        self._d_kum += razd
        self._v_buf.append(v_inst)

        v_gl = float(np.mean(list(self._v_buf)))
        a_inst = (v_gl - self._v_prejsnja) / dt
        # Zavrzi neupravičene pospeške (> 5× v_max fizicno / s)
        if abs(a_inst) < V_MAX_FIZICNO * 5:
            self._a_buf.append(a_inst)

        if v_inst > self._v_max_opaz:
            self._v_max_opaz = v_inst * 0.85

        self._v_prejsnja  = v_gl
        self._prejsnja_mm = roka_mm

    def get_kin(self):
        v_list = list(self._v_buf)
        a_list = list(self._a_buf)
        v = float(np.mean(v_list)) if v_list else 0.0
        a = float(np.mean(a_list)) if a_list else 0.0
        return self._d_kum, v, a

    def zabeleži_blink(self, t_s):
        if self._t_blink is None:
            self._t_blink = t_s

    # ── GLAVNA FUNKCIJA ──────────────────────────────────────────────────

    def narisi(self, frame, ctx):
        out = frame.copy()
        h, w = out.shape[:2]

        roka_px    = ctx.get("roka_px")
        roka_mm    = ctx.get("roka_mm")
        lm         = ctx.get("lm")
        t_s        = ctx.get("t_s", 0.0)
        faza       = ctx.get("faza", "NEZNANO")
        n_v        = ctx.get("n_vst_led",  0)
        n_p        = ctx.get("n_posp_led", 0)
        hom        = ctx.get("hom")
        hom_ok     = ctx.get("hom_ok", False)

        # Čas blinka in konca testa iz led_stroj
        t_blink = ctx.get("t_blink")
        if t_blink is not None:
            self._t_blink = t_blink
        t_konec = ctx.get("t_konec")
        if t_konec is not None and self._t_konec is None:
            self._t_konec = t_konec

        self.posodobi_kinematiko(roka_mm, t_s)
        if roka_px:
            self._traj_px.append(roka_px)

        d, v, a = self.get_kin()

        # ── 0. ROI MEJA ──────────────────────────────────────────────────
        roi_x_max = ctx.get("roi_x_max", 1.0)
        if roi_x_max < 1.0:
            x_meja = int(roi_x_max * w)
            cv2.line(out, (x_meja, 0), (x_meja, h), (0, 0, 200), 2, cv2.LINE_AA)
            besedilo_senca(out, "ROI", (x_meja - 36, 18), 0.38, (0, 0, 200))

        # ── 1. HOМ OVERLAY ───────────────────────────────────────────────
        if hom_ok and hom and hom.veljavna:
            out = self._narisi_hom_overlay(out, hom)

        # ── 2. TRAJEKTORIJA ──────────────────────────────────────────────
        self._narisi_trajektorijo(out, v)

        # ── 3. BARVITI SKELETON ──────────────────────────────────────────
        if lm is not None:
            narisi_skeleton_barvit(out, lm, w, h)

        # ── 4. ROKA PUNKT (konica kazalca) ───────────────────────────────
        if roka_px:
            barva_v = hitrost_v_barvo(v, self._v_max_opaz)
            cv2.circle(out, roka_px, 11, barva_v, -1)
            cv2.circle(out, roka_px, 11, B_BELA, 1, cv2.LINE_AA)

        # ── 5. HUD ───────────────────────────────────────────────────────
        self._narisi_hud(out, w, h, t_s, faza, n_v, n_p, d, v, a, hom_ok)

        # ── 6. PROGRES (LED-based) ────────────────────────────────────────
        self._narisi_progres(out, w, h, n_v, n_p, faza)

        return out

    # ── INTERNI RISARJI ──────────────────────────────────────────────────

    def _narisi_hom_overlay(self, frame, hom):
        out = frame
        px  = hom.px_per_mm or 3.0
        r_luk = max(5, int(hom.luknjica_roi_mm * px))

        for centri in [hom._centri_zg_img, hom._centri_sp_img]:
            if centri is not None:
                for cx, cy in centri.astype(int):
                    cv2.circle(out, (cx, cy), r_luk, B_LUKNJICE, 1, cv2.LINE_AA)
                    cv2.circle(out, (cx, cy), 4,     B_LUKNJICE, -1)

        if hom.center_posodice_mm is not None:
            pi = hom.v_sliko(hom.center_posodice_mm)
            if pi:
                cx, cy  = int(pi[0]), int(pi[1])
                r_fiz   = max(6, int(hom.polmer_posodice_mm * px))
                r_roi   = max(6, int(hom.posodica_roi_mm   * px))
                cv2.circle(out, (cx, cy), r_fiz, B_POSODICA, 2, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), r_roi, B_POSODICA, 1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 5,     B_POSODICA, -1)
                besedilo_senca(out, "posodica",
                               (cx-28, cy-r_fiz-6), 0.36, B_POSODICA)
        return out

    def _narisi_trajektorijo(self, frame, v_trenutna):
        pts = list(self._traj_px)
        if len(pts) < 2:
            return
        n = len(pts)
        for i in range(1, n):
            alpha = i / n
            barva = hitrost_v_barvo(v_trenutna * alpha, self._v_max_opaz)
            cv2.line(frame,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     barva, TRAJ_DEBELINA, cv2.LINE_AA)

    def _narisi_hud(self, frame, w, h, t_s, faza, n_v, n_p, d, v, a, hom_ok):
        narisi_zapolnjen_pravokotnik(frame, 0, 0, 225, 205, (10,10,10), 0.55)
        y = HUD_Y_START

        # Faza
        if faza == "VSTAVLJANJE":
            bf, tf = B_ZELENA,  "VST"
        elif faza == "POSPRAVLJANJE":
            bf, tf = B_RDECA,   "POSP"
        elif faza == "BLINK":
            bf, tf = B_RUMENA,  "BLINK"
        else:
            bf, tf = B_SIVA,    "--"
        besedilo_senca(frame, tf, (HUD_X, y), 0.70, bf, 2)
        y += HUD_VRSTICA

        # Čas testa (od blinka do konca — zamrzne ob DONE)
        if self._t_blink is not None and self._t_blink <= t_s:
            t_ref = self._t_konec if self._t_konec is not None else t_s
            cas = t_ref - self._t_blink
            cas_str   = format_cas(cas)
            cas_barva = (180, 180, 60) if self._t_konec is not None else B_BELA
        else:
            cas_str   = "--:--.--"
            cas_barva = B_SIVA
        besedilo_senca(frame, cas_str, (HUD_X, y), 0.62, cas_barva, 1)
        y += HUD_VRSTICA

        # Zatički (LED)
        besedilo_senca(frame, f"V:{n_v}/9  P:{n_p}/9",
                       (HUD_X, y), 0.50, B_SIVA, 1)
        y += HUD_VRSTICA

        cv2.line(frame, (HUD_X, y-6), (215, y-6), (80,80,80), 1)

        # d / v / a
        besedilo_senca(frame, f"d {d:6.0f} mm",    (HUD_X, y), 0.50, B_CYAN, 1)
        y += int(HUD_VRSTICA * 0.88)
        v_barva = hitrost_v_barvo(v, self._v_max_opaz)
        besedilo_senca(frame, f"v {v:6.0f} mm/s",  (HUD_X, y), 0.50, v_barva, 1)
        y += int(HUD_VRSTICA * 0.88)
        a_barva = (60,60,220) if abs(a) > 150 else B_SIVA
        besedilo_senca(frame, f"a {a:+6.0f} mm/s2", (HUD_X, y), 0.50, a_barva, 1)

        # HOM status spodaj
        hom_b = (60,180,60) if hom_ok else (40,40,200)
        besedilo_senca(frame, f"HOM {'OK' if hom_ok else 'INIT'}",
                       (HUD_X, h-14), 0.38, hom_b, 1, senca=False)

    def _narisi_progres(self, frame, w, h, n_v, n_p, faza):
        """
        Progres bar temelji na n_vst_led / n_posp_led ki pride iz
        led_stroj.stevilo_vstavljenih / pospravljenih — live med videom.
        """
        if faza not in ("VSTAVLJANJE", "POSPRAVLJANJE"):
            return

        y_v = h - 38
        y_p = h - 18
        x0  = w // 2 - (9 * 16) // 2

        narisi_zapolnjen_pravokotnik(
            frame, x0-8, y_v-12, x0+9*16+8, y_p+14, (10,10,10), 0.50)

        narisi_progres_pike(frame, n_v, 9, x0, y_v, B_ZELENA, (50,50,50))
        besedilo_senca(frame, "VST",  (x0-38, y_v+5), 0.36, B_ZELENA)

        narisi_progres_pike(frame, n_p, 9, x0, y_p, B_MODRA,  (50,50,50))
        besedilo_senca(frame, "POSP", (x0-42, y_p+5), 0.36, B_MODRA)