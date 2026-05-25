#!/usr/bin/env python3
# dva_grafi.py — kinematični grafi d/v/a za 9HPT test
#
# Izračuna in prikaže kinematične parametre za TRI točke roke:
#   TIP    — konica kazalca (INDEX_FINGER_TIP, lm idx 8)
#   THUMB  — konica palca  (THUMB_TIP, lm idx 4)
#   CENTER — center roke   (WRIST, lm idx 0)
#
# Grafi:
#   ① Skupni pregled: d/v/a za vse 3 točke v enem grafu (primerjava)
#   ② Podrobni grafi: ločen panel d/v/a za vsako točko z označenimi cikli
#   ③ Fazni grafi: normalizirani po posameznih ciklih vstavljanja/pospravljanja
#   ④ 2D XY trajektorija v homografiranem prostoru (mm)
#
# CSV izvoz:
#   frame, t_s, <tocka>_x_mm, <tocka>_y_mm, <tocka>_d_mm,
#              <tocka>_v_mm_s, <tocka>_a_mm_s2   × 3 točke
#
# Uporaba:
#   from dva_grafi import izracunaj_dva, izracunaj_dva_multi, narisi_dva_multi
#   ali:
#   python3 dva_grafi.py --json results/patient_024/patient_024_rezultati.json

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SRC_MAPA = "/workspace/src"
if SRC_MAPA not in sys.path:
    sys.path.insert(0, SRC_MAPA)

try:
    from scipy.signal import butter, filtfilt
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

IZHOD_MAPA = "/workspace/results"

# Barve za 3 točke (matplotlib)
BARVE = {
    "TIP":    "#2196F3",   # modra   — kazalec
    "THUMB":  "#FF9800",   # oranžna — palec
    "CENTER": "#4CAF50",   # zelena  — center/zapestje
}
OZNAKE = {
    "TIP":    "Kazalec (TIP)",
    "THUMB":  "Palec (THUMB)",
    "CENTER": "Center roke (WRIST)",
}


# ══════════════════════════════════════════════════════════════════════════
# FILTRIRANJE IN IZRAČUN
# ══════════════════════════════════════════════════════════════════════════

def glajenje_butter(signal, fps, cutoff_hz=5.0):
    """
    Butterworth low-pass filter 4. reda.
    Fizikalno utemeljen za biomedicinske signale —
    gibanje roke pri 9HPT ni hitrejše od ~2 Hz.
    """
    if not SCIPY_OK or len(signal) < 15:
        return np.array(signal, dtype=float)
    nyq = fps / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.9)
    b, a = butter(4, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


# ══════════════════════════════════════════════════════════════════════════
# ONE EURO FILTER (Casiez et al. 2012)
# Adaptiven LP filter: agresivno gladi pri mirni roki, odziven pri gibanju
# Idealen za MediaPipe tracking šum
# ══════════════════════════════════════════════════════════════════════════

class OneEuroFilter:
    """
    1€ filter za glajenje MediaPipe landmark koordinat.
    
    Pri mirni roki (nizka hitrost): nizka cutoff → agresivno glajenje šuma
    Pri gibanju (visoka hitrost):   visoka cutoff → hiter odziv brez zamika
    
    Parametri:
        min_cutoff: minimalna cutoff frekvenca [Hz] — agresivnost pri mirni roki
        beta:       faktor povečanja cutoff pri gibanju — odzivnost
        d_cutoff:   cutoff za glajenje derivative [Hz]
    """
    def __init__(self, fps, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.fps       = fps
        self.min_cutoff = min_cutoff
        self.beta      = beta
        self.d_cutoff  = d_cutoff
        self.x_prev    = None
        self.dx_prev   = 0.0

    def _alpha(self, cutoff):
        te  = 1.0 / self.fps
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        # Zgladimo derivativo
        dx      = (x - self.x_prev) * self.fps
        a_d     = self._alpha(self.d_cutoff)
        dx_hat  = a_d * dx + (1.0 - a_d) * self.dx_prev
        # Adaptivna cutoff frekvenca
        cutoff  = self.min_cutoff + self.beta * abs(dx_hat)
        a       = self._alpha(cutoff)
        x_hat   = a * x + (1.0 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        return x_hat

    @staticmethod
    def filtriraj_signal(signal, fps, min_cutoff=1.0, beta=0.007):
        """Filtriraj 1D numpy signal."""
        f = OneEuroFilter(fps, min_cutoff=min_cutoff, beta=beta)
        return np.array([f.filter(float(x)) for x in signal])

    @staticmethod
    def filtriraj_2d(xs, ys, fps, min_cutoff=1.0, beta=0.007):
        """Filtriraj 2D trajektorijo (x, y arrays)."""
        return (OneEuroFilter.filtriraj_signal(xs, fps, min_cutoff, beta),
                OneEuroFilter.filtriraj_signal(ys, fps, min_cutoff, beta))


def izracunaj_dva(casi, traj_mm, fps=25.0, t_konec=None):
    """
    Izračuna d/v/a iz ene 2D trajektorije.

    Parametri:
        casi    → array časov [s]
        traj_mm → list ali array [(x,y), ...] v mm
        fps     → hitrost videa [Hz]

    Vrne dict ali None če premalo točk:
        casi        → array časov [s]
        x_gl, y_gl  → zglajena x in y koordinata [mm]
        d           → kumulativna pot [mm]
        v           → hitrost [mm/s]
        a           → pospešek [mm/s²]
        pot_skupaj  → skupna pot [mm]
        v_max       → maksimalna hitrost [mm/s]
        a_max       → maksimalni pospešek (abs) [mm/s²]
    """
    casi = np.array(casi, dtype=float)
    traj = np.array(traj_mm, dtype=float)
    if len(traj) < 10 or traj.ndim != 2 or traj.shape[1] != 2:
        return None

    # Obreži na čas konca testa (overlay ustavi štetje ob t_konec)
    if t_konec is not None:
        maska = casi <= t_konec
        casi  = casi[maska]
        traj  = traj[maska]
        if len(traj) < 10:
            return None

    dt = 1.0 / fps

    # ── GLADKI SIGNALI za grafe (OneEuro filter) ─────────────────────────
    # OneEuro: agresivno gladi ko roka miruje, odziven pri gibanju
    # Bistveno boljši od LP za MediaPipe šum (Casiez et al. 2012)
    x_gl, y_gl = OneEuroFilter.filtriraj_2d(
        traj[:, 0], traj[:, 1], fps,
        min_cutoff=1.0,   # agresivnost pri mirni roki
        beta=0.007        # odzivnost pri gibanju
    )

    dx = np.diff(x_gl); dy = np.diff(y_gl)
    razdalje_gl = np.sqrt(dx**2 + dy**2)

    v_raw = np.concatenate([[0.0], razdalje_gl / dt])
    v = glajenje_butter(v_raw, fps, cutoff_hz=2.0)
    v = np.clip(v, 0.0, None)

    dv = np.diff(v)
    a_raw = np.concatenate([[0.0], dv / dt])
    a = glajenje_butter(a_raw, fps, cutoff_hz=1.5)

    # ── KUMULATIVNA POT — dead-zone logika (konzistentna z overlay) ──────
    # Seštevamo SUROVE (neglajene) premike z enakimi pogoji kot overlay.py
    DEAD_ZONE_MM  = 2.5
    V_MAX_MM_S    = 500.0
    GIBANJE_MIN_N = 3

    d_kum = 0.0; gibanje_n = 0; prejsnja = traj[0]
    d_arr = [0.0]
    for i in range(1, len(traj)):
        dx_i = traj[i, 0] - prejsnja[0]
        dy_i = traj[i, 1] - prejsnja[1]
        razd = float(np.sqrt(dx_i**2 + dy_i**2))
        if razd < DEAD_ZONE_MM:
            gibanje_n = 0; prejsnja = traj[i]; d_arr.append(d_kum); continue
        gibanje_n += 1
        if gibanje_n < GIBANJE_MIN_N or razd / dt > V_MAX_MM_S:
            prejsnja = traj[i]; d_arr.append(d_kum); continue
        d_kum += razd
        prejsnja = traj[i]
        d_arr.append(d_kum)
    d = np.array(d_arr)

    # Zagotovi enako dolžino
    n = len(casi)
    d    = d[:n];    v    = v[:n];    a    = a[:n]
    x_gl = x_gl[:n]; y_gl = y_gl[:n]

    # Statistike na gladkem signalu — 95p na aktivnih segmentih (v > 20mm/s)
    v_aktiv = v[v > 20.0]
    v_95  = float(np.percentile(v_aktiv, 95)) if len(v_aktiv) > 5 else float(np.max(v))
    a_95  = float(np.percentile(np.abs(a), 95))
    v_max = float(np.max(v))
    a_max = float(np.max(np.abs(a)))

    # Path ratio: dejanska pot / direktna razdalja start→konec
    # Vrednost 1.0 = ravna črta, višja = bolj vijugasta pot
    # Normaliziran parameter — primerljiv med pacienti neodvisno od hitrosti
    displacement = float(np.sqrt(
        (x_gl[-1] - x_gl[0])**2 + (y_gl[-1] - y_gl[0])**2))
    path_ratio = float(d[-1] / displacement) if displacement > 1.0 else None

    return {
        "casi":        casi,
        "x_gl":        x_gl,
        "y_gl":        y_gl,
        "d":           d,
        "v":           v,
        "a":           a,
        "pot_skupaj":  float(d[-1]),
        "displacement": displacement,
        "path_ratio":  path_ratio,
        "v_95":        v_95,
        "a_95":        a_95,
        "v_max":       v_max,
        "a_max":       a_max,
    }


def izracunaj_dva_multi(log_multi, fps=25.0):
    """
    Izračuna d/v/a za vse razpoložljive točke hkrati.

    log_multi → dict {
        "TIP":    { "casi": [...], "traj": [(x,y),...] },
        "THUMB":  { ... },
        "CENTER": { ... },
    }

    Vrne dict { "TIP": kin_dict, "THUMB": kin_dict, "CENTER": kin_dict }
    kjer je kin_dict rezultat izracunaj_dva() ali None.
    """
    rezultati = {}
    for ime, podatki in log_multi.items():
        casi = podatki.get("casi", [])
        traj = podatki.get("traj", [])
        if len(casi) > 10 and len(traj) > 10:
            rezultati[ime] = izracunaj_dva(casi, traj, fps=fps)
        else:
            rezultati[ime] = None
    return rezultati


# ══════════════════════════════════════════════════════════════════════════
# CSV IZVOZ
# ══════════════════════════════════════════════════════════════════════════

def izvozi_csv_multi(kin_multi, pot_csv):
    """
    Izvozi kinematične parametre vseh treh točk v CSV.

    Stolpci:
        frame, t_s,
        TIP_x_mm, TIP_y_mm, TIP_d_mm, TIP_v_mm_s, TIP_a_mm_s2,
        THUMB_x_mm, THUMB_y_mm, THUMB_d_mm, THUMB_v_mm_s, THUMB_a_mm_s2,
        CENTER_x_mm, CENTER_y_mm, CENTER_d_mm, CENTER_v_mm_s, CENTER_a_mm_s2

    Točke so sinhronizirane po skupnem časovnem osi (interpolacija).
    """
    tocke = ["TIP", "THUMB", "CENTER"]

    # Najdi skupno časovno os — vzamemo najdaljšo med razpoložljivimi
    ref_casi = None
    for ime in tocke:
        kin = kin_multi.get(ime)
        if kin is not None:
            c = kin["casi"]
            if ref_casi is None or len(c) > len(ref_casi):
                ref_casi = c

    if ref_casi is None or len(ref_casi) == 0:
        return None

    # Glava CSV
    stolpci = ["frame", "t_s"]
    for ime in tocke:
        for par in ["x_mm", "y_mm", "d_mm", "v_mm_s", "a_mm_s2"]:
            stolpci.append(f"{ime}_{par}")

    os.makedirs(os.path.dirname(pot_csv) or ".", exist_ok=True)

    with open(pot_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(stolpci)

        for i, t in enumerate(ref_casi):
            vrstica = [i, round(float(t), 4)]

            for ime in tocke:
                kin = kin_multi.get(ime)
                if kin is None:
                    vrstica += ["", "", "", "", ""]
                    continue

                # Interpoliraj vrednosti na skupno os (če se časi razlikujejo)
                k_casi = kin["casi"]
                def interp(signal):
                    if len(signal) == len(ref_casi):
                        return float(signal[i])
                    return float(np.interp(t, k_casi, signal))

                vrstica += [
                    round(interp(kin["x_gl"]), 2),
                    round(interp(kin["y_gl"]), 2),
                    round(interp(kin["d"]),    2),
                    round(interp(kin["v"]),    2),
                    round(interp(kin["a"]),    2),
                ]
            w.writerow(vrstica)

    return pot_csv


# ══════════════════════════════════════════════════════════════════════════
# GRAFI
# ══════════════════════════════════════════════════════════════════════════

def _oznaci_cicle(ax, cicli_v, cicli_p, alpha_pas=0.07, alpha_crta=0.45):
    """Označi vstavljanja (zeleno) in pospravljanja (rdeče) na osi."""
    for c in (cicli_v or []):
        t0 = c.get("pickup_start")
        t1 = c.get("insert_complete")
        if t0 and t1:
            ax.axvspan(t0, t1, alpha=alpha_pas, color="green", lw=0)
        if t0:
            ax.axvline(t0, color="green",  alpha=alpha_crta, lw=0.9, ls="-")
        if t1:
            ax.axvline(t1, color="green",  alpha=0.25,       lw=0.7, ls="--")

    for c in (cicli_p or []):
        t0 = c.get("pickup_start")
        t1 = c.get("insert_complete")
        if t0 and t1:
            ax.axvspan(t0, t1, alpha=alpha_pas, color="tomato", lw=0)
        if t0:
            ax.axvline(t0, color="tomato", alpha=alpha_crta, lw=0.9, ls="-")
        if t1:
            ax.axvline(t1, color="tomato", alpha=0.25,       lw=0.7, ls="--")


def _skupna_legenda(cicli_v, cicli_p):
    """Naredi skupno legendo faz."""
    elementi = []
    if cicli_v:
        elementi.append(Patch(color="green",  alpha=0.5, label="Vstavljanje"))
    if cicli_p:
        elementi.append(Patch(color="tomato", alpha=0.5, label="Pospravljanje"))
    return elementi


# ── Graf 1: Skupni pregled (vse točke v en pogled) ────────────────────────

def narisi_dva_pregled(kin_multi, cicli_v=None, cicli_p=None,
                       izhod_pot=None, naslov=""):
    """
    3×1 panel: d(t), v(t), a(t) — vse razpoložljive točke skupaj.
    Primerjalni pogled: takoj vidiš razlike med kazalcem, palcem, centrom.
    """
    razpolozljive = {k: v for k, v in kin_multi.items() if v is not None}
    if not razpolozljive:
        return

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axs:
        ax.set_facecolor("#161616")

    if naslov:
        fig.suptitle(naslov, fontsize=12, color="white", y=0.98)

    oznake_y  = ["Kumulativna pot d [mm]", "Hitrost v [mm/s]", "Pospešek a [mm/s²]"]
    naslovi   = ["Pot  d(t)", "Hitrost  v(t)", "Pospešek  a(t)"]
    kljuci_k  = ["d", "v", "a"]
    stat_kljuci = ["pot_skupaj", "v_max", "a_max"]
    stat_enote  = ["mm", "mm/s", "mm/s²"]

    for row, (ylab, tit, kk, sk, se) in enumerate(
            zip(oznake_y, naslovi, kljuci_k, stat_kljuci, stat_enote)):
        ax = axs[row]

        for ime, kin in razpolozljive.items():
            barva = BARVE[ime]
            ax.plot(kin["casi"], kin[kk],
                    color=barva, lw=1.6, alpha=0.9,
                    label=f"{OZNAKE[ime]}  ({kin[sk]:.0f} {se})")

        _oznaci_cicle(ax, cicli_v, cicli_p)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_ylabel(ylab, color="#ccc", fontsize=9)
        ax.set_title(tit, color="#ddd", fontsize=10, pad=3)
        ax.tick_params(colors="#888")
        ax.grid(alpha=0.15, color="#555")
        for sp in ax.spines.values():
            sp.set_color("#333")

        legenda_pts = [Line2D([0], [0], color=BARVE[k], lw=2,
                               label=f"{OZNAKE[k]}  ({kin_multi[k][sk]:.0f} {se})")
                        for k in razpolozljive]
        faz_leg = _skupna_legenda(cicli_v, cicli_p)
        ax.legend(handles=legenda_pts + faz_leg,
                  fontsize=7.5, loc="upper right",
                  facecolor="#222", labelcolor="white", edgecolor="#444")

    axs[2].set_xlabel("Čas [s]", color="#aaa", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if izhod_pot:
        plt.savefig(izhod_pot, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


# ── Graf 2: Podrobni panel za vsako točko ────────────────────────────────

def narisi_dva_podrobno(kin_multi, cicli_v=None, cicli_p=None,
                        izhod_pot=None, naslov=""):
    """
    3 stolpci (po ena točka) × 3 vrstice (d, v, a).
    Vsak stolpec = ena točka roke, vsaka vrstica = en kinematični parameter.
    """
    tocke = [k for k in ["TIP", "THUMB", "CENTER"] if kin_multi.get(k)]
    n_tock = len(tocke)
    if n_tock == 0:
        return

    fig = plt.figure(figsize=(6 * n_tock, 12))
    fig.patch.set_facecolor("#0f0f0f")
    if naslov:
        fig.suptitle(naslov, fontsize=12, color="white", y=0.99)

    gs = gridspec.GridSpec(3, n_tock, figure=fig,
                           hspace=0.35, wspace=0.28)

    oznake_y = ["d [mm]", "v [mm/s]", "a [mm/s²]"]
    kljuci_k = ["d", "v", "a"]
    stat_k   = ["pot_skupaj", "v_max", "a_max"]
    stat_e   = ["mm", "mm/s", "mm/s²"]

    for col, ime in enumerate(tocke):
        kin = kin_multi[ime]
        barva = BARVE[ime]

        for row, (ylab, kk, sk, se) in enumerate(
                zip(oznake_y, kljuci_k, stat_k, stat_e)):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#161616")

            ax.plot(kin["casi"], kin[kk], color=barva, lw=1.5, alpha=0.92)
            _oznaci_cicle(ax, cicli_v, cicli_p)
            ax.axhline(0, color="#444", lw=0.6)

            ax.set_ylabel(ylab, color="#bbb", fontsize=8)
            ax.tick_params(colors="#777", labelsize=7)
            ax.grid(alpha=0.12, color="#555")
            for sp in ax.spines.values():
                sp.set_color("#333")

            stat_val = kin[sk]
            ax.set_title(f"{OZNAKE[ime]}  |  {kk}(t)  [{stat_val:.0f} {se}]",
                         color="#ddd", fontsize=8.5, pad=3)

            if row == 2:
                ax.set_xlabel("t [s]", color="#999", fontsize=8)

            # Faz legenda samo v prvi vrstici prvega stolpca
            if row == 0 and col == 0:
                faz_leg = _skupna_legenda(cicli_v, cicli_p)
                if faz_leg:
                    ax.legend(handles=faz_leg, fontsize=7,
                              facecolor="#222", labelcolor="white",
                              edgecolor="#444", loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if izhod_pot:
        plt.savefig(izhod_pot, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


# ── Graf 3: 2D trajektorije v mm prostoru ────────────────────────────────

def narisi_trajektorije_2d(kin_multi, hom=None,
                           cicli_v=None, cicli_p=None,
                           izhod_pot=None, naslov=""):
    """
    2D XY prikaz vseh trajektorij v homografiranem prostoru [mm].
    Prikaže luknjice, posodico in barvne trajektorije vseh točk.
    """
    razpolozljive = {k: v for k, v in kin_multi.items() if v is not None}
    if not razpolozljive:
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#161616")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if naslov:
        fig.suptitle(naslov, fontsize=11, color="white")

    # Luknjice (iz homografije)
    if hom and hom.veljavna:
        luk_A = hom.world_A
        for i, (lx, ly) in enumerate(luk_A):
            circ = plt.Circle((lx, ly), hom.luknjica_roi_mm,
                               color="#FFD700", fill=False, lw=1.2, alpha=0.6)
            ax.add_patch(circ)
            ax.plot(lx, ly, "o", color="#FFD700", ms=5, zorder=5)
            ax.annotate(str(i), (lx + 1.5, ly - 2),
                        fontsize=6.5, color="#FFD700", alpha=0.8)

        if hom.world_B is not None:
            for i, (bx, by) in enumerate(hom.world_B):
                circ = plt.Circle((bx, by), hom.luknjica_roi_mm,
                                   color="#90CAF9", fill=False, lw=1.0, alpha=0.5)
                ax.add_patch(circ)
                ax.plot(bx, by, "s", color="#90CAF9", ms=4, alpha=0.6)

        if hom.center_posodice_mm:
            cx, cy = hom.center_posodice_mm
            circ_fiz = plt.Circle((cx, cy), hom.polmer_posodice_mm,
                                   color="#FF9800", fill=False, lw=2, alpha=0.7)
            ax.add_patch(circ_fiz)
            ax.plot(cx, cy, "+", color="#FF9800", ms=12, mew=2)

    # Trajektorije
    for ime, kin in razpolozljive.items():
        xs, ys = kin["x_gl"], kin["y_gl"]
        barva = BARVE[ime]
        ax.plot(xs, ys, "-", color=barva, lw=1.3, alpha=0.75,
                label=f"{OZNAKE[ime]}  ({kin['pot_skupaj']:.0f} mm)")
        # Start / end markerji
        ax.plot(xs[0],  ys[0],  "^", color=barva, ms=9, zorder=7, alpha=0.9)
        ax.plot(xs[-1], ys[-1], "s", color=barva, ms=9, zorder=7, alpha=0.9)

    # Vstavljanja in pospravljanja — označi točke na trajektoriji CENTER (ali TIP)
    ref_kin = kin_multi.get("CENTER") or kin_multi.get("TIP")
    if ref_kin:
        for c in (cicli_v or []):
            ti = c.get("insert_complete")
            if ti:
                idx = np.argmin(np.abs(ref_kin["casi"] - ti))
                if idx < len(ref_kin["x_gl"]):
                    ax.scatter(ref_kin["x_gl"][idx], ref_kin["y_gl"][idx],
                               color="lime", s=55, marker="^", zorder=8,
                               edgecolors="white", lw=0.5)

        for c in (cicli_p or []):
            ti = c.get("insert_complete")
            if ti:
                idx = np.argmin(np.abs(ref_kin["casi"] - ti))
                if idx < len(ref_kin["x_gl"]):
                    ax.scatter(ref_kin["x_gl"][idx], ref_kin["y_gl"][idx],
                               color="tomato", s=55, marker="v", zorder=8,
                               edgecolors="white", lw=0.5)

    ax.set_xlabel("X [mm]", color="#aaa", fontsize=9)
    ax.set_ylabel("Y [mm]", color="#aaa", fontsize=9)
    ax.tick_params(colors="#777")
    ax.grid(alpha=0.12, color="#555")
    for sp in ax.spines.values():
        sp.set_color("#333")

    legenda_traj = [Line2D([0], [0], color=BARVE[k], lw=2,
                            label=f"{OZNAKE[k]}  ({kin_multi[k]['pot_skupaj']:.0f} mm)")
                    for k in razpolozljive]
    faz_leg = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="lime",
               markersize=8, label="Vstavljanje"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="tomato",
               markersize=8, label="Pospravljanje"),
    ]
    ax.legend(handles=legenda_traj + faz_leg,
              fontsize=8, facecolor="#222", labelcolor="white",
              edgecolor="#444", loc="upper right")

    plt.tight_layout()
    if izhod_pot:
        plt.savefig(izhod_pot, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


# ── Graf 4: Normalizirani cikli ───────────────────────────────────────────

def narisi_normirani_cikli(kin_multi, cicli_v, cicli_p,
                           izhod_pot=None, naslov=""):
    """
    Hitrost po posameznih ciklih, normaliziran čas [0→1].
    2 stolpca (vstavljanje / pospravljanje) × N razpoložljivih točk.
    """
    razpolozljive = [k for k in ["TIP", "THUMB", "CENTER"] if kin_multi.get(k)]
    if not razpolozljive:
        return
    n = len(razpolozljive)

    fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    if naslov:
        fig.suptitle(naslov + " — Hitrost po ciklih (normirani čas)", fontsize=11,
                     color="white", y=0.99)

    if n == 1:
        axs = [axs]  # uniformni dostop

    def _seg(kin, t0, t1):
        mask = (kin["casi"] >= t0) & (kin["casi"] <= t1)
        if not mask.any():
            return None, None
        t = kin["casi"][mask]
        v = kin["v"][mask]
        return (t - t0) / max(t1 - t0, 1e-6), v

    for row, ime in enumerate(razpolozljive):
        kin   = kin_multi[ime]
        barva = BARVE[ime]

        # Vstavljanje
        ax_v = axs[row][0]
        ax_v.set_facecolor("#161616")
        for c in (cicli_v or []):
            t0 = c.get("pickup_start")
            t1 = c.get("insert_complete")
            if t0 and t1:
                tn, vn = _seg(kin, t0, t1)
                if tn is not None:
                    ax_v.plot(tn, vn, color=barva, alpha=0.55, lw=1.0)
        ax_v.set_title(f"{OZNAKE[ime]} — Vstavljanje", color="#ddd", fontsize=9)
        ax_v.set_ylabel("v [mm/s]", color="#bbb", fontsize=8)
        ax_v.axhline(0, color="#444", lw=0.6)
        ax_v.grid(alpha=0.12, color="#555")
        ax_v.tick_params(colors="#777")
        for sp in ax_v.spines.values(): sp.set_color("#333")

        # Pospravljanje
        ax_p = axs[row][1]
        ax_p.set_facecolor("#161616")
        for c in (cicli_p or []):
            t0 = c.get("pickup_start")
            t1 = c.get("insert_complete")
            if t0 and t1:
                tn, vn = _seg(kin, t0, t1)
                if tn is not None:
                    ax_p.plot(tn, vn, color=barva, alpha=0.55, lw=1.0)
        ax_p.set_title(f"{OZNAKE[ime]} — Pospravljanje", color="#ddd", fontsize=9)
        ax_p.axhline(0, color="#444", lw=0.6)
        ax_p.grid(alpha=0.12, color="#555")
        ax_p.tick_params(colors="#777")
        for sp in ax_p.spines.values(): sp.set_color("#333")

    axs[-1][0].set_xlabel("Normirani čas [0=začetek, 1=konec]",
                           color="#aaa", fontsize=8)
    axs[-1][1].set_xlabel("Normirani čas [0=začetek, 1=konec]",
                           color="#aaa", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if izhod_pot:
        plt.savefig(izhod_pot, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════
# GLAVNA FUNKCIJA: nariši vse grafe iz kin_multi
# ══════════════════════════════════════════════════════════════════════════

def narisi_vse(kin_multi, cicli_v=None, cicli_p=None,
               hom=None, izhod_predpona=None, naslov=""):
    """
    Naredi vse 4 grafe in vrne seznam ustvarjenih datotek.

    izhod_predpona → npr. "/workspace/results/patient_024/camP_0"
                     Grafi se shranijo kot:
                       camP_0_dva_pregled.png
                       camP_0_dva_podrobno.png
                       camP_0_dva_trajektorija.png
                       camP_0_dva_normirano.png
    """
    shranjene = []

    def pot(suffix):
        return f"{izhod_predpona}_{suffix}.png" if izhod_predpona else None

    narisi_dva_pregled(
        kin_multi, cicli_v, cicli_p,
        izhod_pot=pot("dva_pregled"), naslov=naslov)
    if pot("dva_pregled"):
        shranjene.append(pot("dva_pregled"))

    narisi_dva_podrobno(
        kin_multi, cicli_v, cicli_p,
        izhod_pot=pot("dva_podrobno"), naslov=naslov)
    if pot("dva_podrobno"):
        shranjene.append(pot("dva_podrobno"))

    narisi_trajektorije_2d(
        kin_multi, hom=hom, cicli_v=cicli_v, cicli_p=cicli_p,
        izhod_pot=pot("dva_trajektorija"), naslov=naslov)
    if pot("dva_trajektorija"):
        shranjene.append(pot("dva_trajektorija"))

    if cicli_v or cicli_p:
        narisi_normirani_cikli(
            kin_multi, cicli_v, cicli_p,
            izhod_pot=pot("dva_normirano"), naslov=naslov)
        if pot("dva_normirano"):
            shranjene.append(pot("dva_normirano"))

    return shranjene


# ══════════════════════════════════════════════════════════════════════════
# INTEGRACIJA: razširitev detect_combined.py
# ══════════════════════════════════════════════════════════════════════════

# V detect_combined.py — ProcesorCombined:
#
# 1. V __init__ dodaj logerje za vse 3 točke:
#       self._log_multi = {
#           "TIP":    {"casi": [], "traj": []},
#           "THUMB":  {"casi": [], "traj": []},
#           "CENTER": {"casi": [], "traj": []},
#       }
#
# 2. V SledilecRoke.zazaj() dodaj metodo zazaj_multi():
#       def zazaj_multi(self, frame):
#           """Vrne dict {TIP, THUMB, CENTER} → (px, py) ali None."""
#           rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#           res = self.hands.process(rgb)
#           if not res.multi_hand_landmarks:
#               return None, None
#           lm = res.multi_hand_landmarks[0]
#           h, w = frame.shape[:2]
#           def pt(idx):
#               p = lm.landmark[idx]
#               return (int(p.x * w), int(p.y * h))
#           tocke = {
#               "TIP":    pt(8),   # INDEX_FINGER_TIP
#               "THUMB":  pt(4),   # THUMB_TIP
#               "CENTER": pt(0),   # WRIST
#           }
#           return tocke, lm
#
# 3. V glavnem loopu (po zazaj_multi):
#       for ime, px_tocka in tocke_px.items():
#           mm_tocka = self.hom.v_mm(px_tocka) if self._hom_ok else None
#           if mm_tocka:
#               self._log_multi[ime]["casi"].append(t_s)
#               self._log_multi[ime]["traj"].append(mm_tocka)
#
# 4. Na koncu procesiraj(), po shranitvi obstoječih grafov:
#       from dva_grafi import izracunaj_dva_multi, narisi_vse, izvozi_csv_multi
#       kin_multi = izracunaj_dva_multi(self._log_multi, fps=fps)
#       predpona  = self.izhod_graf.replace("_graf.png", "")
#       narisi_vse(kin_multi, cicli_v, cicli_p,
#                  hom=self.hom,
#                  izhod_predpona=predpona,
#                  naslov=f"{ime} — kinematika")
#       izvozi_csv_multi(kin_multi,
#                        predpona.replace("_graf", "") + "_kinematika_multi.csv")


# ══════════════════════════════════════════════════════════════════════════
# CLI — direkten zagon iz JSON rezultatov
# ══════════════════════════════════════════════════════════════════════════

def _iz_json(json_pot, izhod_mapa=None, fps=25.0):
    """Zažene analizo iz JSON datoteke ki jo ustvari pipeline.py."""
    with open(json_pot, encoding="utf-8") as f:
        podatki = json.load(f)

    id_pac = podatki.get("id_pacienta", "?")
    videi  = podatki.get("videi", [])

    if izhod_mapa:
        os.makedirs(izhod_mapa, exist_ok=True)

    for r_video in videi:
        ime      = r_video.get("video", "?")
        casi_hom = r_video.get("casi_hom",  [])
        traj_mm  = r_video.get("traj_mm",   [])
        cicli_v  = r_video.get("cicli_vstavljanje",   [])
        cicli_p  = r_video.get("cicli_pospravljanje", [])

        if not casi_hom or not traj_mm:
            print(f"  {ime}: ni trajektorije, preskočeno")
            continue

        # V JSON je samo ena točka (center) — razširi na _multi format
        log_multi = {
            "TIP":    {"casi": casi_hom, "traj": traj_mm},
            "THUMB":  {"casi": [],       "traj": []},
            "CENTER": {"casi": casi_hom, "traj": traj_mm},
        }

        kin_multi = izracunaj_dva_multi(log_multi, fps=fps)

        predpona = os.path.join(izhod_mapa or os.path.dirname(json_pot), ime)
        naslov   = f"{id_pac} — {ime}"

        shranjene = narisi_vse(kin_multi, cicli_v, cicli_p,
                               izhod_predpona=predpona, naslov=naslov)

        # CSV
        csv_pot = predpona + "_kinematika.csv"
        izvozi_csv_multi(kin_multi, csv_pot)

        print(f"  {ime}:")
        for s in shranjene:
            print(f"    Graf: {os.path.basename(s)}")
        print(f"    CSV:  {os.path.basename(csv_pot)}")
        for ime_t, kin in kin_multi.items():
            if kin:
                print(f"    {OZNAKE[ime_t]:25s}  "
                      f"d={kin['pot_skupaj']:.0f}mm  "
                      f"v_max={kin['v_max']:.0f}mm/s  "
                      f"a_max={kin['a_max']:.0f}mm/s²")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="d/v/a kinematični grafi — kazalec, palec, center roke")
    parser.add_argument("--json",  type=str, required=True,
                        help="Pot do JSON iz pipeline.py")
    parser.add_argument("--izhod", type=str, default=None,
                        help="Izhodna mapa za grafe in CSV")
    parser.add_argument("--fps",   type=float, default=25.0)
    args = parser.parse_args()

    izhod = args.izhod or os.path.dirname(os.path.abspath(args.json))
    _iz_json(args.json, izhod, fps=args.fps)