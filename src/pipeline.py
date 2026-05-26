#!/usr/bin/env python3
# pipeline.py — 9HPT analiza, interaktivna vstopna točka
#
# Tok:
#   1. Vnos ID pacienta
#   2. Prikaz razpoložljivih videoposnetkov s preverjanjem veljavnosti
#   3. Izbira videa (ali vseh)
#   4. Tiha obdelava → jedrnat zaključni izpis
#   5. Izvoz JSON + kinematični CSV
#
# Zahteve:
#   pip install mediapipe opencv-python scipy matplotlib --break-system-packages

import os
import sys
import glob
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ===== KONFIGURACIJA =====

PODATKI_MAPA = "/data/Data"
IZHOD_MAPA   = "/workspace/results"
SRC_MAPA     = "/workspace/src"

if SRC_MAPA not in sys.path:
    sys.path.insert(0, SRC_MAPA)

# ===== UVOZI MODULOV =====

try:
    from detect_combined import ProcesorCombined, Strategija
    from analizator_y    import AnalizatorYOsi, analiziraj_iz_rezultatov
    from luknjice_led    import preveri_veljavnost_videa
    from dva_grafi       import izracunaj_dva
    MODULI_OK = True
except ImportError as e:
    print(f"[NAPAKA] Manjka modul: {e}")
    MODULI_OK = False

try:
    from csv_reader import preberi_csv_pacienta
    CSV_MODUL_OK = True
except ImportError:
    CSV_MODUL_OK = False


# ===== BARVE ZA TERMINAL =====

class B:
    """ANSI barve za terminal."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    ZELENA = "\033[92m"
    RUMENA = "\033[93m"
    RDECA  = "\033[91m"
    SIVA   = "\033[90m"
    CYAN   = "\033[96m"

def ok(s):    return f"{B.ZELENA}✓{B.RESET} {s}"
def warn(s):  return f"{B.RUMENA}⚠{B.RESET} {s}"
def err(s):   return f"{B.RDECA}✗{B.RESET} {s}"
def bold(s):  return f"{B.BOLD}{s}{B.RESET}"
def sivo(s):  return f"{B.SIVA}{s}{B.RESET}"


# ===== PREVERJANJE VIDEOPOSNETKOV =====

def preveri_videe(videi):
    """
    Preveri vsak video in vrne seznam z rezultatom preverjanja.
    Vrne: list dict { pot, ime, veljavno, razlog, trajanje }
    """
    rezultati = []
    for pot in videi:
        ime = os.path.basename(pot).replace(".mp4", "")

        # Najprej preveri, ali datoteka sploh obstaja
        if not os.path.isfile(pot):
            rezultati.append({
                "pot": pot, "ime": ime,
                "veljavno": False, "razlog": "DATOTEKA_NI_DOSEGLJIVA",
                "trajanje": 0
            })
            continue

        # Preberi osnovno informacijo o videu
        try:
            import cv2
            cap = cv2.VideoCapture(pot)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n_framov = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            trajanje = n_framov / fps if fps > 0 else 0
            cap.release()
        except Exception:
            rezultati.append({
                "pot": pot, "ime": ime,
                "veljavno": False, "razlog": "NAPAKA_BRANJA",
                "trajanje": 0
            })
            continue

        # Preveri vsebinsko veljavnost (LED blink)
        veljavno, razlog = preveri_veljavnost_videa(pot)
        rezultati.append({
            "pot": pot, "ime": ime,
            "veljavno": veljavno, "razlog": razlog,
            "trajanje": trajanje
        })

    return rezultati


def prikazi_seznam_videov(preverjen):
    """Izpiše oštevilčen seznam videoposnetkov z njihovim statusom."""
    print()
    print(bold("  Razpoložljivi videoposnetki:"))
    print()
    for i, v in enumerate(preverjen, 1):
        status = ok(f"[{v['trajanje']:.0f}s]") if v["veljavno"] else err(v["razlog"])
        print(f"  [{i}] {v['ime']:<35} {status}")
    print(f"  [0] Obdelaj vse veljavne videoposnetke")
    print()


# ===== ZANESLJIVOST ANALIZE =====

PRAG_MIN_ZATICEV = 7   # < 7 → napaka zaznave, priporoči drug kot kamere

def preveri_zanesljivost(n_vstavljanj, n_pospravljanj, ime_videa=""):
    """
    Vrne (zanesljivo: bool, sporocilo: str).
    Pogoj: vsaj 7 zatičev v vsaki aktivni fazi (< 7 → priporoči drug kot kamere).
    """
    napake = []

    if n_vstavljanj < PRAG_MIN_ZATICEV:
        napake.append(
            f"Vstavljanje: zaznano {n_vstavljanj}/9 "
            f"(minimum {PRAG_MIN_ZATICEV})"
        )
    if n_pospravljanj < PRAG_MIN_ZATICEV:
        napake.append(
            f"Pospravljanje: zaznano {n_pospravljanj}/9 "
            f"(minimum {PRAG_MIN_ZATICEV})"
        )

    if napake:
        msg = ("Napaka v zaznavi, poskusi z drugim pogledom kamere.\n  "
               + "\n  ".join(napake))
        return False, msg

    return True, "OK"


# ===== OBDELAVA ENEGA VIDEA =====

def _izracunaj_pot_geometricno(r, r_y=None):
    """
    Izračuna geometrijsko pot roke iz zaznanih ciklov.

    Za vsak cikel (vstavljanje/pospravljanje) vzamemo razdaljo med
    centrom posodice in ciljno luknjico. To je anatomsko pravilna pot
    brez akumulacije MediaPipe šuma.

    Vrne skupno pot v mm ali None če ni dovolj podatkov.
    """
    # Pridobi homografijo za pozicije luknjic in posodice
    hom = r.get("homografija")
    if hom is None:
        hom = r.get("hom")  # rezervno ime

    center_posodice = None
    luknjice_mm     = []

    if hom is not None:
        try:
            center_posodice = hom.center_posodice_mm
            luknjice_mm     = list(hom.get_luknjice_mm() or [])
        except Exception:
            pass

    # Pridobi cikle — najprej iz FSM homografije, potem iz Y-analizatorja
    cicli_v = r.get("cicli_vstavljanje", []) or []
    cicli_p = r.get("cicli_pospravljanje", []) or []

    # FSM cicli imajo target_hole indeks — to je najprimerneje za razdaljo
    pot = 0.0
    n_ciklov = 0

    if center_posodice and luknjice_mm:
        cp = np.array(center_posodice)

        for c in cicli_v:
            idx = c.get("target_hole")
            if idx is not None and 0 <= idx < len(luknjice_mm):
                luk = np.array(luknjice_mm[idx])
                pot += float(np.linalg.norm(luk - cp)) * 2  # posodica→luknjica→posodica
                n_ciklov += 1

        for c in cicli_p:
            idx = c.get("target_hole")
            if idx is not None and 0 <= idx < len(luknjice_mm):
                luk = np.array(luknjice_mm[idx])
                pot += float(np.linalg.norm(luk - cp)) * 2
                n_ciklov += 1

    # Rezervna metoda: povprečna razdalja posodica↔luknjice × število ciklov
    if n_ciklov == 0 and center_posodice and luknjice_mm:
        cp  = np.array(center_posodice)
        povp_razd = float(np.mean([np.linalg.norm(np.array(l) - cp)
                                    for l in luknjice_mm]))
        n_v = len(cicli_v) or (len(r_y.get("cicli_vstavljanje", [])) if r_y else 0)
        n_p = len(cicli_p) or (len(r_y.get("cicli_pospravljanje", [])) if r_y else 0)
        n_skupaj = n_v + n_p
        if n_skupaj > 0:
            pot = povp_razd * 2 * n_skupaj
            n_ciklov = n_skupaj

    return round(pot, 1) if n_ciklov > 0 else None


def analiziraj_video(pot_videa, izhod_mapa, fps_override=None, verbose=False):
    """
    Požene celoten pipeline za en video. Med izvajanjem ne izpisuje ničesar
    (razen v primeru kritične napake).
    Vrne dict z rezultati ali None ob napaki.
    """
    ime = os.path.basename(pot_videa).replace(".mp4", "")
    os.makedirs(izhod_mapa, exist_ok=True)

    izhod_graf  = os.path.join(izhod_mapa, f"{ime}_graf.png")
    izhod_video = os.path.join(izhod_mapa, f"{ime}_debug.mp4")

    # Preusmeritev izpisa v /dev/null med obdelavo
    import io, contextlib
    devnull = io.StringIO()

    try:
        with contextlib.redirect_stdout(devnull):
            procesor = ProcesorCombined(
                pot_videa   = pot_videa,
                razmik_mm   = 32,
                strategija  = Strategija.AUTO,
                izhod_video = izhod_video,
                izhod_graf  = izhod_graf,
                verbose     = False,
            )
            r = procesor.procesiraj()

        # Y-os analiza (tiho)
        with contextlib.redirect_stdout(devnull):
            r_y = analiziraj_iz_rezultatov(r, verbose=False)

        # Časovnik zatičev iz analizator_y_zone (pravilna definicija)
        # casi_zaticev = od blink → prehod meje[0] → prehod meje[1] → ...
        # To se bolj ujema z referenčnimi CSV časi kot movement_time
        try:
            from analizator_y import izracunaj_case_zaticev
            if r_y:
                led_obj  = r.get("led_stroj")
                # cas_zacetka/cas_konca sta v frameih — delimo s fps led_stroja
                _fps_led = getattr(led_obj, "fps", fps_override or 25.0)
                t_blink_on  = (led_obj.cas_zacetka / _fps_led
                               if led_obj and led_obj.cas_zacetka else None)
                t_blink_off = (led_obj.cas_konca / _fps_led
                               if led_obj and led_obj.cas_konca else None)
                casi_zaticev = izracunaj_case_zaticev(
                    r_y,
                    t_blink_on=t_blink_on,
                    t_blink_off=t_blink_off,
                    verbose=False)
            else:
                casi_zaticev = None
        except Exception as _e_cz:
            import traceback as _tb_cz
            print(f"  [DBG] casi_zaticev napaka: {_e_cz}")
            casi_zaticev = None

        # Kinematični grafi — po redirect bloku da vidimo morebitne napake
        if izhod_graf and r.get("hom_ok"):
            try:
                from dva_grafi import izracunaj_dva_multi, narisi_vse, izvozi_csv_multi
                log_multi = r.get("log_multi", {})
                fps_videa = fps_override or 25.0
                kin_multi = izracunaj_dva_multi(log_multi, fps=fps_videa)
                predpona  = izhod_graf.replace("_graf.png", "")
                cicli_v_kin = r_y.get("cicli_vstavljanje",  []) if r_y else r.get("cicli_vstavljanje",  [])
                cicli_p_kin = r_y.get("cicli_pospravljanje",[]) if r_y else r.get("cicli_pospravljanje",[])
                narisi_vse(kin_multi, cicli_v_kin, cicli_p_kin,
                           hom=r.get("homografija"),
                           izhod_predpona=predpona,
                           naslov=ime)
                izvozi_csv_multi(kin_multi, predpona + "_kinematika_multi.csv")
            except Exception as _e_dva:
                import traceback as _tb
                print(f"  [DVA] Napaka: {_e_dva}")
                _tb.print_exc()

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"napaka": str(e), "traceback": tb, "video": ime}

    if r_y:
        cicli_v = r_y.get("cicli_vstavljanje",  [])
        cicli_p = r_y.get("cicli_pospravljanje", [])
    else:
        cicli_v = r.get("cicli_vstavljanje",  [])
        cicli_p = r.get("cicli_pospravljanje", [])

    # Y-graf
    if r_y:
        try:
            izhod_y = izhod_graf.replace("_graf.png", "_graf_y.png")
            AnalizatorYOsi().narisi_graf(r_y, izhod_y)
        except Exception:
            pass

    # Kinematika d/v/a
    # traj_mm = TIP (za analizator_y) — ne primereno za kinematiko center roke
    # Vzamemo CENTER (WRIST) iz log_multi ki se polni neodvisno
    fps_videa = fps_override or 25.0
    casi_hom  = r.get("casi_hom", [])
    traj_mm   = r.get("traj_mm",  [])
    kin = None

    log_multi    = r.get("log_multi", {})
    center_casi  = log_multi.get("CENTER", {}).get("casi", [])
    center_traj  = log_multi.get("CENTER", {}).get("traj", [])

    # t_konec: obreži trajektorijo na konec testa (overlay to dela avtomatsko)
    led_obj     = r.get("led_stroj")
    t_konec_kin = (led_obj.cas_konca / fps_videa
                   if led_obj and led_obj.cas_konca else None)

    if len(center_casi) > 10:
        kin = izracunaj_dva(center_casi, center_traj,
                            fps=fps_videa, t_konec=t_konec_kin)
    elif len(casi_hom) > 10:
        kin = izracunaj_dva(casi_hom, traj_mm,
                            fps=fps_videa, t_konec=t_konec_kin)

    # Identifikacija luknjic iz Y in X amplitude
    rez_luk = None
    try:
        from analizator_luknjic import analiziraj_luknjice, narisi_mreze,                                        narisi_vrstni_red_na_casovnici,                                        izvozi_csv_luknjic
        _cicli_v = r_y.get("cicli_vstavljanje", []) if r_y else cicli_v
        _cicli_p = r_y.get("cicli_pospravljanje", []) if r_y else cicli_p
        if _cicli_v or _cicli_p:
            _xs = [p[0] for p in traj_mm]
            _ys = [p[1] for p in traj_mm]
            rez_luk = analiziraj_luknjice(
                _cicli_v, _cicli_p, list(casi_hom), _xs, _ys,
                verbose=False)
    except Exception:
        pass

    # LED stroj časi
    led = r.get("led_stroj")
    led_casi = {
        "skupni_cas":        getattr(led, "cas_testa_sekunde",       None) if led else None,
        "cas_vstavljanja":   getattr(led, "cas_vstavljanja_sekunde",  None) if led else None,
        "cas_pospravljanja": getattr(led, "cas_pospravljanja_sekunde", None) if led else None,
        "stevilo_zaticev":   getattr(led, "stevilo_zatičev",          None) if led else None,
        "roka":              getattr(led, "roka",                     None) if led else None,
    }

    # Časi posameznih zatičev
    # Primarna metoda: casi_zaticev iz izracunaj_case_zaticev
    # (od blink/prehoda_meje do naslednjega prehoda_meje)
    # Rezervna: movement_time iz Y-analizatorja
    if casi_zaticev and casi_zaticev.get("casi_v"):
        casi_vst  = [round(t, 3) for t in casi_zaticev["casi_v"]]
        casi_posp = [round(t, 3) for t in casi_zaticev["casi_p"]]
    else:
        casi_vst = []
        casi_posp = []
        for c in cicli_v:
            mt = c.get("movement_time")
            if mt:
                casi_vst.append(round(mt, 3))
        for c in cicli_p:
            mt = c.get("movement_time")
            if mt:
                casi_posp.append(round(mt, 3))

    return {
        "video":             ime,
        "pot_videa":         pot_videa,
        "hom_ok":            r.get("hom_ok", False),
        "px_per_mm":         r.get("px_per_mm"),
        "n_vstavljanj":      len(cicli_v),
        "n_pospravljanj":    len(cicli_p),
        "casi_vstavljanj":    casi_vst,
        "casi_pospravljanj":  casi_posp,
        "casi_zaticev":       casi_zaticev,
        "t_reakcija":         casi_zaticev.get("t_reakcija") if casi_zaticev else None,
        "t_skupaj_v":         casi_zaticev.get("t_skupaj_v") if casi_zaticev else None,
        "t_skupaj_p":         casi_zaticev.get("t_skupaj_p") if casi_zaticev else None,
        # Alias za evalvacija.py kompatibilnost
        "mt_vstavljanje_y":   casi_vst,
        "mt_pospravljanje_y": casi_posp,
        "mt_skupaj":          sum(casi_vst) + sum(casi_posp),
        "led":               led_casi,
        "kin":               {
            "pot_skupaj":  kin.get("pot_skupaj")  if kin else None,
            "displacement": kin.get("displacement") if kin else None,
            "path_ratio":  kin.get("path_ratio")  if kin else None,
            "v_95":        kin.get("v_95")        if kin else None,
            "a_95":        kin.get("a_95")        if kin else None,
            "v_max":       kin.get("v_max")       if kin else None,
            "a_max":       kin.get("a_max")       if kin else None,
        },
        "cicli_vstavljanje":   cicli_v,
        "cicli_pospravljanje": cicli_p,
        "luknjice":            rez_luk,
        "casi_hom":            list(np.array(casi_hom).tolist()) if len(casi_hom) > 0 else [],
        "traj_mm":             [list(t) for t in traj_mm] if traj_mm else [],
    }

    # Grafi in CSV za luknjice
    if izhod_graf and rez_luk:
        try:
            from analizator_luknjic import narisi_mreze,                                            narisi_vrstni_red_na_casovnici,                                            izvozi_csv_luknjic
            _predp = izhod_graf.replace("_graf.png", "")
            narisi_mreze(rez_luk,
                izhod_pot=_predp + "_luknjice_mreza.png",
                naslov=ime)
            narisi_vrstni_red_na_casovnici(
                rez_luk, list(casi_hom),
                [p[1] for p in traj_mm], [p[0] for p in traj_mm],
                izhod_pot=_predp + "_luknjice_casovnica.png",
                naslov=ime)
            izvozi_csv_luknjic(rez_luk, _predp + "_luknjice.csv")
        except Exception:
            pass

    return rezultat


# ===== KINEMATIČNI CSV =====

def izvozi_kin_csv(r_video, izhod_mapa):
    """
    Izvozi kinematične parametre v CSV datoteko.
    Stolpci: frame, t_s, x_mm, y_mm, d_mm, v_mm_s, a_mm_s2
    """
    # Vzami CENTER iz log_multi — ne TIP iz traj_mm
    log_multi = r_video.get("log_multi", {})
    casi = log_multi.get("CENTER", {}).get("casi", [])
    traj = log_multi.get("CENTER", {}).get("traj", [])
    if not casi or not traj:
        # Rezervno: TIP
        casi = r_video.get("casi_hom", [])
        traj = r_video.get("traj_mm",  [])
    if not casi or not traj:
        return None

    fps = 25.0  # privzeto
    kin = izracunaj_dva(casi, traj, fps=fps)
    if kin is None:
        return None

    ime    = r_video["video"]
    pot    = os.path.join(izhod_mapa, f"{ime}_kinematika.csv")
    n      = min(len(kin["casi"]), len(kin["d"]), len(kin["v"]), len(kin["a"]),
                 len(kin["x_gl"]), len(kin["y_gl"]))

    with open(pot, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x_mm", "y_mm",
                    "d_mm_kumulativno", "v_mm_s", "a_mm_s2"])
        for i in range(n):
            w.writerow([
                i,
                round(float(kin["casi"][i]), 4),
                round(float(kin["x_gl"][i]),  2),
                round(float(kin["y_gl"][i]),  2),
                round(float(kin["d"][i]),      2),
                round(float(kin["v"][i]),      2),
                round(float(kin["a"][i]),      2),
            ])

    return pot


# ===== IZPIS REZULTATOV =====

def izpisi_rezultate(r_video, csv_podatki=None):
    """Jedrnat izpis po končani obdelavi videa."""
    ime  = r_video["video"]
    led  = r_video["led"]
    kin  = r_video["kin"]

    print()
    print(bold(f"  ── {ime} ──────────────────────────────────────"))

    # Zanesljivost
    n_v = r_video["n_vstavljanj"]
    n_p = r_video["n_pospravljanj"]
    zanes, sporocilo = preveri_zanesljivost(n_v, n_p)
    if not zanes:
        print(f"  {err(sporocilo)}")
    else:
        print(f"  {ok('Analiza zanesljiva')}")

    # Homografija
    if not r_video.get("hom_ok"):
        print(f"  {warn('Homografija ni uspela — kinematika v pikslih')}")

    # Roka in diagnoza
    roka = led.get("roka") if led else None
    print(f"  Roka:                   {bold(roka or '?')}")

    # Čas testa
    cas_skupaj = led.get("skupni_cas") if led else None
    cas_vst    = led.get("cas_vstavljanja") if led else None
    cas_posp   = led.get("cas_pospravljanja") if led else None
    print(f"  Čas testa:              {bold(f'{cas_skupaj:.2f} s') if cas_skupaj else sivo('ni podatka')}")
    print(f"    ↳ vstavljanje:        {f'{cas_vst:.2f} s'  if cas_vst  else sivo('?')}")
    print(f"    ↳ pospravljanje:      {f'{cas_posp:.2f} s' if cas_posp else sivo('?')}")

    # Zatički
    st_zat = led.get("stevilo_zaticev") if led else None
    print(f"  Zatički (LED):          {st_zat or sivo('?')}")
    print(f"  Zaznano vstavljanj:     {n_v}/9  {'✓' if n_v == 9 else warn('')}")
    print(f"  Zaznano pospravljanj:   {n_p}/9  {'✓' if n_p == 9 else warn('')}")

    # Časi posameznih zatičev
    casi_v = r_video.get("casi_vstavljanj", [])
    casi_p = r_video.get("casi_pospravljanj", [])
    t_reakcija = r_video.get("t_reakcija")
    if t_reakcija is not None:
        print(f"  Reakcijski čas:         {t_reakcija:.3f} s  (blink → 1. pobiranje)")
    if casi_v:
        casi_str = "  ".join(f"{t:.2f}" for t in casi_v)
        print(f"  Časi vstavljanj [s]:    {casi_str}")
    if casi_p:
        casi_str = "  ".join(f"{t:.2f}" for t in casi_p)
        print(f"  Časi pospravljanj [s]:  {casi_str}")
    t_v = r_video.get("t_skupaj_v")
    t_p = r_video.get("t_skupaj_p")
    if t_v and t_p:
        print(f"  Skupaj vstavljanje:     {t_v:.2f} s")
        print(f"  Skupaj pospravljanje:   {t_p:.2f} s")

    # Kinematika
    if kin.get("pot_skupaj") is not None:
        print(f"  Kinematika zapestja (WRIST, 2D):")
        print(f"    ↳ pot skupaj:         {kin['pot_skupaj']:.0f} mm")
        print(f"    ↳ hitrost 95p:        {kin['v_95']:.0f} mm/s")
        print(f"    ↳ pospešek 95p:       {kin['a_95']:.0f} mm/s²")
        # Opomba: absolutna pot je okvirna vrednost (markerless sistem)
        # Zanesljiva za relativno primerjavo med pacienti, ne za abs. norme

    # Primerjava z CSV
    if csv_podatki:
        _primerjaj_z_csv(r_video, csv_podatki)

    print()


def _primerjaj_z_csv(r_video, csv_podatki):
    """Kratka primerjava izmerjenega s CSV referenco."""
    ime = r_video["video"]
    skupni_casi = csv_podatki.get("skupni_casi", {})
    meta        = csv_podatki.get("metadata", {})

    # Določi kateri ključ v CSV ustreza videu
    if "camP_0" in ime or "camP_1" in ime:
        kljuc = "P1"
    elif "camP_2" in ime:
        kljuc = "P2"
    elif "camS_0" in ime or "camS_1" in ime:
        kljuc = "S1"
    elif "camS_2" in ime:
        kljuc = "S2"
    else:
        return

    ref_cas = skupni_casi.get(kljuc)
    if ref_cas is None:
        return

    izmerjeno = r_video["led"].get("skupni_cas") if r_video.get("led") else None
    if izmerjeno:
        napaka = izmerjeno - ref_cas
        napaka_pct = 100 * napaka / ref_cas if ref_cas else 0
        print(f"  Primerjava CSV ({kljuc}):   izmerjeno {izmerjeno:.2f}s  "
              f"ref {ref_cas:.2f}s  ({napaka_pct:+.1f}%)")

    print(f"  Metadata:               spol={meta.get('spol','?')}  "
          f"diagnoza={meta.get('diagnoza','?')}  "
          f"roka={meta.get('roka','?')}")


def izpisi_glavo(id_pacienta, csv_podatki=None):
    """Izpiše glavo z osnovnimi podatki pacienta."""
    print()
    print("  " + "─" * 58)
    print(f"  {bold('9HPT ANALIZA')}  ·  pacient: {bold(id_pacienta)}")
    print("  " + "─" * 58)

    if csv_podatki:
        meta = csv_podatki.get("metadata", {})
        skupni = csv_podatki.get("skupni_casi", {})
        print(f"  Spol:        {meta.get('spol','?')}")
        print(f"  Diagnoza:    {meta.get('diagnoza','?')}")
        print(f"  Dom. roka:   {meta.get('roka','?')}")
        print(f"  Datum:       {meta.get('datum','?')}")
        if skupni:
            casi_str = "  ".join(f"{k}={v:.1f}s" for k, v in skupni.items() if v)
            print(f"  Ref. časi:   {casi_str}")
    print()


# ===== SHRANJEVANJE =====

def shrani_json(podatki, pot):
    """Shrani dict v JSON z numpy-safe serializacijo."""
    def serialize(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        raise TypeError(f"Neznani tip: {type(obj)}")

    with open(pot, "w", encoding="utf-8") as f:
        json.dump(podatki, f, indent=2, ensure_ascii=False, default=serialize)


# ===== INTERAKTIVNI VMESNIK =====

def vprasaj_pacienta(pot_podatkov):
    """Vpraša za ID pacienta in ga validira."""
    mape = sorted([
        d for d in os.listdir(pot_podatkov)
        if os.path.isdir(os.path.join(pot_podatkov, d))
        and d.startswith("patient_")
    ])

    print()
    print(bold("  Razpoložljivi pacienti:"))
    for i, m in enumerate(mape, 1):
        print(f"    [{i}] {m}")
    print()

    while True:
        vnos = input("  Vnesi ID pacienta (npr. patient_024) ali številko: ").strip()
        if not vnos:
            continue
        # Številka
        if vnos.isdigit():
            idx = int(vnos) - 1
            if 0 <= idx < len(mape):
                return mape[idx]
            else:
                print(err(f"  Neveljavna številka. Veljavno: 1–{len(mape)}"))
                continue
        # Direktni ID
        if os.path.isdir(os.path.join(pot_podatkov, vnos)):
            return vnos
        print(err(f"  Pacient '{vnos}' ne obstaja."))


def vprasaj_video(preverjen):
    """
    Izpiše seznam videoposnetkov in vpraša za izbiro.
    Vrne list izbranih rezultatov preverjanja.
    """
    prikazi_seznam_videov(preverjen)

    veljavni = [v for v in preverjen if v["veljavno"]]
    n_vseh   = len(preverjen)

    if not veljavni:
        print(err("  Ni veljavnih videoposnetkov za analizo."))
        return []

    while True:
        vnos = input(f"  Izbira (1–{n_vseh}, 0=vsi veljavni): ").strip()
        if not vnos.isdigit():
            continue
        n = int(vnos)
        if n == 0:
            print(f"  {ok(f'Izbranih {len(veljavni)} veljavnih videoposnetkov')}")
            return veljavni
        if 1 <= n <= n_vseh:
            izbran = preverjen[n - 1]
            if not izbran["veljavno"]:
                print(warn(f"  Videoposnetek [{n}] ni veljavno: {izbran['razlog']}"))
                nadaljuj = input("  Kljub temu obdelaj? [d/N]: ").strip().lower()
                if nadaljuj != "d":
                    continue
            return [preverjen[n - 1]]
        print(err(f"  Vpiši številko od 0 do {n_vseh}."))


# ===== NAPREDEK =====

def prikazi_napredek(ime, trenutni, skupaj):
    """Enostavna vrstica napredka."""
    pct = int(100 * trenutni / max(skupaj, 1))
    bloki = int(pct / 5)
    bar = "█" * bloki + "░" * (20 - bloki)
    print(f"\r  [{bar}] {pct:3d}%  {ime[:40]:<40}", end="", flush=True)


# ===== GLAVNA ANALIZA =====

def analiziraj_pacienta_interaktivno(
        id_pacienta,
        pot_podatkov = PODATKI_MAPA,
        izhod_mapa   = IZHOD_MAPA,
        izbrani_videi = None,   # list poti; None = interaktivno
):
    """
    Celoten pipeline za enega pacienta.
    Če izbrani_videi ni podan, vpraša interaktivno.
    """
    pot_pacienta = os.path.join(pot_podatkov, id_pacienta)
    izhod_pac    = os.path.join(izhod_mapa, id_pacienta)
    os.makedirs(izhod_pac, exist_ok=True)

    # Preberi CSV
    csv_podatki = None
    if CSV_MODUL_OK:
        try:
            csv_podatki = preberi_csv_pacienta(pot_podatkov, id_pacienta)
        except Exception:
            pass

    # Izpiši glavo
    izpisi_glavo(id_pacienta, csv_podatki)

    # Poišči videoposnetke
    vse_poti = sorted(glob.glob(os.path.join(pot_pacienta, "*.mp4")))
    if not vse_poti:
        print(err(f"  Ni .mp4 datotek v {pot_pacienta}"))
        return None

    # Preveri veljavnost
    print(f"  Preverjam {len(vse_poti)} videoposnetkov ...", end="", flush=True)
    preverjen = preveri_videe(vse_poti)
    print(f"\r  {ok(f'Preverjanje končano ({len(vse_poti)} videoposnetkov)')}")

    # Izbira videoposnetkov
    if izbrani_videi is not None:
        # CLI način — filtriraj po podanih poteh
        izbrani = [v for v in preverjen if v["pot"] in izbrani_videi]
        # Pokaži seznam za info
        prikazi_seznam_videov(preverjen)
    else:
        # Interaktivni način
        izbrani = vprasaj_video(preverjen)

    if not izbrani:
        return None

    # Obdelava
    rezultati_videov = []
    skupaj = len(izbrani)

    print()
    print(bold(f"  Obdelava {skupaj} videoposnetkov ..."))
    print()

    for i, v_info in enumerate(izbrani, 1):
        prikazi_napredek(v_info["ime"], i - 1, skupaj)

        r = analiziraj_video(v_info["pot"], izhod_pac)
        print(f"\r  {'✓' if r and not r.get('napaka') else '✗'} "
              f"[{i}/{skupaj}] {v_info['ime']:<45}")

        if r and r.get("napaka"):
            print(f"       {err(r['napaka'])}")
            tb = r.get("traceback", "")
            if tb:
                # Izpiši samo zadnje 3 vrstice traceback-a
                vrstice = [l for l in tb.strip().split("\n") if l.strip()]
                for l in vrstice[-6:]:
                    print(f"       {B.SIVA}{l}{B.RESET}")
            continue
        if r is None:
            print(f"       {err('Obdelava ni uspela')}")
            continue

        # Kinematični CSV
        kin_csv = izvozi_kin_csv(r, izhod_pac)

        # Izpis rezultatov
        izpisi_rezultate(r, csv_podatki)

        # Zanesljivost
        zanes, sporocilo = preveri_zanesljivost(
            r["n_vstavljanj"], r["n_pospravljanj"])
        r["zanesljivo"] = zanes
        r["zanesljivost_sporocilo"] = sporocilo
        r["kin_csv"] = kin_csv

        rezultati_videov.append(r)

    if not rezultati_videov:
        print(err("  Ni uspešno obdelanih videoposnetkov."))
        return None

    # Skupni JSON
    n_v = sum(r["n_vstavljanj"]   for r in rezultati_videov)
    n_p = sum(r["n_pospravljanj"] for r in rezultati_videov)
    n_vid = len(rezultati_videov)

    rezultat = {
        "id_pacienta":   id_pacienta,
        "datum_analize": datetime.now().isoformat(timespec="seconds"),
        "csv_ok":        csv_podatki is not None,
        "povzetek": {
            "n_videov":        n_vid,
            "n_vstavljanj":    n_v,
            "n_pospravljanj":  n_p,
            "tocnost_v":       round(n_v / (9 * n_vid), 3) if n_vid else 0,
            "tocnost_p":       round(n_p / (9 * n_vid), 3) if n_vid else 0,
        },
        "metadata": csv_podatki.get("metadata", {}) if csv_podatki else {},
        "skupni_casi_ref": csv_podatki.get("skupni_casi", {}) if csv_podatki else {},
        "videi": rezultati_videov,
    }

    json_pot = os.path.join(izhod_pac, f"{id_pacienta}_rezultati.json")
    shrani_json(rezultat, json_pot)

    # Zaključni povzetek
    print()
    print("  " + "─" * 58)
    toc_v = 100 * rezultat["povzetek"]["tocnost_v"]
    toc_p = 100 * rezultat["povzetek"]["tocnost_p"]
    print(f"  {bold('POVZETEK')}  ·  {id_pacienta}")
    print(f"  Vstavljanje:    {n_v}/{9*n_vid}  ({toc_v:.0f}%)")
    print(f"  Pospravljanje:  {n_p}/{9*n_vid}  ({toc_p:.0f}%)")
    print(f"  JSON:  {json_pot}")
    print("  " + "─" * 58)
    print()

    # Avtomatska evalvacija — samo če CSV referenčni podatki obstajajo
    if csv_podatki is not None:
        try:
            from evalvacija import evalviraj_pacienta
            print(f"  {bold('EVALVACIJA')}  (primerjava z referenčnimi CSV)")
            evalviraj_pacienta(
                rezultati_json_pot = json_pot,
                csv_podatki        = csv_podatki,
                izhod_mapa         = izhod_pac,
                verbose            = True,
            )
        except ImportError:
            pass   # evalvacija.py ni v src/ — preskoči tiho
        except Exception as _e_eval:
            print(f"  {warn(f'Evalvacija ni uspela: {_e_eval}')}")

    return rezultat


# ===== CLI VSTOPNA TOČKA =====

def main():
    parser = argparse.ArgumentParser(
        description="9HPT pipeline — analiza testa devetih zatičev",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--pacient",  type=str, default=None,
                        help="ID pacienta (npr. patient_024)\n"
                             "Če ni podan, program vpraša interaktivno.")
    parser.add_argument("--video",    type=str, default=None,
                        help="Pot do specifičnega videa (neobvezno)")
    parser.add_argument("--podatki",  type=str, default=PODATKI_MAPA)
    parser.add_argument("--izhod",    type=str, default=IZHOD_MAPA)

    args = parser.parse_args()

    if not MODULI_OK:
        print(err("Manjkajo moduli. Preveri pot do src/ in namestitvene odvisnosti."))
        sys.exit(1)

    # Določi pacienta
    id_pacienta = args.pacient
    if not id_pacienta:
        try:
            id_pacienta = vprasaj_pacienta(args.podatki)
        except KeyboardInterrupt:
            print("\n  Prekinjeno.")
            sys.exit(0)

    # Določi videoposnetke
    izbrani_videi = None
    if args.video:
        if not os.path.isfile(args.video):
            print(err(f"Video ne obstaja: {args.video}"))
            sys.exit(1)
        izbrani_videi = [args.video]

    try:
        analiziraj_pacienta_interaktivno(
            id_pacienta    = id_pacienta,
            pot_podatkov   = args.podatki,
            izhod_mapa     = args.izhod,
            izbrani_videi  = izbrani_videi,
        )
    except KeyboardInterrupt:
        print("\n  Prekinjeno.")
        sys.exit(0)


if __name__ == "__main__":
    main()