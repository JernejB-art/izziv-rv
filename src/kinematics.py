# kinematics.py
# Iz 2D trajektorije centra roke izračuna kinematične parametre:
# pot d(t), hitrost v(t) in pospešek a(t).
# Signal pred izračunom zgladimo z Savitzky-Golay filtrom.
# Na 1D projekciji (x-os) zaznamo faze testa: pobiranje in vstavljanje zatičev.

import numpy as np
from scipy.signal import savgol_filter, find_peaks

# Indeksi MediaPipe točk, ki definirajo center roke
# 0=zapestje, 5=kazalec MCP, 9=sredinec MCP, 13=prstanec MCP, 17=mezinec MCP
CENTER_TOCKE = [0, 5, 9, 13, 17]

def izracun_center_roke(landmarks, sirina, visina):
    """
    Iz 21 MediaPipe točk izračuna center roke kot povprečje
    zapestja in MCP sklepov (sklepi, kjer se prsti priključijo na dlan) — bolj stabilen kot povprečje vseh točk.

    landmarks -> MediaPipe objekt z .landmark[]
    sirina, visina -> dimenzije sličice v pikslih
    vrne -> (x, y) koordinati centra v pikslih
    """
    x = np.mean([landmarks.landmark[i].x * sirina for i in CENTER_TOCKE])
    y = np.mean([landmarks.landmark[i].y * visina for i in CENTER_TOCKE])
    return x, y

def glajenje_signal(signal, okno=11, stopnja=3):
    """
    Savitzky-Golay filter za glajenje trajektorije.
    Ohrani obliko signala (vrhove, doline) bolje kot drseče povprečje.

    signal -> 1D numpy array
    okno   -> dolžina okna (mora biti liho število)
    stopnja -> stopnja polinoma (manjša = bolj gladko)
    vrne   -> zglajen signal
    """
    # Okno ne sme biti večje od signala
    okno = min(okno, len(signal) - (1 if len(signal) % 2 == 0 else 0))
    if okno % 2 == 0:
        okno -= 1
    return savgol_filter(signal, okno, stopnja)

def izracun_kinematika(pozicije, fps):
    """
    Iz 2D trajektorije izračuna pot, hitrost in pospešek.

    pozicije -> seznam (x, y) koordinat za vsak frame
    fps      -> hitrost videa v sličicah na sekundo
    vrne     -> slovar z numpy arrayi za vse parametre
    """
    pozicije = np.array(pozicije)
    dt = 1.0 / fps

    # Zgladimo x in y trajektorijo pred izračunom
    x_gladko = glajenje_signal(pozicije[:, 0])
    y_gladko = glajenje_signal(pozicije[:, 1])
    pozicije_gladko = np.column_stack([x_gladko, y_gladko])

    # Pot: evklidska razdalja med zaporednimi točkami
    # sqrt(dx² + dy²) za vsak korak
    razdalje = np.linalg.norm(np.diff(pozicije_gladko, axis=0), axis=1)
    pot = np.cumsum(razdalje)

    # Hitrost: razdalja / čas
    hitrost = razdalje / dt

    # Pospešek: sprememba hitrosti / čas
    pospesek = np.diff(hitrost) / dt

    # Časovne osi
    cas_pot = np.arange(1, len(pot) + 1) * dt
    cas_hitrost = np.arange(1, len(hitrost) + 1) * dt
    cas_pospesek = np.arange(2, len(pospesek) + 2) * dt

    return {
        "pozicije": pozicije_gladko,
        "cas_pot": cas_pot,
        "pot": pot,
        "cas_hitrost": cas_hitrost,
        "hitrost": hitrost,
        "cas_pospesek": cas_pospesek,
        "pospesek": pospesek
    }

def zaznava_faze_testa(pozicije, fps):
    """
    Na 1D projekciji gibanja (x-os) zazna faze testa:
    - vrhovi (maxima) → pobiranje zatiča iz posodice
    - doline (minima) → vstavljanje zatiča v luknjico

    pozicije -> numpy array (N, 2) zglajena trajektorija
    fps      -> hitrost videa
    vrne     -> slovar z indeksi in časi vrhov in dolin
    """
    # Projekcija na x-os (gibanje levo-desno je glavno gibanje v testu)
    x = pozicije[:, 0]
    x_gladko = glajenje_signal(x, okno=21)

    dt = 1.0 / fps

    # Zaznava vrhov (pobiranje) - minimalna razdalja med vrhovi: 1.5s
    vrhovi, _ = find_peaks(x_gladko, distance=int(1.5 * fps))

    # Zaznava dolin (vstavljanje) - iščemo vrhove na negativnem signalu
    doline, _ = find_peaks(-x_gladko, distance=int(1.5 * fps))

    return {
        "vrhovi_idx": vrhovi,
        "doline_idx": doline,
        "vrhovi_cas": vrhovi * dt,
        "doline_cas": doline * dt,
        "stevilo_pobiranje": len(vrhovi),
        "stevilo_vstavljanje": len(doline),
        "x_signal": x_gladko
    }