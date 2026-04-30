# kinematics.py
# Iz 2D trajektorije centra roke izračuna kinematične parametre:
# pot d(t), hitrost v(t) in pospešek a(t).
# Signal pred izračunom zgladimo z Savitzky-Golay filtrom.
# Na 1D projekciji (x-os) zaznamo faze testa: pobiranje in vstavljanje zatičev.

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

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

def glajenje_signal(signal, fps, cutoff=5):
    """
    Butterworth low-pass filter za glajenje trajektorije.
    Fizikalno utemeljen za biomedicinske signale — gibanje roke
    pri 9HPT testu ni hitrejše od 2 Hz, cutoff=5 Hz zagotavlja
    da ohranimo vse relevantne gibe in odstranimo šum.

    signal -> 1D numpy array
    fps    -> hitrost videa v sličicah na sekundo
    cutoff -> mejna frekvenca v Hz (privzeto 5 Hz)
    vrne   -> zglajen signal
    """
    nyq = fps / 2  # Nyquistova frekvenca = polovica fps
    b, a = butter(4, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

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
    x_gladko = glajenje_signal(pozicije[:, 0], fps)
    y_gladko = glajenje_signal(pozicije[:, 1], fps)
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

def najdi_aktivno_obmocje(gibanje_gladko, fps, okno_s=2.0):
    """
    Najde začetek in konec aktivnega gibanja z drsenjem variance.
    Izreže mirovanje roke pred in po testu.
    okno_s -> dolžina okna v sekundah za izračun lokalne variance
    """
    okno = int(okno_s * fps)
    n = len(gibanje_gladko)
    
    # Izračunaj lokalno varianco z drsečim oknom
    varianca = np.array([
        np.var(gibanje_gladko[max(0, i-okno):i+okno])
        for i in range(n)
    ])
    
    # Prag = 20% maksimalne variance
    prag = np.max(varianca) * 0.2
    aktivno = varianca > prag
    
    aktivni_indeksi = np.where(aktivno)[0]
    if len(aktivni_indeksi) == 0:
        return 0, n
    
    zacetek = max(0, aktivni_indeksi[0])
    konec = min(n, aktivni_indeksi[-1])
    return zacetek, konec

def filtriraj_outlierje(indeksi, signal, prag=2.0):
    """
    Odstrani lažne vrhove/doline ki so statistični outlierji.
    prag -> koliko standardnih deviacij od mediane je še sprejemljivo
    """
    if len(indeksi) == 0:
        return indeksi
    
    visine = np.abs(signal[indeksi])
    mediana = np.median(visine)
    std = np.std(visine)
    
    # Ohrani samo tiste ki so znotraj praga
    maska = np.abs(visine - mediana) < prag * std
    return indeksi[maska]

# Indeks MediaPipe točke za konico kazalca
KAZALEC_TIP = 8

def izracun_kazalec(landmarks, sirina, visina):
    """
    Vrne koordinate konice kazalca (točka 8 v MediaPipe).
    Bolj občutljiva na fine gibe kot center roke.

    landmarks -> MediaPipe objekt z .landmark[]
    sirina, visina -> dimenzije sličice v pikslih
    vrne -> (x, y) koordinati konice kazalca v pikslih
    """
    x = landmarks.landmark[KAZALEC_TIP].x * sirina
    y = landmarks.landmark[KAZALEC_TIP].y * visina
    return x, y

def ohrani_lokalne_ekstreme(indeksi, signal, fps, okno_s=0.8):
    """
    Za vsak zaznani vrh/dolino preveri ali je res lokalni ekstrem
    v širši okolici (okno_s sekund). Zavrže lažne dvojnike.

    indeksi -> numpy array indeksov vrhov/dolin
    signal  -> 1D signal
    fps     -> hitrost videa
    okno_s  -> širina okolice v sekundah
    """
    if len(indeksi) == 0:
        return indeksi

    okno = int(okno_s * fps)
    ohranjeni = []

    for idx in indeksi:
        # Določi okolico
        zac = max(0, idx - okno)
        kon = min(len(signal), idx + okno)
        okolica = signal[zac:kon]

        # Vrh je veljaven samo če je maksimum v svoji okolici
        lokalni_max = zac + np.argmax(np.abs(okolica))
        if abs(lokalni_max - idx) < int(0.3 * fps):
            ohranjeni.append(idx)

    return np.array(ohranjeni)

def izmenjuj_ekstreme(vrhovi, doline, zacni_z_vrhom=True):
    """
    Zagotovi pravilno izmenjevanje vrhov in dolin.
    Test se vedno začne s pobiranjem (vrh) in konča z vstavljanjem (dolina).

    vrhovi       -> array indeksov vrhov (pobiranje)
    doline       -> array indeksov dolin (vstavljanje)
    zacni_z_vrhom -> True = začni s pobiranjem
    vrne         -> (filtrirani_vrhovi, filtrirane_doline)
    """
    if len(vrhovi) == 0 or len(doline) == 0:
        return vrhovi, doline

    # Združi vse ekstreme v eno zaporedje s oznako tipa
    # 1 = vrh (pobiranje), -1 = dolina (vstavljanje)
    vsi = [(idx, 1) for idx in vrhovi] + [(idx, -1) for idx in doline]
    vsi = sorted(vsi, key=lambda x: x[0])

    # Ohrani samo pravilno izmenjevanje
    ohranjeni = []
    zadnji_tip = -1 if zacni_z_vrhom else 1  # začnemo nasprotno da prvi element uspe

    for idx, tip in vsi:
        if tip != zadnji_tip:
            ohranjeni.append((idx, tip))
            zadnji_tip = tip

    filtrirani_vrhovi = np.array([idx for idx, tip in ohranjeni if tip == 1])
    filtrirane_doline = np.array([idx for idx, tip in ohranjeni if tip == -1])

    return filtrirani_vrhovi, filtrirane_doline

def filtriraj_pozicije(pozicije, prag_std=3.0):
    """
    Odstrani pozicije ki so statistični outlierji.
    Nadomesti jih z interpolacijo sosednjih vrednosti.
    Prepreči da bi napačne zaznave pokvarile SVD analizo.

    pozicije -> numpy array (N, 2)
    prag_std -> koliko standardnih deviacij je še sprejemljivo
    vrne     -> počiščene pozicije
    """
    pozicije = np.array(pozicije, dtype=float)
    
    for os in range(2):
        signal = pozicije[:, os]
        mediana = np.median(signal)
        std = np.std(signal)
        
        # Zazna outlierje
        outlierji = np.abs(signal - mediana) > prag_std * std
        
        # Interpoliraj outlierje
        indeksi = np.arange(len(signal))
        pozicije[:, os] = np.interp(
            indeksi,
            indeksi[~outlierji],
            signal[~outlierji]
        )
    
    return pozicije

def zaznava_faze_testa(pozicije, fps):
    dt = 1.0 / fps

    # Počisti outlier pozicije pred SVD
    pozicije = filtriraj_pozicije(pozicije)
    # Glavna os gibanja z SVD
    center = pozicije - np.mean(pozicije, axis=0)
    _, _, Vt = np.linalg.svd(center)
    gibanje_1d = center @ Vt[0]
    gibanje_gladko = glajenje_signal(gibanje_1d, fps)

    sredina = np.median(gibanje_gladko)

    # Izreži mirovanje
    zacetek, konec = najdi_aktivno_obmocje(gibanje_gladko, fps)
    gibanje_aktivno = gibanje_gladko[zacetek:konec]

    # Zaznava vseh ekstremov z minimalno razdaljo 0.5s
    # Izračunaj prominence samo na aktivnem območju
    razpon_aktivno = np.max(gibanje_aktivno) - np.min(gibanje_aktivno)

    vsi_vrhovi, _ = find_peaks(gibanje_aktivno,
                                distance=int(0.8 * fps),
                                prominence=razpon_aktivno * 0.05)

    vse_doline, _ = find_peaks(-gibanje_aktivno,
                                distance=int(0.8 * fps),
                                prominence=razpon_aktivno * 0.05)

    vsi_vrhovi = vsi_vrhovi + zacetek
    vse_doline = vse_doline + zacetek

    # Samo izmenjavanje brez cone filtra
    vrhovi, doline = izmenjuj_ekstreme(vsi_vrhovi, vse_doline)

    return {
        "vrhovi_idx": vrhovi,
        "doline_idx": doline,
        "vrhovi_cas": vrhovi * dt,
        "doline_cas": doline * dt,
        "stevilo_pobiranje": len(vrhovi),
        "stevilo_vstavljanje": len(doline),
        "gibanje_1d": gibanje_gladko,
        "sredina": sredina
    }