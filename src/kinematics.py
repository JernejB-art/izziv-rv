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

    vrhovi        -> array indeksov vrhov (pobiranje)
    doline        -> array indeksov dolin (vstavljanje)
    zacni_z_vrhom -> True = začni s pobiranjem
    vrne          -> (filtrirani_vrhovi, filtrirane_doline)
    """
    if len(vrhovi) == 0 or len(doline) == 0:
        return vrhovi, doline

    vsi = [(idx, 1) for idx in vrhovi] + [(idx, -1) for idx in doline]
    vsi = sorted(vsi, key=lambda x: x[0])

    ohranjeni = []
    zadnji_tip = -1 if zacni_z_vrhom else 1

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

def zaznava_zaticev(pozicije_kazalec, fps):
    """
    Prešteje število prenesenih zatičev iz trajektorije kazalca.

    Algoritem po Balaceanu et al. (2024):
    - x-koordinata kazalca se giblje med posodico in luknjicami
    - Sredina delovnega prostora = midpoint med max in min x
    - Vrhovi na strani posodice = pobiranje zatičev
    - Doline na strani luknjic = vstavljanje zatičev

    pozicije_kazalec -> seznam (x, y) koordinat konice kazalca
    fps              -> hitrost videa v sličicah/sekundo
    vrne             -> slovar z rezultati štetja
    """
    if len(pozicije_kazalec) < fps * 2:
        return {'pobiranje': 0, 'vstavljanje': 0, 'stevilo_zaticev': 0}

    tocke = np.array(pozicije_kazalec)
    x = tocke[:, 0].astype(float)

    # Zgladimo x signal pred iskanjem ekstremov
    x = glajenje_signal(x, fps)

    # Sredina delovnega prostora (meja med posodico in luknjicami)
    sredina = (x.max() + x.min()) / 2

    # Minimalna razdalja med dvema eventoma (~0.5 sekunde)
    min_razdalja = int(fps * 0.5)

    # Prominence = vsaj 5% celotnega razpona gibanja
    razpon = x.max() - x.min()
    min_prominence = razpon * 0.05

    # Vrhovi = pobiranje iz posodice, doline = vstavljanje v luknjice
    vrhovi, _ = find_peaks( x, distance=min_razdalja, prominence=min_prominence)
    doline, _ = find_peaks(-x, distance=min_razdalja, prominence=min_prominence)

    # Filtriraj glede na cono (katera stran sredine)
    pobiranje   = np.array([v for v in vrhovi if x[v] > sredina])
    vstavljanje = np.array([d for d in doline if x[d] < sredina])

    # Zagotovi pravilno izmenjevanje (pobiranje → vstavljanje → pobiranje...)
    pobiranje, vstavljanje = izmenjuj_ekstreme(pobiranje, vstavljanje, zacni_z_vrhom=True)

    return {
        'pobiranje':       len(pobiranje),
        'vstavljanje':     len(vstavljanje),
        'stevilo_zaticev': min(len(pobiranje), len(vstavljanje)),
        'vrhovi_idx':      pobiranje,
        'doline_idx':      vstavljanje,
        'x_kazalec':       x,
        'sredina':         sredina,
    }

def izracun_casov_zaticev(rezultat_zaticev, kin, fps):
    """
    Določi točen čas prijema/odlaganja iz minimuma hitrosti.
    
    Po Balaceanu et al. (2024): moment prijema/odlaganja = ko hitrost
    pade pod 5% maksimalne hitrosti in ostane tam vsaj 40ms.
    
    Iščemo minimum hitrosti PRED vsakim vrhom/dolino trajektorije
    — to je moment ko se roka ustavi ob prijemu/odlaganju.
    """
    dt = 1.0 / fps
    vrhovi = rezultat_zaticev['vrhovi_idx']
    doline = rezultat_zaticev['doline_idx']
    hitrost = kin['hitrost']

    # Okno pred vrhom/dolino kjer iščemo minimum hitrosti (~1s)
    okno_pred = int(fps * 1.0)
    okno_po   = int(fps * 0.3)

    casi_pobiranje = []
    casi_vstavljanje = []

    for i in range(min(len(vrhovi), len(doline))):
        v = vrhovi[i]
        d = doline[i]

        # Pobiranje: minimum hitrosti strogo PRED vrhom
        zac_p = max(0, v - okno_pred)
        lokalni_min_p = zac_p + np.argmin(hitrost[zac_p:v])
        casi_pobiranje.append(lokalni_min_p * dt)

        # Vstavljanje: minimum hitrosti strogo MED vrhom in dolino
        zac_v = v
        kon_v = min(len(hitrost), d + okno_po)
        lokalni_min_v = zac_v + np.argmin(hitrost[zac_v:kon_v])
        casi_vstavljanje.append(lokalni_min_v * dt)

    casi_postavljanja = np.array([
        max(0.0, casi_vstavljanje[i] - casi_pobiranje[i])
        for i in range(min(len(casi_pobiranje), len(casi_vstavljanje)))
    ])

    skupni_cas = casi_vstavljanje[-1] - casi_pobiranje[0] if (len(casi_pobiranje) > 0 and len(casi_vstavljanje) > 0) else 0

    return {
        'casi_pobiranje':    casi_pobiranje,
        'casi_vstavljanje':  casi_vstavljanje,
        'casi_postavljanja': casi_postavljanja,
        'skupni_cas':        skupni_cas,
    }

def zaznava_prehodov(pozicije_kazalec, posodica, fps):
    cx_p, cy_p, r_p = posodica
    dt = 1.0 / fps
    tocke = np.array(pozicije_kazalec)

    razdalje = np.sqrt(
        (tocke[:, 0] - cx_p)**2 +
        (tocke[:, 1] - cy_p)**2
    )

    # Hysteresis: vstop pri manjšem radiju, izstop pri večjem
    # Prepreči osciliranje ko je kazalec na robu
    r_vstop  = r_p * 0.85  # vstopi ko je res znotraj
    r_izstop = r_p * 1.15  # izstopi šele ko je res zunaj

    dogodki = []
    znotraj = False  # začnemo zunaj

    for i in range(len(razdalje)):
        if not znotraj and razdalje[i] < r_vstop:
            znotraj = True
            dogodki.append({'frame': i, 'cas': i * dt, 'tip': 'vstop_posodica'})
        elif znotraj and razdalje[i] > r_izstop:
            znotraj = False
            dogodki.append({'frame': i, 'cas': i * dt, 'tip': 'izstop_posodica'})

    # Štetje ciklov z minimalnim časom obiska
    MIN_CAS_OBISKA = 0.25
    casi_pobiranje = []
    casi_transport = []
    stevilo_zaticev = 0

    i = 0
    while i < len(dogodki) - 1:
        if dogodki[i]['tip'] == 'vstop_posodica' and dogodki[i+1]['tip'] == 'izstop_posodica':
            t_pobiranje = dogodki[i+1]['cas'] - dogodki[i]['cas']
            if t_pobiranje >= MIN_CAS_OBISKA:
                casi_pobiranje.append(t_pobiranje)
                stevilo_zaticev += 1
                if i + 2 < len(dogodki):
                    t_transport = dogodki[i+2]['cas'] - dogodki[i+1]['cas']
                    casi_transport.append(t_transport)
            i += 2
        else:
            i += 1

    return {
        'dogodki':         dogodki,
        'casi_pobiranje':  np.array(casi_pobiranje),
        'casi_transport':  np.array(casi_transport),
        'stevilo_zaticev': stevilo_zaticev,
        'razdalje':        razdalje,
    }

def filtriraj_skoke_kazalec(pozicije_kazalec, fps, max_hitrost_px=800):
    """
    Odstrani frame-e kjer kazalec naredi nerealen skok.
    max_hitrost_px -> maksimalna fizična hitrost kazalca v px/s
    """
    tocke = np.array(pozicije_kazalec, dtype=float)
    dt = 1.0 / fps
    max_skok = 150  # max skok v pikslih med dvema framoma

    ociscene = [tocke[0]]
    for i in range(1, len(tocke)):
        skok = np.linalg.norm(tocke[i] - ociscene[-1])
        if skok < max_skok:
            ociscene.append(tocke[i])
        else:
            # Zadrži prejšnjo pozicijo namesto napačne
            ociscene.append(ociscene[-1])

    return np.array(ociscene)