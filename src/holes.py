# luknjice_led.py
# Zaznava osvetljenih luknjic za merjenje časa in štetje zatičev v 9HPT testu.
#
# Algoritem:
#   1. Definiraj dve ROI območji (zgornje in spodnje 3x3 luknjice) glede na kamero
#   2. V vsakem frame-u preštej svetle luknjice v vsaki ROI
#   3. Stroj stanj:
#       IDLE   -> obe področji ugasnjeni (začetek)
#       BLINK  -> obe se hkrati prižgeta (semafor signal)
#       ACTIVE -> eno področje sveti (kam vstavljamo), drugo ugasnjeno
#                 -> timer teče, luknjice v aktivnem področju se ugašajo ko vstavljamo zatiče
#       DONE   -> drugo področje se prižge (test končan)
#   4. Čas testa = DONE.začetek - ACTIVE.začetek
#   5. Število zatičev = luknjice, ki so med ACTIVE prešle iz SVETLO -> TEMNO

import cv2
import numpy as np
import re
from scipy.ndimage import uniform_filter1d

ROI_PARAMETRI = {
    'camP_0': { 'zgornje': (0.51, 0.20, 0.68, 0.40),
                'spodnje': (0.46, 0.66, 0.60, 0.80), },
    'camP_1': { 'zgornje': (0.49, 0.06, 0.65, 0.28),
                'spodnje': (0.50, 0.63, 0.66, 0.85), },
    'camP_2': { 'zgornje': (0.35, 0.15, 0.50, 0.30),
                'spodnje': (0.42, 0.55, 0.59, 0.75), },
}

PRAG_SVETLOST = 190
PRAG_STEVILO_LUKNJIC = 5
MIN_CAS_STABILNOSTI = 0.3


def doloci_kamero(ime_datoteke):
    match = re.search(r'(camP_\d+)', ime_datoteke)
    return match.group(1) if match else 'camP_0'


def izrezi_roi(frame, roi_rel):
    visina, sirina = frame.shape[:2]
    x1 = int(sirina * roi_rel[0])
    y1 = int(visina * roi_rel[1])
    x2 = int(sirina * roi_rel[2])
    y2 = int(visina * roi_rel[3])
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def izboljsaj_kontrast_roi(roi_bgr):
    siva = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    siva = clahe.apply(siva)
    gama = 1.4
    tabela = np.array([255 * (i / 255) ** (1.0 / gama) for i in range(256)], dtype=np.uint8)
    siva = cv2.LUT(siva, tabela)
    return siva


def izracun_dinamicnega_praga(siva):
    """
    Dinamično določi prag za zaznavo svetlih luknjic glede na porazdelitev
    svetlosti v ROI območju.

    Algoritem:
    - Vzame percentil 85% kot oceno "svetle" vrednosti ozadja/plošče
    - Prag = ozadje + k * (max - ozadje)
    - V svetlem okolju bo prag višji, v temnem nižji
    - Zagotovi da prag nikoli ne pade pod absolutni minimum (150)
      ali preseže absolutni maksimum (240)

    siva -> sivinska slika ROI
    vrne -> integer prag (0-255)
    """
    p85  = int(np.percentile(siva, 85))   # tipična svetlost ozadja plošče
    pmax = int(np.percentile(siva, 99))   # svetlost najsvetlejših točk (luknjice)

    razpon = pmax - p85
    if razpon < 20:
        # Premalo kontrasta — privzeti absolutni prag
        return PRAG_SVETLOST

    # Prag = 70% poti med ozadjem in maksimumom
    prag = p85 + int(0.70 * razpon)

    # Clip na razumne meje
    return max(150, min(240, prag))


def stej_svetle_luknjice(roi_bgr, roi_referenca=None):
    """
    Zaznava svetlih okroglih luknjic v ROI območju.

    Samo konturna metoda — lokalni maksimumi so bili preagresivni
    in zaznavali odseve, robove plošče in zatiče.

    Postopek:
    1. CLAHE + gama za kontrast
    2. Dinamični prag (glede na svetlost ROI)
    3. Morfologija za čiščenje šuma
    4. Konture s filtrom površine in okroglosti
    5. Če zaznamo več kot 9 — obdržimo samo 9 najsvetlejših
       (v 3x3 mreži je fizično max 9 luknjic)

    roi_bgr -> BGR slika ROI območja
    vrne -> (stevilo, seznam_centrov)
    """
    if roi_bgr.size == 0:
        return 0, []

    siva = izboljsaj_kontrast_roi(roi_bgr)

    if roi_referenca is not None and roi_referenca.shape == roi_bgr.shape:
        # Primerjava z referenčnim frame-om (posnet med BLINK ko so vse luknjice prižgane).
        # Gledamo samo točke ki so SVETLEJŠE kot referenca — to so prazne luknjice.
        # Luknjice z zatičem so temnejše ali enake referenci → jih ne zaznamo.
        # Odporno na globalno spremembo osvetlitve ker gledamo relativno razliko.
        siva_ref = izboljsaj_kontrast_roi(roi_referenca)
        # signed difference: pozitivno = trenutni frame svetlejši od reference
        razlika = cv2.subtract(siva, siva_ref)  # satura pri 0, ne gre v negativno
        prag = izracun_dinamicnega_praga(razlika)
        prag = max(15, min(prag, 60))  # razlika mora biti vsaj 15 sivinskih enot
        _, maska = cv2.threshold(razlika, prag, 255, cv2.THRESH_BINARY)
    else:
        prag = izracun_dinamicnega_praga(siva)
        _, maska = cv2.threshold(siva, prag, 255, cv2.THRESH_BINARY)
    jedro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, jedro)
    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, jedro)

    konture, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kandidati = []
    for k in konture:
        povrsina = cv2.contourArea(k)
        if not (8 < povrsina < 2500):
            continue
        obseg = cv2.arcLength(k, True)
        if obseg == 0:
            continue
        okroglost = 4 * np.pi * povrsina / (obseg ** 2)
        if okroglost < 0.35:
            continue
        M = cv2.moments(k)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Povprečna svetlost tega območja — za razvrščanje
            maska_k = np.zeros(siva.shape, dtype=np.uint8)
            cv2.drawContours(maska_k, [k], -1, 255, -1)
            svetlost = cv2.mean(siva, mask=maska_k)[0]
            kandidati.append((svetlost, cx, cy))

    # Če je zaznanih več kot 9, obdrži samo 9 najsvetlejših
    # (fizično je v mreži max 9 luknjic)
    kandidati.sort(key=lambda x: x[0], reverse=True)
    kandidati = kandidati[:9]

    centri = [(cx, cy) for (_, cx, cy) in kandidati]
    return len(centri), centri

def je_obmocje_prizgano(stevilo_luknjic):
    return stevilo_luknjic >= PRAG_STEVILO_LUKNJIC


class StrojStanj9HPT:
    def __init__(self, fps):
        self.fps = fps
        self.stanje = 'IDLE'
        self.cas_zacetka = None
        self.cas_konca = None
        self.aktivno_obmocje = None
        self.stevilo_zatičev = 0
        self.max_luknjic_aktivno = 0
        self.min_luknjic_aktivno = 9
        self.dogodki = []
        self._stabilnost_counter = 0
        self._zadnje_stanje_zg = None
        self._zadnje_stanje_sp = None
        self._cas_zadnje_spremembe = 0

    def posodobi(self, frame_idx, st_zgornje, st_spodnje):
        """
        Zaporedje signalov:
          IDLE  -> obe ugasnjeni (začetek)
          BLINK -> obe se hkrati prižgeta (semafor)
          ACTIVE-> ena ugasne, druga ostane = ta je aktivna, test teče
          DONE  -> druga (neaktivna) se prižge = test končan
        """
        dt = 1.0 / self.fps
        cas = frame_idx * dt
        zg_on = je_obmocje_prizgano(st_zgornje)
        sp_on = je_obmocje_prizgano(st_spodnje)

        if self.stanje == 'IDLE':
            # Čakamo da sta obe prižgani (>=7 luknjic) vsaj 3 frame-e — to je začetno stanje
            # Tu shranimo referenčni frame za primerjavo (vse luknjice prižgane)
            pravi_on = (zg_on and sp_on and st_zgornje >= 7 and st_spodnje >= 7)
            if pravi_on:
                self._idle_counter = getattr(self, '_idle_counter', 0) + 1
            else:
                self._idle_counter = 0

            if self._idle_counter >= 3:
                self.stanje = 'PRED_BLINK'
                self._zabeleži_dogodek(frame_idx, cas, 'PRED_BLINK')

        elif self.stanje == 'PRED_BLINK':
            # Obe prižgani — čakamo da obe ugasneta (OFF faza blinka)
            # Toleriramo da ena ugasne malce prej kot druga
            if not zg_on and not sp_on:
                self.stanje = 'BLINK_OFF'
                self._zabeleži_dogodek(frame_idx, cas, 'BLINK_OFF')

        elif self.stanje == 'BLINK_OFF':
            # Obe ugasnjeni — čakamo ponovni prižig obeh (ON faza blinka)
            if zg_on and sp_on and st_zgornje >= 7 and st_spodnje >= 7:
                self.stanje = 'BLINK_ON'
                self._zabeleži_dogodek(frame_idx, cas, 'BLINK_ON')

        elif self.stanje == 'BLINK_ON':
            # Obe prižgani po OFF — zdaj čakamo da ena ugasne = ACTIVE
            if zg_on and not sp_on:
                self.stanje = 'ACTIVE'
                self.aktivno_obmocje = 'zgornje'
                self.cas_zacetka = frame_idx
                self.max_luknjic_aktivno = st_zgornje
                self._zabeleži_dogodek(frame_idx, cas, 'TEST_ZACETEK_ZG')
            elif sp_on and not zg_on:
                self.stanje = 'ACTIVE'
                self.aktivno_obmocje = 'spodnje'
                self.cas_zacetka = frame_idx
                self.max_luknjic_aktivno = st_spodnje
                self._zabeleži_dogodek(frame_idx, cas, 'TEST_ZACETEK_SP')
            elif not zg_on and not sp_on:
                # Obe ugasnili — morda še en blink cikel, nazaj v BLINK_OFF
                self.stanje = 'BLINK_OFF'
                self._zabeleži_dogodek(frame_idx, cas, 'BLINK_OFF')

        elif self.stanje == 'ACTIVE':
            if self.aktivno_obmocje == 'zgornje':
                trenutno = st_zgornje
                st_drugo = st_spodnje
            else:
                trenutno = st_spodnje
                st_drugo = st_zgornje

            self.min_luknjic_aktivno = min(self.min_luknjic_aktivno, trenutno)

            # Konec testa: neaktivno področje mora imeti >=7 luknjic
            # (višji prag kot PRAG_STEVILO_LUKNJIC=5) in to vsaj 4 frame-e zapored
            # — prepreči lažni konec iz kratkotrajnega odseva ali premika roke
            pravi_konec = st_drugo >= 7
            if pravi_konec:
                self._konec_counter = getattr(self, '_konec_counter', 0) + 1
            else:
                self._konec_counter = 0

            if self._konec_counter >= 4:
                self.stanje = 'DONE'
                self.cas_konca = frame_idx
                self.stevilo_zatičev = max(0, self.max_luknjic_aktivno - self.min_luknjic_aktivno)
                self._zabeleži_dogodek(frame_idx, cas, 'TEST_KONEC')

        elif self.stanje == 'DONE':
            pass

        return self.stanje

    def _zabeleži_dogodek(self, frame_idx, cas, tip):
        self.dogodki.append({'frame': frame_idx, 'cas': cas, 'tip': tip})
        print(f"  [HPT] Frame {frame_idx:4d} | t={cas:6.2f}s | {tip}")

    @property
    def cas_testa_sekunde(self):
        if self.cas_zacetka is None or self.cas_konca is None:
            return None
        return (self.cas_konca - self.cas_zacetka) / self.fps

    @property
    def roka(self):
        if self.aktivno_obmocje == 'zgornje':
            return 'leva'
        elif self.aktivno_obmocje == 'spodnje':
            return 'desna'
        return 'neznana'


def preveri_veljavnost_videa(pot_videa, n_vzorcev=20):
    """
    Hitro preveri ali video vsebuje celoten potek testa (blink + aktiven + konec).
    Vzorči N frame-ov iz začetka in konca videa — ne procesira celotnega videa.

    Možne napake:
      KRATEK_VIDEO     -> video je prekratek za smiseln test (<10s)
      NI_BLINK_ZACETEK -> na začetku videa ni detektiranega blinka (test že tekel)
      NI_KONEC         -> na koncu videa ni detektiranega konca (video se predčasno konča)
      SAMO_ENO_POLJE   -> skozi cel video sveti samo eno področje (posnet sredi testa)

    pot_videa  -> pot do videa
    n_vzorcev  -> število frame-ov za vzorčenje iz začetka/konca
    vrne -> (veljavno: bool, razlog: str)
    """
    import os
    ime_datoteke = os.path.basename(pot_videa)
    kamera = doloci_kamero(ime_datoteke)
    roi_param = ROI_PARAMETRI.get(kamera, ROI_PARAMETRI['camP_0'])

    cap = cv2.VideoCapture(pot_videa)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skupaj_framov = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trajanje_s = skupaj_framov / fps if fps > 0 else 0

    # Prekratek video
    if trajanje_s < 10:
        cap.release()
        return False, f"KRATEK_VIDEO: trajanje {trajanje_s:.1f}s < 10s"

    # Vzorči začetek (prvih n_vzorcev frame-ov)
    st_zg_zacetek, st_sp_zacetek = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(n_vzorcev):
        ret, frame = cap.read()
        if not ret:
            break
        roi_zg, _ = izrezi_roi(frame, roi_param['zgornje'])
        roi_sp, _ = izrezi_roi(frame, roi_param['spodnje'])
        n_zg, _ = stej_svetle_luknjice(roi_zg)
        n_sp, _ = stej_svetle_luknjice(roi_sp)
        st_zg_zacetek.append(n_zg)
        st_sp_zacetek.append(n_sp)

    # Vzorči konec (zadnjih n_vzorcev frame-ov)
    st_zg_konec, st_sp_konec = [], []
    zacetek_konca = max(0, skupaj_framov - n_vzorcev)
    cap.set(cv2.CAP_PROP_POS_FRAMES, zacetek_konca)
    for _ in range(n_vzorcev):
        ret, frame = cap.read()
        if not ret:
            break
        roi_zg, _ = izrezi_roi(frame, roi_param['zgornje'])
        roi_sp, _ = izrezi_roi(frame, roi_param['spodnje'])
        n_zg, _ = stej_svetle_luknjice(roi_zg)
        n_sp, _ = stej_svetle_luknjice(roi_sp)
        st_zg_konec.append(n_zg)
        st_sp_konec.append(n_sp)

    cap.release()

    if not st_zg_zacetek or not st_zg_konec:
        return False, "NAPAKA_BRANJA: ne morem prebrati frame-ov"

    zg_zac = np.median(st_zg_zacetek)
    sp_zac = np.median(st_sp_zacetek)
    zg_kon = np.median(st_zg_konec)
    sp_kon = np.median(st_sp_konec)

    obe_zac = zg_zac >= 7 and sp_zac >= 7
    obe_kon = zg_kon >= 7 and sp_kon >= 7
    samo_ena_zac = (zg_zac >= 5) != (sp_zac >= 5)  # XOR — samo ena sveti

    # Na začetku sveti samo eno področje → test je že v teku
    if samo_ena_zac:
        aktivno = 'zgornje' if zg_zac >= 5 else 'spodnje'
        return False, f"NI_BLINK_ZACETEK: na začetku sveti samo {aktivno} področje — video posnet sredi testa"

    # Na začetku obe ugasnjeni in na koncu obe ugasnjeni → verjetno samo en del
    if zg_zac < 5 and sp_zac < 5 and zg_kon < 5 and sp_kon < 5:
        return False, "SAMO_TEMA: na začetku in koncu obe področji ugasnjeni — nepopoln posnetek"

    # Na koncu mora svetiti samo eno področje ALI obe (znak zaključka)
    # Neveljavno: samo eno področje sveti na koncu = test še v teku
    samo_ena_kon = (zg_kon >= 5) != (sp_kon >= 5)
    if samo_ena_kon:
        aktivno = 'zgornje' if zg_kon >= 5 else 'spodnje'
        return False, f"NI_KONEC: na koncu videa sveti samo {aktivno} področje — video se konča pred zaključkom testa"

    # Na koncu obe ugasnjeni in na začetku obe prižgani → test je v teku brez konca
    if zg_kon < 5 and sp_kon < 5 and obe_zac:
        return False, "NI_KONEC: na koncu obe področji ugasnjeni — test se ni zaključil"

    # Na koncu sveti samo eno področje z visoko gotovostjo (>=7) = sredi testa
    if (zg_kon >= 7 and sp_kon < 3) or (sp_kon >= 7 and zg_kon < 3):
        aktivno = 'zgornje' if zg_kon >= 7 else 'spodnje'
        return False, f"NI_KONEC: na koncu aktivno samo {aktivno} področje — pacient še ni pobral zatičev"

    return True, "OK"


def analiziraj_led_luknjice(pot_videa, izhod_video=None, izhod_graf=None):
    import os
    import matplotlib.pyplot as plt

    ime_datoteke = os.path.basename(pot_videa)
    kamera = doloci_kamero(ime_datoteke)
    roi_param = ROI_PARAMETRI.get(kamera, ROI_PARAMETRI['camP_0'])

    print(f"Kamera: {kamera}")
    print(f"ROI zgornje: {roi_param['zgornje']}")
    print(f"ROI spodnje: {roi_param['spodnje']}")

    cap = cv2.VideoCapture(pot_videa)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sirina = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    visina = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if izhod_video:
        writer = cv2.VideoWriter(izhod_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (sirina, visina))

    stroj = StrojStanj9HPT(fps)
    stevila_zg = []
    stevila_sp = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_zg_slika, koord_zg = izrezi_roi(frame, roi_param['zgornje'])
        roi_sp_slika, koord_sp = izrezi_roi(frame, roi_param['spodnje'])

        n_zg, centri_zg = stej_svetle_luknjice(roi_zg_slika)
        n_sp, centri_sp = stej_svetle_luknjice(roi_sp_slika)

        stevila_zg.append(n_zg)
        stevila_sp.append(n_sp)

        stanje = stroj.posodobi(frame_idx, n_zg, n_sp)

        if writer is not None:
            frame = _narisi_debug(
                frame, koord_zg, koord_sp,
                n_zg, n_sp, centri_zg, centri_sp,
                stanje, stroj, frame_idx, fps
            )
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    stevila_zg = np.array(stevila_zg, dtype=float)
    stevila_sp = np.array(stevila_sp, dtype=float)
    zg_glajeno = uniform_filter1d(stevila_zg, size=15)
    sp_glajeno = uniform_filter1d(stevila_sp, size=15)

    if izhod_graf:
        _narisi_graf(zg_glajeno, sp_glajeno, stroj, fps, izhod_graf)

    rezultati = {
        'cas_testa':       stroj.cas_testa_sekunde,
        'stevilo_zaticov': stroj.stevilo_zatičev,
        'roka':            stroj.roka,
        'aktivno_obmocje': stroj.aktivno_obmocje,
        'cas_zacetka_s':   stroj.cas_zacetka / fps if stroj.cas_zacetka else None,
        'cas_konca_s':     stroj.cas_konca / fps if stroj.cas_konca else None,
        'dogodki':         stroj.dogodki,
        'stevila_zg':      stevila_zg,
        'stevila_sp':      stevila_sp,
    }

    _izpisi_rezultate(rezultati)
    return rezultati


def _narisi_debug(frame, koord_zg, koord_sp, n_zg, n_sp,
                   centri_zg, centri_sp, stanje, stroj, frame_idx, fps):
    x1z, y1z, x2z, y2z = koord_zg
    x1s, y1s, x2s, y2s = koord_sp

    barva_zg = (0, 255, 0) if je_obmocje_prizgano(n_zg) else (0, 0, 200)
    barva_sp = (0, 255, 0) if je_obmocje_prizgano(n_sp) else (0, 0, 200)

    if stanje == 'ACTIVE':
        if stroj.aktivno_obmocje == 'zgornje':
            barva_zg = (0, 200, 255)
        else:
            barva_sp = (0, 200, 255)

    cv2.rectangle(frame, (x1z, y1z), (x2z, y2z), barva_zg, 2)
    cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), barva_sp, 2)

    for (cx, cy) in centri_zg:
        cv2.circle(frame, (x1z + cx, y1z + cy), 5, (0, 255, 255), -1)
    for (cx, cy) in centri_sp:
        cv2.circle(frame, (x1s + cx, y1s + cy), 5, (0, 255, 255), -1)

    cas = frame_idx / fps
    cv2.putText(frame, f"Stanje: {stanje}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Zg: {n_zg}  Sp: {n_sp}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if stanje == 'ACTIVE' and stroj.cas_zacetka is not None:
        t_elapsed = cas - stroj.cas_zacetka / fps
        cv2.putText(frame, f"Cas: {t_elapsed:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if stanje == 'DONE' and stroj.cas_testa_sekunde is not None:
        cv2.putText(frame, f"KONEC: {stroj.cas_testa_sekunde:.2f}s | Zatiči: {stroj.stevilo_zatičev}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

    return frame


def _narisi_graf(zg_glajeno, sp_glajeno, stroj, fps, izhod_graf):
    import matplotlib.pyplot as plt

    cas_os = np.arange(len(zg_glajeno)) / fps

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cas_os, zg_glajeno, color='steelblue', label='Zgornje področje')
    ax.plot(cas_os, sp_glajeno, color='darkorange', label='Spodnje področje')
    ax.axhline(PRAG_STEVILO_LUKNJIC, color='gray', linestyle='--',
               label=f'Prag = {PRAG_STEVILO_LUKNJIC}')

    barve_dogodkov = {
        'PRED_BLINK':      'gray',
        'BLINK_OFF':       'orange',
        'BLINK_ON':        'purple',
        'TEST_ZACETEK_ZG': 'green',
        'TEST_ZACETEK_SP': 'green',
        'TEST_KONEC':      'red',
    }
    for d in stroj.dogodki:
        barva = barve_dogodkov.get(d['tip'], 'black')
        ax.axvline(d['cas'], color=barva, linestyle=':', alpha=0.8)
        ax.text(d['cas'] + 0.1, ax.get_ylim()[1] * 0.9,
                d['tip'], rotation=90, fontsize=7, color=barva)

    if stroj.cas_testa_sekunde is not None:
        ax.set_title(
            f"LED luknjice — Čas testa: {stroj.cas_testa_sekunde:.2f}s | "
            f"Zatičev: {stroj.stevilo_zatičev} | Roka: {stroj.roka}"
        )
    else:
        ax.set_title("LED luknjice — test ni bil zaznán")

    ax.set_xlabel("Čas [s]")
    ax.set_ylabel("Število svetlih luknjic")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(izhod_graf)
    plt.close()
    print(f"Graf shranjen: {izhod_graf}")


def _izpisi_rezultate(r):
    print("\n=== REZULTATI LED ZAZNAVE ===")
    if r['cas_testa'] is not None:
        print(f"Čas testa:        {r['cas_testa']:.2f} s")
        print(f"Začetek:          {r['cas_zacetka_s']:.2f} s")
        print(f"Konec:            {r['cas_konca_s']:.2f} s")
        print(f"Število zatičev:  {r['stevilo_zaticov']}")
        print(f"Roka:             {r['roka']}")
        print(f"Aktivno področje: {r['aktivno_obmocje']}")
    else:
        print("Test ni bil zaznán — preverite ROI parametre!")
    print("=============================\n")


def kalibriraj_roi(pot_videa, frame_n=0, izhod_slika='/workspace/results/roi_kalibracija.png'):
    """
    Shrani frame z vsemi obstoječimi ROI območji narisan za vizualni pregled.
    Uporabi za prilagoditev ROI_PARAMETRI za posamezno kamero.

    pot_videa  -> pot do videa
    frame_n    -> kateri frame prikaži (0 = prvi)
    """
    import os
    cap = cv2.VideoCapture(pot_videa)
    for _ in range(frame_n + 1):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Napaka pri branju frame-a!")
        return

    kamera = doloci_kamero(os.path.basename(pot_videa))
    roi_param = ROI_PARAMETRI.get(kamera, ROI_PARAMETRI['camP_0'])
    visina, sirina = frame.shape[:2]

    for ime, roi in roi_param.items():
        x1 = int(sirina * roi[0]); y1 = int(visina * roi[1])
        x2 = int(sirina * roi[2]); y2 = int(visina * roi[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Napis NAD okvirjem — ne zakriva luknjic znotraj
        napis_y = y1 - 8 if y1 > 25 else y2 + 20
        cv2.putText(frame, ime, (x1 + 5, napis_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Pokaži zaznane luknjice
        roi_slika = frame[y1:y2, x1:x2].copy()
        n, centri = stej_svetle_luknjice(roi_slika)
        for (cx, cy) in centri:
            cv2.circle(frame, (x1 + cx, y1 + cy), 6, (0, 255, 0), 2)

        print(f"ROI '{ime}': {n} svetlih luknjic")

    cv2.imwrite(izhod_slika, frame)
    print(f"Kalibracija shranjena: {izhod_slika}")
    return frame


def integriraj_v_detect(pot_videa, fps, pozicije_kazalec=None):
    """
    Poenostavljen vmesnik za integracijo z detect.py.
    Najprej preveri veljavnost posnetka — če je nepopoln, vrne None.

    Primer uporabe v detect.py:
        from luknjice_led import integriraj_v_detect
        led_rezultat = integriraj_v_detect(vhod, fps)
        if led_rezultat is None:
            print("Neveljavni posnetek — preskoči")
        else:
            cas_testa_led = led_rezultat['cas_testa']
    """
    veljavno, razlog = preveri_veljavnost_videa(pot_videa)
    if not veljavno:
        print(f"[LED] NEVELJAVNI POSNETEK: {razlog}")
        return None
    return analiziraj_led_luknjice(
        pot_videa,
        izhod_video='/workspace/results/led_debug.mp4',
        izhod_graf='/workspace/results/led_luknjice.png'
    )


if __name__ == "__main__":
    import sys

    pot = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/Data/patient_024/patient_024camP_0_20230511_14_11_19.mp4"

    print("=== KALIBRACIJA ROI ===")
    kalibriraj_roi(pot, frame_n=10)

    print("\n=== PREVERJANJE VELJAVNOSTI ===")
    veljavno, razlog = preveri_veljavnost_videa(pot)
    if not veljavno:
        print(f"NEVELJAVNI POSNETEK: {razlog}")
        print("Analiza preskočena.")
        exit(0)
    print(f"Posnetek OK: {razlog}")

    print("\n=== ANALIZA LED LUKNJIC ===")
    rezultat = analiziraj_led_luknjice(
        pot,
        izhod_video='/workspace/results/led_debug.mp4',
        izhod_graf='/workspace/results/led_luknjice.png'
    )