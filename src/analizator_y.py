# analizator_y.py
# Post-procesna analiza Y koordinate roke v mm prostoru za zaznavo ciklov.
#
# IDEJA:
#   Po koncu video procesiranja imamo shranjeno celotno trajektorijo (t, x_mm, y_mm).
#   Y os v homografiji mm prostoru je "os naloge" — roka niha med:
#     - nizek Y  = luknjice (vstavljanje)
#     - visok Y  = posodica (pobiranje)
#
#   Iščemo:
#     - DOLINE (lokalni minimumi Y) = vstavljanje v luknjico
#     - VRHOVI (lokalni maksimumi Y) = pobiranje iz posodice
#
#   Faza (vstavljanje/pospravljanje) določi smer:
#     VSTAVLJANJE:    vrh → dolina = en cikel
#     POSPRAVLJANJE:  dolina → vrh = en cikel
#
# Prednosti pred FSM:
#   - Ne potrebuje točnega centra posodice
#   - Deluje tudi ko MediaPipe izgubi roko (interpolacija)
#   - Robustno na perspektivne napake
#   - Enostavno za nastavitev (samo 2 parametra)

import numpy as np
from scipy.signal import savgol_filter, find_peaks


# ===== PARAMETRI =====

# Glajenje Y signala
SG_OKNO_Y = 15    # Savitzky-Golay okno (frame-ov)
SG_RED_Y  = 2

# find_peaks parametri
MIN_PROMINENCA_MM = 20.0   # min višina vrha/doline (mm) — filtrira šum
MIN_RAZDALJA_S    = 0.4    # min čas med zaporednima vrhovoma (s)

# Meja med posodico in luknjicami (v mm na Y osi)
# Vrednost med centrom posodice in centrom luknjic
# Privzeto: None = določi avtomatično iz podatkov (sredina med min in max Y)
Y_MEJA_MM = None


class AnalizatorYOsi:
    """
    Post-procesna analiza Y koordinate trajektorije za zaznavo ciklov.

    Uporaba:
        analizator = AnalizatorYOsi(fps=25.0)
        rezultati = analizator.analiziraj(
            casi, xs_mm, ys_mm,
            t_zacetek_pospravljanja=22.3   # iz LED stroja
        )
    """

    def __init__(self, fps=25.0,
                 min_prominenca_mm=MIN_PROMINENCA_MM,
                 min_razdalja_s=MIN_RAZDALJA_S):
        self.fps              = fps
        self.min_prominenca   = min_prominenca_mm
        self.min_razdalja_s   = min_razdalja_s

    def _zgladiti_y(self, ys):
        """Zgladi Y signal z Savitzky-Golay filtrom."""
        n = len(ys)
        if n < SG_OKNO_Y:
            return np.array(ys)
        okno = SG_OKNO_Y if SG_OKNO_Y % 2 == 1 else SG_OKNO_Y + 1
        try:
            return savgol_filter(ys, okno, SG_RED_Y)
        except Exception:
            return np.array(ys)

    def _najdi_ekstreme(self, casi, ys_gl):
        """
        Poišči lokalne minimume in maksimume v zglajenem Y signalu.
        Vrne (idx_vrhovi, idx_doline) kot numpy array indeksov.
        """
        min_razdalja_frame = max(3, int(self.min_razdalja_s * self.fps))

        # Vrhovi (visok Y = posodica)
        idx_vrhovi, _ = find_peaks(
            ys_gl,
            prominence=self.min_prominenca,
            distance=min_razdalja_frame
        )

        # Doline (nizek Y = luknjice)
        idx_doline, _ = find_peaks(
            -ys_gl,
            prominence=self.min_prominenca,
            distance=min_razdalja_frame
        )

        return idx_vrhovi, idx_doline

    def _doloci_y_mejo(self, ys_gl, t_preklop_idx):
        if t_preklop_idx is not None and t_preklop_idx > 10:
            y_vst = ys_gl[:t_preklop_idx]
            y_pos = ys_gl[t_preklop_idx:]
            # Luknjice = nižji Y v fazi vstavljanja
            # Posodica = višji Y v fazi pospravljanja  
            y_luknjice = np.percentile(y_vst, 20)   # spodnji 20% = luknjice
            y_posodica = np.percentile(y_pos, 80)    # zgornji 20% = posodica
            y_meja = (y_luknjice + y_posodica) / 2.0
        else:
            y_meja = (np.max(ys_gl) + np.min(ys_gl)) / 2.0
        return y_meja

    def analiziraj(self, casi, xs_mm, ys_mm,
                   t_zacetek=None,
                   t_preklop_pospravljanja=None,
                   t_konec=None,
                   verbose=True):
        """
        Glavna analiza. Poišče cikle vstavljanja in pospravljanja.

        casi                    → numpy array časov [s]
        xs_mm, ys_mm            → numpy array koordinat v mm
        t_zacetek               → čas začetka testa [s] (iz LED stroja)
        t_preklop_pospravljanja → čas preklopa faze [s] (iz LED stroja)
        t_konec                 → čas konca testa [s] (iz LED stroja)

        Vrne dict z:
            cicli_vstavljanje   → seznam dict-ov z pickup/insert časi
            cicli_pospravljanje → seznam dict-ov
            y_meja_mm           → izračunana meja Y
            ys_gl               → zglajen Y signal
            idx_vrhovi          → indeksi vrhov
            idx_doline          → indeksi dolin
        """
        casi   = np.array(casi)
        ys_mm  = np.array(ys_mm)
        xs_mm  = np.array(xs_mm)
        n      = len(casi)

        if n < 10:
            return self._prazen_rezultat()

        # 1. Zgladi Y
        ys_gl = self._zgladiti_y(ys_mm)

        # 2. Določi indeks preklopa faze
        t_preklop_idx = None
        if t_preklop_pospravljanja is not None:
            diffs = np.abs(casi - t_preklop_pospravljanja)
            t_preklop_idx = int(np.argmin(diffs))

        # 3. Določi Y mejo
        if hasattr(self, 'y_meja') and self.y_meja is not None:
            y_meja = self.y_meja
        elif Y_MEJA_MM is not None:
            y_meja = Y_MEJA_MM
        else:
            y_meja = self._doloci_y_mejo(ys_gl, t_preklop_idx)

        # 4. Poišči ekstreme
        idx_vrhovi, idx_doline = self._najdi_ekstreme(casi, ys_gl)

        if verbose:
            print(f"\n[AnalizatorY] Y meja: {y_meja:.1f}mm")
            print(f"[AnalizatorY] Vrhovi (posodica): {len(idx_vrhovi)}")
            print(f"[AnalizatorY] Doline (luknjice): {len(idx_doline)}")

        # 5. Razdelimo ekstreme po fazi
        #    VSTAVLJANJE: t < t_preklop → iščemo pare vrh→dolina
        #    POSPRAVLJANJE: t >= t_preklop → iščemo pare dolina→vrh

        cicli_v = self._sestavi_cikle_vstavljanje(
            casi, ys_gl, idx_vrhovi, idx_doline,
            t_od=t_zacetek,
            t_do=None,
            y_meja=y_meja,
            verbose=verbose
        )

        # Določi preklop iz zadnjega vstavljanja
        t_preklop_dejanski = cicli_v[-1]['insert_complete'] if cicli_v else t_preklop_pospravljanja

        cicli_p = self._sestavi_cikle_pospravljanje(
            casi, ys_gl, idx_vrhovi, idx_doline,
            t_od=t_preklop_dejanski+0.15,
            t_do=t_konec,
            y_meja=y_meja,
            verbose=verbose
        )

        if verbose:
            print(f"[AnalizatorY] Vstavljanje: {len(cicli_v)} ciklov")
            print(f"[AnalizatorY] Pospravljanje: {len(cicli_p)} ciklov")
        print(f"[DEBUG] t_preklop={t_preklop_pospravljanja}, t_preklop_idx={t_preklop_idx}")
        print(f"[DEBUG] y_vst mediana={np.median(ys_gl[:t_preklop_idx]):.1f}, y_pos mediana={np.median(ys_gl[t_preklop_idx:]):.1f}")
        return {
            'cicli_vstavljanje':   cicli_v,
            'cicli_pospravljanje': cicli_p,
            'y_meja_mm':           y_meja,
            'ys_gl':               ys_gl,
            'idx_vrhovi':          idx_vrhovi,
            'idx_doline':          idx_doline,
            'casi':                casi,
            'xs_mm':               xs_mm,
            'ys_mm':               ys_mm,
        }

    def _sestavi_cikle_vstavljanje(self, casi, ys_gl,
                                    idx_vrhovi, idx_doline,
                                    t_od, t_do, y_meja, verbose):
        """
        VSTAVLJANJE: cikel = vrh (posodica) → dolina (luknjica)
        Pickup start = vrh, insert complete = naslednja dolina
        """
        cicli = []

        # Filtriraj ekstreme na časovno okno
        def v_oknu(idx_arr):
            mask = np.ones(len(idx_arr), dtype=bool)
            if t_od is not None:
                mask &= casi[idx_arr] >= t_od
            if t_do is not None:
                mask &= casi[idx_arr] <= t_do
            return idx_arr[mask]

        vrhovi = v_oknu(idx_vrhovi)
        doline = v_oknu(idx_doline)

        # Pari: za vsak vrh poišči naslednjo dolino
        for v_idx in vrhovi:
            t_vrh = casi[v_idx]
            y_vrh = ys_gl[v_idx]

            # Mora biti nad mejo (posodica)
            if y_vrh < y_meja:
                continue

            # Poišči naslednjo dolino po tem vrhu
            naslednje_doline = doline[doline > v_idx]
            if len(naslednje_doline) == 0:
                continue

            d_idx = naslednje_doline[0]
            t_dolina = casi[d_idx]
            y_dolina = ys_gl[d_idx]

            # Dolina mora biti pod mejo (luknjice)
            if y_dolina > y_meja:
                continue

            movement_time = t_dolina - t_vrh

            # Minimalni movement time (prepreči lažne pare)
            if movement_time < 0.15:
                continue

            cicli.append({
                'pickup_start':    t_vrh,
                'pickup_complete': t_vrh,         # ni ločene pickup faze pri tej metodi
                'insert_start':    t_dolina - 0.1,
                'insert_complete': t_dolina,
                'movement_time':   movement_time,
                'pickup_duration': 0.0,
                'insert_duration': 0.1,
                'y_vrh':           float(y_vrh),
                'y_dolina':        float(y_dolina),
                'faza':            'VSTAVLJANJE',
                'metoda':          'Y-analiza',
            })

            if verbose:
                print(f"  [Y-V] t={t_vrh:.2f}s→{t_dolina:.2f}s "
                      f"(Y: {y_vrh:.0f}→{y_dolina:.0f}mm, "
                      f"premik={movement_time:.2f}s)")

            if len(cicli) >= 9:
                break
        return cicli

    def _sestavi_cikle_pospravljanje(self, casi, ys_gl,
                                      idx_vrhovi, idx_doline,
                                      t_od, t_do, y_meja, verbose):
        """
        POSPRAVLJANJE: cikel = dolina (luknjica) → vrh (posodica)
        Pickup start = dolina, insert complete = naslednji vrh
        """
        cicli = []

        def v_oknu(idx_arr):
            mask = np.ones(len(idx_arr), dtype=bool)
            if t_od is not None:
                mask &= casi[idx_arr] >= t_od
            if t_do is not None:
                mask &= casi[idx_arr] <= t_do
            return idx_arr[mask]

        vrhovi = v_oknu(idx_vrhovi)
        print(f"  [DEBUG] vrhovi v fazi P: {len(vrhovi)}, casi: {casi[vrhovi]}")
        doline = v_oknu(idx_doline)

        # Pari: za vsako dolino poišči naslednji vrh
        for v_idx in vrhovi:
            t_vrh = casi[v_idx]
            y_vrh = ys_gl[v_idx]
            
            if y_vrh < y_meja:
                continue
            
            # Poišči PREJŠNJO dolino pred tem vrhom
            prejsnje_doline = doline[doline < v_idx]
            if len(prejsnje_doline) == 0:
                continue
            
            d_idx = prejsnje_doline[-1]  # zadnja dolina pred vrhom
            t_dolina = casi[d_idx]
            y_dolina = ys_gl[d_idx]
            
            if y_dolina > y_meja:
                continue
            
            movement_time = t_vrh - t_dolina
            if movement_time < 0.15:
                continue
            print(f"  [DEBUG-P] vrh t={t_vrh:.2f} y={y_vrh:.0f} | "
                    f"dolina t={t_dolina:.2f} y={y_dolina:.0f} | "
                    f"mt={movement_time:.2f} | y_meja={y_meja:.0f}")

            cicli.append({
                'pickup_start':    t_dolina,
                'pickup_complete': t_dolina,
                'insert_start':    t_vrh - 0.1,
                'insert_complete': t_vrh,
                'movement_time':   movement_time,
                'pickup_duration': 0.0,
                'insert_duration': 0.1,
                'y_dolina':        float(y_dolina),
                'y_vrh':           float(y_vrh),
                'faza':            'POSPRAVLJANJE',
                'metoda':          'Y-analiza',
            })

            if verbose:
                print(f"  [Y-P] t={t_dolina:.2f}s→{t_vrh:.2f}s "
                      f"(Y: {y_dolina:.0f}→{y_vrh:.0f}mm, "
                      f"premik={movement_time:.2f}s)")
            if len(cicli) >= 9:
                break
        return cicli

    def _prazen_rezultat(self):
        return {
            'cicli_vstavljanje':   [],
            'cicli_pospravljanje': [],
            'y_meja_mm':           None,
            'ys_gl':               np.array([]),
            'idx_vrhovi':          np.array([]),
            'idx_doline':          np.array([]),
            'casi':                np.array([]),
            'xs_mm':               np.array([]),
            'ys_mm':               np.array([]),
        }

    def narisi_graf(self, rezultat, izhod_pot=None):
        """
        Nariše Y koordinato z označenimi ekstremi in cikli.
        """
        import matplotlib.pyplot as plt

        casi  = rezultat['casi']
        ys_mm = rezultat['ys_mm']
        ys_gl = rezultat['ys_gl']
        idx_v = rezultat['idx_vrhovi']
        idx_d = rezultat['idx_doline']
        y_m   = rezultat['y_meja_mm']

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # Zgornji: Y signal
        axs[0].plot(casi, ys_mm, color='lightsteelblue',
                    alpha=0.5, lw=0.8, label='Y surovi')
        axs[0].plot(casi, ys_gl, color='steelblue',
                    lw=1.5, label='Y zglajen')

        if y_m is not None:
            axs[0].axhline(y_m, color='gray', linestyle='--',
                           alpha=0.7, label=f'meja={y_m:.0f}mm')

        # Označi vrhove in doline
        if len(idx_v):
            axs[0].scatter(casi[idx_v], ys_gl[idx_v],
                           color='darkorange', s=80, zorder=5,
                           marker='^', label='vrh (posodica)')
        if len(idx_d):
            axs[0].scatter(casi[idx_d], ys_gl[idx_d],
                           color='steelblue', s=80, zorder=5,
                           marker='v', label='dolina (luknjice)')

        # Označi cikle vstavljanja
        for c in rezultat['cicli_vstavljanje']:
            axs[0].axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.08, color='green')
            axs[0].axvline(c['pickup_start'],
                           color='green', alpha=0.6, lw=1)
            axs[0].axvline(c['insert_complete'],
                           color='lime', alpha=0.6, lw=1)

        # Označi cikle pospravljanja
        for c in rezultat['cicli_pospravljanje']:
            axs[0].axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.08, color='tomato')
            axs[0].axvline(c['pickup_start'],
                           color='tomato', alpha=0.6, lw=1)
            axs[0].axvline(c['insert_complete'],
                           color='red', alpha=0.6, lw=1)

        axs[0].invert_yaxis()   # Y narašča navzdol (luknjice spodaj)
        axs[0].set_ylabel('Y [mm]')
        axs[0].set_title('Y koordinata roke — '
                         f"V:{len(rezultat['cicli_vstavljanje'])} "
                         f"P:{len(rezultat['cicli_pospravljanje'])}")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.3)

        # Spodnji: movement time po ciklih
        mt_v = [c['movement_time'] for c in rezultat['cicli_vstavljanje']]
        mt_p = [c['movement_time'] for c in rezultat['cicli_pospravljanje']]
        x_v  = np.arange(len(mt_v))
        x_p  = np.arange(len(mt_p))

        if mt_v:
            axs[1].bar(x_v - 0.2, mt_v, 0.35,
                       label='Vstavljanje', color='steelblue', alpha=0.8)
        if mt_p:
            axs[1].bar(x_p + 0.2, mt_p, 0.35,
                       label='Pospravljanje', color='tomato', alpha=0.8)
        axs[1].set_ylabel('Movement time [s]')
        axs[1].set_xlabel('Cikel #')
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        if izhod_pot:
            plt.savefig(izhod_pot, dpi=130)
            plt.close()
            print(f"[AnalizatorY] Graf shranjen: {izhod_pot}")
        else:
            plt.show()


# ===== INTEGRACIJA Z DETECT_COMBINED =====

def analiziraj_iz_rezultatov(rezultati_combined, verbose=True):
    """
    Zažene Y-analizo na rezultatih iz ProcesorCombined.

    rezultati_combined → dict iz ProcesorCombined.procesiraj()
    Vrne rezultate Y analize.
    """
    casi_hom = rezultati_combined.get('casi_hom', np.array([]))
    traj_mm  = rezultati_combined.get('traj_mm',  [])

    if len(casi_hom) == 0 or len(traj_mm) == 0:
        print("[AnalizatorY] Ni podatkov za analizo")
        return None

    xs_mm = np.array([p[0] for p in traj_mm])
    ys_mm = np.array([p[1] for p in traj_mm])

    # Pridobi čase iz LED stroja
    led = rezultati_combined.get('led_stroj')
    fps = 25.0  # privzeto

    t_zacetek = None
    t_preklop = None
    t_konec   = None

    if led:
        for d in led.dogodki:
            if d['tip'] in ('TEST_ZACETEK_ZG', 'TEST_ZACETEK_SP'):
                t_zacetek = d['cas']
            elif d['tip'] == 'ZACETEK_POSPRAVLJANJA':
                t_preklop = d['cas']
            elif d['tip'] == 'TEST_KONEC':
                t_konec = d['cas']

    if verbose:
        print(f"[AnalizatorY] t_zacetek={t_zacetek}, "
              f"t_preklop={t_preklop}, t_konec={t_konec}")

    analizator = AnalizatorYOsi(fps=fps)   # vedno ustvari

    # Določi Y mejo iz homografije (bolj zanesljivo kot iz podatkov)
    hom = rezultati_combined.get('homografija')
    if hom and hom.center_posodice_mm:
        y_posodica = hom.center_posodice_mm[1]
        y_luknjice = 32.0
        analizator.y_meja = (y_posodica + y_luknjice) / 2.0
        print(f"[AnalizatorY] Y meja iz homografije: {analizator.y_meja:.1f}mm")

    return analizator.analiziraj(
        casi_hom, xs_mm, ys_mm,
        t_zacetek=t_zacetek,
        t_preklop_pospravljanja=t_preklop,
        t_konec=t_konec,
        verbose=verbose
    )


# ===== TEST =====
if __name__ == "__main__":
    # Simuliraj sinusno trajektorijo
    fps  = 25.0
    casi = np.linspace(0, 35, int(35 * fps))
    # Y niha med 50 (luknjice) in 200 (posodica) s periodo ~2s
    freq_vst = 0.5   # 2s period
    freq_pos = 0.6
    t_preklop = 20.0

    ys = np.where(
        casi < t_preklop,
        125 + 75 * np.sin(2 * np.pi * freq_vst * casi),   # vstavljanje
        125 + 75 * np.sin(2 * np.pi * freq_pos * casi + np.pi)  # pospravljanje
    ) + np.random.randn(len(casi)) * 5

    xs = np.ones_like(ys) * 32   # X konstanten

    analizator = AnalizatorYOsi(fps=fps, min_prominenca_mm=30)
    rezultat = analizator.analiziraj(
        casi, xs, ys,
        t_zacetek=4.0,
        t_preklop_pospravljanja=t_preklop,
        t_konec=34.0
    )

    print(f"\nVstavljanje: {len(rezultat['cicli_vstavljanje'])}")
    print(f"Pospravljanje: {len(rezultat['cicli_pospravljanje'])}")

    analizator.narisi_graf(rezultat, '/tmp/test_y_analiza.png')
    print("Test OK")