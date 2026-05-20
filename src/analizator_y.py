# analizator_y.py v3
# Zaznava ciklov 9HPT testa iz Y koordinate roke v mm prostoru.
#
# PRISTOP:
#   - Lokalni ekstremi (vrhovi/doline) za zaznavo trenutkov pickup/insert
#   - 3 fizična območja iz homografije za filtriranje ekstremov:
#       OBMOČJE_A:   Y < meja_A       ← luknjice zgornje (world_A)
#       OBMOČJE_POS: meja_A < Y < meja_B  ← posodica
#       OBMOČJE_B:   Y > meja_B       ← luknjice spodnje (world_B)
#   - Pravilo: doline samo v A ali B, vrhovi samo v POS ali B (odvisno od aktivne mreže)
#   - ZG mreža: vstavljanje = vrh(POS) → dolina(A), pospravljanje = dolina(A) → vrh(POS)
#   - SP mreža: vstavljanje = dolina(POS) → vrh(B), pospravljanje = vrh(B) → dolina(POS)

import numpy as np
from scipy.signal import savgol_filter, find_peaks


# ===== PARAMETRI =====

SG_OKNO_Y       = 15
SG_RED_Y        = 2
MIN_PROMINENCA  = 15.0   # mm
MIN_RAZDALJA_S  = 0.5   # s


class AnalizatorYOsi:

    def __init__(self, fps=25.0):
        self.fps          = fps
        self.y_pos_center = 170.0
        self.y_luk_A      = 32.0
        self.meja_A       = 101.0
        self.meja_B       = 239.0
        self.aktivna_mreza = 'ZG'

    def _nastavi_meje(self, y_pos_center):
        self.y_pos_center = y_pos_center
        y_luk_B      = y_pos_center * 2 - self.y_luk_A
        self.meja_A  = (self.y_luk_A + y_pos_center) / 2.0
        self.meja_B  = (y_pos_center + y_luk_B) / 2.0
        return self.meja_A, self.meja_B

    def _zgladiti_y(self, ys):
        n  = len(ys)
        ys = np.array(ys, dtype=float)
        if n < SG_OKNO_Y:
            return ys
        okno = SG_OKNO_Y if SG_OKNO_Y % 2 == 1 else SG_OKNO_Y + 1
        try:
            return savgol_filter(ys, okno, SG_RED_Y)
        except Exception:
            return ys

    def _najdi_ekstreme(self, casi, ys_gl):
        """
        Najde lokalne ekstreme in jih filtrira glede na fizična območja.

        ZG mreža:
          - Vrhovi (posodica) → morajo biti v OBMOČJE_POS (meja_A < Y < meja_B)
          - Doline (luknjice) → morajo biti v OBMOČJE_A (Y < meja_A)

        SP mreža:
          - Doline (posodica) → morajo biti v OBMOČJE_POS (meja_A < Y < meja_B)
          - Vrhovi (luknjice) → morajo biti v OBMOČJE_B (Y > meja_B)
        """
        min_razdalja = max(3, int(MIN_RAZDALJA_S * self.fps))

        idx_vrhovi, _ = find_peaks(
             ys_gl, prominence=MIN_PROMINENCA, distance=min_razdalja)
        idx_doline, _ = find_peaks(
            -ys_gl, prominence=MIN_PROMINENCA, distance=min_razdalja)

        if self.aktivna_mreza == 'ZG':
            # Vrhovi = posodica → v območju POS
            mask_v = ((ys_gl[idx_vrhovi] > self.meja_A) &
                      (ys_gl[idx_vrhovi] < self.meja_B))
            idx_vrhovi = idx_vrhovi[mask_v]
            # Doline = luknjice A → v območju A
            mask_d = ys_gl[idx_doline] < self.meja_A
            idx_doline = idx_doline[mask_d]

        elif self.aktivna_mreza == 'SP':
            # Doline = posodica → v območju POS
            mask_d = ((ys_gl[idx_doline] > self.meja_A) &
                      (ys_gl[idx_doline] < self.meja_B))
            idx_doline = idx_doline[mask_d]
            # Vrhovi = luknjice B → v območju B
            mask_v = ys_gl[idx_vrhovi] > self.meja_B
            idx_vrhovi = idx_vrhovi[mask_v]

        # Deduplikacija: če sta dva ekstrema istega tipa bližje kot MIN_RAZDALJA_S,
        # ohrani samo tistega z večjo prominenco
        def dedupliciraj(idx_arr, ys_gl, min_razdalja_frame):
            if len(idx_arr) < 2:
                return idx_arr
            ohranjeni = []
            i = 0
            while i < len(idx_arr):
                # Zberi vse ekstreme ki so si blizu
                skupina = [idx_arr[i]]
                while (i + 1 < len(idx_arr) and
                       idx_arr[i+1] - idx_arr[i] < min_razdalja_frame * 2):
                    i += 1
                    skupina.append(idx_arr[i])
                # Iz skupine izberi tistega z največjo absolutno vrednostjo
                najboljsi = max(skupina, key=lambda x: abs(ys_gl[x]))
                ohranjeni.append(najboljsi)
                i += 1
            return np.array(sorted(ohranjeni))

        min_razdalja = max(3, int(MIN_RAZDALJA_S * self.fps))
        idx_vrhovi = dedupliciraj(idx_vrhovi, ys_gl, min_razdalja)
        idx_doline = dedupliciraj(idx_doline, ys_gl, min_razdalja)

        return idx_vrhovi, idx_doline

    def _sestavi_cikle_zg(self, casi, ys_gl, idx_vrhovi, idx_doline,
                           t_od, t_do, verbose):
        """
        ZG mreža:
          Vstavljanje:    vrh(POS) → dolina(A)
          Pospravljanje:  dolina(A) → vrh(POS)
        """
        def v_oknu(idx):
            mask = np.ones(len(idx), dtype=bool)
            if t_od: mask &= casi[idx] >= t_od
            if t_do: mask &= casi[idx] <= t_do
            return idx[mask]

        vrhovi = v_oknu(idx_vrhovi)
        doline = v_oknu(idx_doline)
        cicli_v, cicli_p = [], []

        # Vstavljanje: za vsak vrh → naslednja dolina
        for v_idx in vrhovi:
            nsl = doline[doline > v_idx]
            if not len(nsl): continue
            d_idx = nsl[0]
            mt = casi[d_idx] - casi[v_idx]
            if mt < 0.15: continue
            cicli_v.append(self._cikel(
                casi[v_idx], casi[d_idx], mt, 'VSTAVLJANJE'))
            if verbose:
                print(f"  [Y-V] t={casi[v_idx]:.2f}→{casi[d_idx]:.2f}s  "
                      f"vrh(POS)→dolina(A)  mt={mt:.2f}s")
            if len(cicli_v) >= 9: break

        # Pospravljanje: za vsako dolino → naslednji vrh
        for d_idx in doline:
            nsl = vrhovi[vrhovi > d_idx]
            if not len(nsl): continue
            v_idx = nsl[0]
            mt = casi[v_idx] - casi[d_idx]
            if mt < 0.15: continue
            cicli_p.append(self._cikel(
                casi[d_idx], casi[v_idx], mt, 'POSPRAVLJANJE'))
            if verbose:
                print(f"  [Y-P] t={casi[d_idx]:.2f}→{casi[v_idx]:.2f}s  "
                      f"dolina(A)→vrh(POS)  mt={mt:.2f}s")
            if len(cicli_p) >= 9: break

        return cicli_v, cicli_p

    def _sestavi_cikle_sp(self, casi, ys_gl, idx_vrhovi, idx_doline,
                           t_od, t_do, verbose):
        """
        SP mreža — vlogi sta zamenjani:
          Vstavljanje:    dolina(POS) → vrh(B)
          Pospravljanje:  vrh(B) → dolina(POS)
        """
        def v_oknu(idx):
            mask = np.ones(len(idx), dtype=bool)
            if t_od: mask &= casi[idx] >= t_od
            if t_do: mask &= casi[idx] <= t_do
            return idx[mask]

        vrhovi = v_oknu(idx_vrhovi)   # luknjice B
        doline = v_oknu(idx_doline)   # posodica
        cicli_v, cicli_p = [], []

        # Vstavljanje: za vsako dolino(POS) → naslednji vrh(B)
        for d_idx in doline:
            nsl = vrhovi[vrhovi > d_idx]
            if not len(nsl): continue
            v_idx = nsl[0]
            mt = casi[v_idx] - casi[d_idx]
            if mt < 0.15: continue
            cicli_v.append(self._cikel(
                casi[d_idx], casi[v_idx], mt, 'VSTAVLJANJE'))
            if verbose:
                print(f"  [Y-V] t={casi[d_idx]:.2f}→{casi[v_idx]:.2f}s  "
                      f"dolina(POS)→vrh(B)  mt={mt:.2f}s")
            if len(cicli_v) >= 9: break

        # Pospravljanje: za vsak vrh(B) → naslednja dolina(POS)
        for v_idx in vrhovi:
            nsl = doline[doline > v_idx]
            if not len(nsl): continue
            d_idx = nsl[0]
            mt = casi[d_idx] - casi[v_idx]
            if mt < 0.15: continue
            cicli_p.append(self._cikel(
                casi[v_idx], casi[d_idx], mt, 'POSPRAVLJANJE'))
            if verbose:
                print(f"  [Y-P] t={casi[v_idx]:.2f}→{casi[d_idx]:.2f}s  "
                      f"vrh(B)→dolina(POS)  mt={mt:.2f}s")
            if len(cicli_p) >= 9: break

        return cicli_v, cicli_p

    def _cikel(self, t_start, t_end, mt, faza):
        return {
            'pickup_start':    t_start,
            'pickup_complete': t_start,
            'insert_start':    t_end - 0.1,
            'insert_complete': t_end,
            'movement_time':   mt,
            'pickup_duration': 0.0,
            'insert_duration': 0.1,
            'faza':            faza,
            'metoda':          'Y-v3',
        }

    def analiziraj(self, casi, xs_mm, ys_mm,
                   t_zacetek=None,
                   t_preklop_pospravljanja=None,
                   t_konec=None,
                   verbose=True):

        casi  = np.array(casi)
        ys_mm = np.array(ys_mm)
        xs_mm = np.array(xs_mm)

        if len(casi) < 10:
            return self._prazen_rezultat()

        ys_gl = self._zgladiti_y(ys_mm)

        if verbose:
            print(f"[AnalizatorY] Mreza={self.aktivna_mreza}  "
                  f"Meje: A<{self.meja_A:.0f}  "
                  f"POS={self.meja_A:.0f}-{self.meja_B:.0f}  "
                  f"B>{self.meja_B:.0f}mm")

        idx_vrhovi, idx_doline = self._najdi_ekstreme(casi, ys_gl)

        if verbose:
            print(f"[AnalizatorY] Vrhovi: {len(idx_vrhovi)}  "
                  f"Doline: {len(idx_doline)}")

        t_preklop_eff = t_preklop_pospravljanja or t_konec

        if self.aktivna_mreza == 'SP':
            cicli_v, _ = self._sestavi_cikle_sp(
                casi, ys_gl, idx_vrhovi, idx_doline,
                t_od=t_zacetek, t_do=t_preklop_eff, verbose=verbose)
            t_meja = cicli_v[-1]['insert_complete'] if cicli_v else t_preklop_eff
            _, cicli_p = self._sestavi_cikle_sp(
                casi, ys_gl, idx_vrhovi, idx_doline,
                t_od=t_meja, t_do=t_konec, verbose=verbose)
        else:
            cicli_v, _ = self._sestavi_cikle_zg(
                casi, ys_gl, idx_vrhovi, idx_doline,
                t_od=t_zacetek, t_do=t_preklop_eff, verbose=verbose)
            t_meja = cicli_v[-1]['insert_complete'] if cicli_v else t_preklop_eff
            _, cicli_p = self._sestavi_cikle_zg(
                casi, ys_gl, idx_vrhovi, idx_doline,
                t_od=t_meja, t_do=t_konec, verbose=verbose)

        # # Filtriraj po t_preklop
        # if t_preklop_pospravljanja is not None:
        #     cicli_v = [c for c in cicli_v
        #                if c['pickup_start'] < t_preklop_pospravljanja + 1.0]
        #     cicli_p = [c for c in cicli_p
        #                if c['pickup_start'] > t_preklop_pospravljanja - 1.0]

        cicli_v = cicli_v[:9]
        cicli_p = cicli_p[:9]

        if verbose:
            print(f"[AnalizatorY] Vstavljanje: {len(cicli_v)}/9")
            print(f"[AnalizatorY] Pospravljanje: {len(cicli_p)}/9")

        return {
            'cicli_vstavljanje':   cicli_v,
            'cicli_pospravljanje': cicli_p,
            'meja_A':              self.meja_A,
            'meja_B':              self.meja_B,
            'y_meja_mm':           self.meja_A,
            'ys_gl':               ys_gl,
            'idx_vrhovi':          idx_vrhovi,
            'idx_doline':          idx_doline,
            'casi':                casi,
            'xs_mm':               xs_mm,
            'ys_mm':               ys_mm,
            'zaporedje':           [],
            'aktivna_mreza':       self.aktivna_mreza,
        }

    def _prazen_rezultat(self):
        return {
            'cicli_vstavljanje':   [],
            'cicli_pospravljanje': [],
            'meja_A':              self.meja_A,
            'meja_B':              self.meja_B,
            'y_meja_mm':           None,
            'ys_gl':               np.array([]),
            'idx_vrhovi':          np.array([]),
            'idx_doline':          np.array([]),
            'casi':                np.array([]),
            'xs_mm':               np.array([]),
            'ys_mm':               np.array([]),
            'zaporedje':           [],
            'aktivna_mreza':       self.aktivna_mreza,
        }

    def narisi_graf(self, rezultat, izhod_pot=None):
        import matplotlib.pyplot as plt

        casi   = rezultat['casi']
        ys_mm  = rezultat['ys_mm']
        ys_gl  = rezultat['ys_gl']
        idx_v  = rezultat['idx_vrhovi']
        idx_d  = rezultat['idx_doline']
        meja_A = rezultat.get('meja_A', self.meja_A)
        meja_B = rezultat.get('meja_B', self.meja_B)
        mreza  = rezultat.get('aktivna_mreza', self.aktivna_mreza)

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        ax = axs[0]

        if len(casi) > 0:
            y_min = min(np.min(ys_mm) - 20, -50)
            y_max = max(np.max(ys_mm) + 20, 400)

            # Območja
            ax.axhspan(y_min,  meja_A, alpha=0.07, color='steelblue',
                       label=f'Luknjice A (<{meja_A:.0f}mm)')
            ax.axhspan(meja_A, meja_B, alpha=0.07, color='darkorange',
                       label=f'Posodica ({meja_A:.0f}-{meja_B:.0f}mm)')
            ax.axhspan(meja_B, y_max,  alpha=0.07, color='seagreen',
                       label=f'Luknjice B (>{meja_B:.0f}mm)')
            ax.axhline(meja_A, color='steelblue', lw=1, linestyle='--', alpha=0.5)
            ax.axhline(meja_B, color='seagreen',  lw=1, linestyle='--', alpha=0.5)

            ax.plot(casi, ys_mm, color='lightsteelblue', alpha=0.4, lw=0.8,
                    label='Y surovi')
            ax.plot(casi, ys_gl, color='steelblue', lw=1.5, label='Y zglajen')

            # Ekstremi
            if len(idx_v):
                lbl = 'vrh (luknjice B)' if mreza == 'SP' else 'vrh (posodica)'
                ax.scatter(casi[idx_v], ys_gl[idx_v], color='darkorange',
                           s=80, zorder=5, marker='^', label=lbl)
            if len(idx_d):
                lbl = 'dolina (posodica)' if mreza == 'SP' else 'dolina (luknjice A)'
                ax.scatter(casi[idx_d], ys_gl[idx_d], color='steelblue',
                           s=80, zorder=5, marker='v', label=lbl)

            # Cikli
            for c in rezultat.get('cicli_vstavljanje', []):
                ax.axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.15, color='green')
                ax.axvline(c['pickup_start'],    color='green', alpha=0.7, lw=1)
                ax.axvline(c['insert_complete'], color='lime',  alpha=0.7, lw=1)

            for c in rezultat.get('cicli_pospravljanje', []):
                ax.axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.15, color='tomato')
                ax.axvline(c['pickup_start'],    color='tomato', alpha=0.7, lw=1)
                ax.axvline(c['insert_complete'], color='red',    alpha=0.7, lw=1)

        ax.invert_yaxis()
        ax.set_ylabel('Y [mm]')
        n_v = len(rezultat.get('cicli_vstavljanje', []))
        n_p = len(rezultat.get('cicli_pospravljanje', []))
        ax.set_title(f'Y koordinata roke [{mreza} mreža] — V:{n_v}  P:{n_p}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        # Movement time
        ax = axs[1]
        mt_v = [c['movement_time'] for c in rezultat.get('cicli_vstavljanje', [])]
        mt_p = [c['movement_time'] for c in rezultat.get('cicli_pospravljanje', [])]
        if mt_v:
            ax.bar(np.arange(len(mt_v)) - 0.2, mt_v, 0.35,
                   label='Vstavljanje', color='steelblue', alpha=0.8)
        if mt_p:
            ax.bar(np.arange(len(mt_p)) + 0.2, mt_p, 0.35,
                   label='Pospravljanje', color='tomato', alpha=0.8)
        ax.set_ylabel('Movement time [s]')
        ax.set_xlabel('Cikel #')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        if izhod_pot:
            plt.savefig(izhod_pot, dpi=130)
            plt.close()
            print(f"[AnalizatorY] Graf shranjen: {izhod_pot}")
        else:
            plt.show()


# ===== INTEGRACIJA =====

def analiziraj_iz_rezultatov(rezultati_combined, verbose=True):
    casi_hom = rezultati_combined.get('casi_hom', np.array([]))
    traj_mm  = rezultati_combined.get('traj_mm', [])

    if len(casi_hom) == 0 or len(traj_mm) == 0:
        print("[AnalizatorY] Ni podatkov")
        return None

    xs_mm = np.array([p[0] for p in traj_mm])
    ys_mm = np.array([p[1] for p in traj_mm])

    led       = rezultati_combined.get('led_stroj')
    fps       = 25.0
    t_zacetek = None
    t_preklop = None
    t_konec   = None

    analizator = AnalizatorYOsi(fps=fps)

    if led:
        for d in led.dogodki:
            if d['tip'] == 'TEST_ZACETEK_ZG':
                t_zacetek = d['cas']
                analizator.aktivna_mreza = 'ZG'
            elif d['tip'] == 'TEST_ZACETEK_SP':
                t_zacetek = d['cas']
                analizator.aktivna_mreza = 'SP'
            elif d['tip'] == 'ZACETEK_POSPRAVLJANJA':
                t_preklop = d['cas']
            elif d['tip'] == 'TEST_KONEC':
                t_konec = d['cas']

    hom = rezultati_combined.get('homografija')
    if hom and hom.center_posodice_mm:
        y_pos = hom.center_posodice_mm[1]
        meja_A, meja_B = analizator._nastavi_meje(y_pos)
        if verbose:
            print(f"[AnalizatorY] Meje: A<{meja_A:.0f}  "
                  f"POS={meja_A:.0f}-{meja_B:.0f}  B>{meja_B:.0f}mm  "
                  f"[mreza={analizator.aktivna_mreza}]")

    return analizator.analiziraj(
        casi_hom, xs_mm, ys_mm,
        t_zacetek=t_zacetek,
        t_preklop_pospravljanja=t_preklop,
        t_konec=t_konec,
        verbose=verbose,
    )


# ===== TEST =====
if __name__ == "__main__":
    fps = 25.0
    casi = np.linspace(0, 35, int(35 * fps))
    t_preklop = 20.0

    # ZG simulacija
    ys = np.where(
        casi < t_preklop,
        50 + 100 * np.abs(np.sin(np.pi * 0.5 * casi)),
        50 + 100 * np.abs(np.sin(np.pi * 0.6 * casi)),
    ) + np.random.randn(len(casi)) * 3

    a = AnalizatorYOsi(fps=fps)
    a._nastavi_meje(170.0)
    a.aktivna_mreza = 'ZG'

    r = a.analiziraj(casi, np.ones_like(ys)*32, ys,
                     t_zacetek=4.0,
                     t_preklop_pospravljanja=t_preklop,
                     t_konec=34.0, verbose=True)

    print(f"V={len(r['cicli_vstavljanje'])}  P={len(r['cicli_pospravljanje'])}")
    a.narisi_graf(r, '/tmp/test_v3.png')
    print("OK")