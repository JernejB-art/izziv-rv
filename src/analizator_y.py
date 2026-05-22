# analizator_y_zone.py v2
# Pristop tekočega ekstrema po conah.
#
# ZG mreža: posodica = Y < meja_A, luknjice = Y > meja_B
#   vstavljanje: max(Y) v obisku posodice → min(Y) v obisku luknjic
#   pospravljanje: min(Y) v obisku luknjic → max(Y) v obisku posodice
#
# SP mreža: posodica = Y > meja_B, luknjice = Y < meja_A
#   vstavljanje: min(Y) v obisku posodice → max(Y) v obisku luknjic
#   pospravljanje: max(Y) v obisku luknjic → min(Y) v obisku posodice

import numpy as np
from scipy.signal import savgol_filter

SG_OKNO_Y      = 9
SG_RED_Y       = 2
MIN_CAS_V_CONI = 0.2   # s — minimalni čas v coni (filtrira šumne kratke prehode)
MIN_AMP_MM     = 15.0  # mm — minimalna amplituda med ekstremoma

# Tip cone
CONA_POS = 'POS'
CONA_LUK = 'LUK'
CONA_NIK = None


class AnalizatorYOsi:

    def __init__(self, fps=25.0):
        self.fps           = fps
        self.meja_A        = 101.0
        self.meja_B        = 239.0
        self.aktivna_mreza = 'ZG'

    def _nastavi_meje(self, y_pos_center, y_luk_A=32.0):
        """
        Nastavi meje con glede na center posodice in položaj luknjic.
        y_luk_A = Y koordinata luknjic ZG mreže (majhen Y = zgoraj).
        Za SP mrežo so luknjice na nasprotni strani (velik Y = spodaj).
        meja_A = meja med luknjicami ZG in posodico (sredina)
        meja_B = meja med posodico in luknjicami SP (sredina)
        """
        y_luk_B     = y_pos_center * 2 - y_luk_A
        self.meja_A = (y_luk_A + y_pos_center) / 2.0
        self.meja_B = (y_pos_center + y_luk_B) / 2.0
        self.y_pos_center = y_pos_center
        self.y_luk_A = y_luk_A
        self.y_luk_B = y_luk_B
        return self.meja_A, self.meja_B


    def _zgladiti(self, ys):
        ys = np.array(ys, dtype=float)
        if len(ys) < SG_OKNO_Y:
            return ys
        okno = SG_OKNO_Y if SG_OKNO_Y % 2 == 1 else SG_OKNO_Y + 1
        try:
            return savgol_filter(ys, okno, SG_RED_Y)
        except Exception:
            return ys

    def _katera_cona(self, y):
        """
        ZG mreža: posodica = Y > meja_A  (spodaj = velik Y)
                  luknjice = Y < meja_A  (zgoraj = majhen Y)
        SP mreža: posodica = Y < meja_B  (zgoraj = majhen Y)
                  luknjice = Y > meja_B  (spodaj = velik Y)
        """
        if self.aktivna_mreza == 'ZG':
            if y > self.meja_A:
                return CONA_POS   # posodica: Y > meja_A
            else:
                return CONA_LUK   # luknjice: Y < meja_A
        else:  # SP
            if y < self.meja_B:
                return CONA_POS   # posodica: Y < meja_B
            else:
                return CONA_LUK   # luknjice: Y > meja_B

    def _ekstrem_cone(self, cona):
        """
        ZG: posodica = max Y (spodaj),  luknjice = min Y (zgoraj)
        SP: posodica = min Y (zgoraj),  luknjice = max Y (spodaj)
        """
        if self.aktivna_mreza == 'ZG':
            return 'max' if cona == CONA_POS else 'min'
        else:  # SP
            return 'min' if cona == CONA_POS else 'max'


    def _poberi_obiske(self, casi, ys_gl, t_od=None, t_do=None):
        """
        Pregleda signal frame po frame in vrne seznam obiskov con.
        Vsak obisk = dict z:
            cona, t_start, t_end, t_ekstrem, y_ekstrem, i_ekstrem
        Filtrira obiske krajše od MIN_CAS_V_CONI.
        """
        min_frames = max(2, int(MIN_CAS_V_CONI * self.fps))
        obiski = []
        trenutna_cona = CONA_NIK
        i_start = 0
        tek_ext_y = None
        tek_ext_i = None

        for i in range(len(casi)):
            t = casi[i]
            if t_od and t < t_od:
                continue
            if t_do and t > t_do:
                break

            y = ys_gl[i]
            cona = self._katera_cona(y)

            if cona != trenutna_cona:
                # Zaključi prejšnji obisk
                if trenutna_cona is not None and tek_ext_i is not None:
                    dolzina = i - i_start
                    if dolzina >= min_frames:
                        obiski.append({
                            'cona':      trenutna_cona,
                            't_start':   casi[i_start],
                            't_end':     casi[i - 1],
                            't_ekstrem': casi[tek_ext_i],
                            'y_ekstrem': tek_ext_y,
                            'i_ekstrem': tek_ext_i,
                        })
                # Začni nov obisk
                trenutna_cona = cona
                i_start = i
                tek_ext_y = y if cona is not None else None
                tek_ext_i = i if cona is not None else None
            elif cona is not None:
                # Posodobi tekoči ekstrem
                tip = self._ekstrem_cone(cona)
                if tip == 'max' and y > tek_ext_y:
                    tek_ext_y = y
                    tek_ext_i = i
                elif tip == 'min' and y < tek_ext_y:
                    tek_ext_y = y
                    tek_ext_i = i

        # Zaključi zadnji obisk
        if trenutna_cona is not None and tek_ext_i is not None:
            dolzina = len(casi) - i_start
            if dolzina >= min_frames:
                obiski.append({
                    'cona':      trenutna_cona,
                    't_start':   casi[i_start],
                    't_end':     casi[-1],
                    't_ekstrem': casi[tek_ext_i],
                    'y_ekstrem': tek_ext_y,
                    'i_ekstrem': tek_ext_i,
                })

        return obiski

    def _sestavi_cikle(self, obiski, faza, verbose):
        """
        Iz seznama obiskov sestavi cikle.
        faza='VSTAVLJANJE':   POS → LUK
        faza='POSPRAVLJANJE': LUK → POS
        """
        if faza == 'VSTAVLJANJE':
            prva, druga = CONA_POS, CONA_LUK
        else:
            prva, druga = CONA_LUK, CONA_POS

        cicli = []
        porabljeni = set()

        for i, ob1 in enumerate(obiski):
            if ob1['cona'] != prva:
                continue
            if i in porabljeni:
                continue

            # Poišči naslednji obisk druge cone
            for j in range(i + 1, len(obiski)):
                if j in porabljeni:
                    continue
                ob2 = obiski[j]
                if ob2['cona'] != druga:
                    continue
                # Mora se začeti po koncu prvega
                if ob2['t_start'] <= ob1['t_end']:
                    continue

                t_start = ob1['t_ekstrem']
                t_end   = ob2['t_ekstrem']
                mt = t_end - t_start
                amp = abs(ob2['y_ekstrem'] - ob1['y_ekstrem'])

                if mt < 0.1:
                    break
                if amp < MIN_AMP_MM:
                    break

                porabljeni.add(i)
                porabljeni.add(j)

                cicli.append({
                    'pickup_start':    t_start,
                    'pickup_complete': t_start,
                    'insert_start':    t_end - 0.1,
                    'insert_complete': t_end,
                    'movement_time':   mt,
                    'pickup_duration': 0.0,
                    'insert_duration': 0.1,
                    'faza':            faza,
                    'metoda':          'Y-zone-v2',
                })

                if verbose:
                    print(f"  [Z] {faza}: t={t_start:.2f}→{t_end:.2f}s  "
                          f"mt={mt:.2f}s  amp={amp:.1f}mm")

                if len(cicli) >= 9:
                    return cicli
                break  # vsak prvi obisk se poveže z najbližjim naslednjim

        return cicli

    def _zaznaj_preklop(self, obiski, casi):
        """Največja časovna vrzel med zaporednimi obiski = naravni premor."""
        if len(obiski) < 4:
            return None
        casi_zac = [o['t_start'] for o in obiski]
        vrzeli = np.diff(casi_zac)
        n = len(vrzeli)
        od, do = n // 4, 3 * n // 4
        if od >= do:
            return None
        i_max = od + int(np.argmax(vrzeli[od:do]))
        return (casi_zac[i_max] + casi_zac[i_max + 1]) / 2.0

    def analiziraj(self, casi, xs_mm, ys_mm,
                   t_zacetek=None, t_preklop_pospravljanja=None,
                   t_konec=None, verbose=True):

        casi  = np.array(casi)
        ys_mm = np.array(ys_mm)
        xs_mm = np.array(xs_mm)

        if len(casi) < 10:
            return self._prazen_rezultat()

        ys_gl = self._zgladiti(ys_mm)

        if verbose:
            print(f"[AnalizatorZone v2] Mreza={self.aktivna_mreza}  "
                  f"Meje: A={self.meja_A:.0f}  B={self.meja_B:.0f}mm")

        # Poberi vse obiske con
        vsi_obiski = self._poberi_obiske(casi, ys_gl,
                                         t_od=t_zacetek, t_do=t_konec)
        if verbose:
            n_pos = sum(1 for o in vsi_obiski if o['cona'] == CONA_POS)
            n_luk = sum(1 for o in vsi_obiski if o['cona'] == CONA_LUK)
            print(f"[AnalizatorZone v2] Obiski: posodica={n_pos}  luknjice={n_luk}")

        # Določi t_preklop
        t_preklop_eff = (t_preklop_pospravljanja or
                         self._zaznaj_preklop(vsi_obiski, casi) or
                         t_konec)
        if verbose and t_preklop_eff:
            print(f"[AnalizatorZone v2] t_preklop={t_preklop_eff:.2f}s")

        # Vstavljanje: obiski do t_preklop
        obiski_v = [o for o in vsi_obiski
                    if t_preklop_eff is None or o['t_start'] <= t_preklop_eff]
        cicli_v = self._sestavi_cikle(obiski_v, 'VSTAVLJANJE', verbose)

        # t_meja = zadnje vstavljanje
        t_meja_p = (cicli_v[-1]['insert_complete']
                    if cicli_v else t_preklop_eff)

        # Pospravljanje: obiski po t_meja
        obiski_p = [o for o in vsi_obiski
                    if t_meja_p is None or o['t_start'] > t_meja_p]
        cicli_p = self._sestavi_cikle(obiski_p, 'POSPRAVLJANJE', verbose)

        cicli_v = cicli_v[:9]
        cicli_p = cicli_p[:9]

        if verbose:
            print(f"[AnalizatorZone v2] Vstavljanje: {len(cicli_v)}/9")
            print(f"[AnalizatorZone v2] Pospravljanje: {len(cicli_p)}/9")

        return {
            'cicli_vstavljanje':   cicli_v,
            'cicli_pospravljanje': cicli_p,
            'meja_A':              self.meja_A,
            'meja_B':              self.meja_B,
            'y_meja_mm':           self.meja_A,
            'ys_gl':               ys_gl,
            'idx_vrhovi':          np.array([o['i_ekstrem'] for o in vsi_obiski
                                             if o['cona'] == CONA_POS]),
            'idx_doline':          np.array([o['i_ekstrem'] for o in vsi_obiski
                                             if o['cona'] == CONA_LUK]),
            'casi':                casi,
            'xs_mm':               xs_mm,
            'ys_mm':               ys_mm,
            'zaporedje':           [],
            'aktivna_mreza':       self.aktivna_mreza,
            '_obiski':             vsi_obiski,
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        casi   = rezultat['casi']
        ys_mm  = rezultat['ys_mm']
        ys_gl  = rezultat['ys_gl']
        meja_A = rezultat['meja_A']
        meja_B = rezultat['meja_B']
        mreza  = rezultat['aktivna_mreza']
        obiski = rezultat.get('_obiski', [])

        fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        ax = axs[0]

        if len(casi) > 0:
            y_min = min(np.min(ys_mm) - 20, -50)
            y_max = max(np.max(ys_mm) + 20, 400)

            # Ozadje con — ena meja na mrežo
            if mreza == 'ZG':
                # ZG: posodica = Y > meja_A, luknjice = Y < meja_A
                ax.axhspan(y_min,  meja_A, alpha=0.06, color='seagreen',
                           label=f'Luknjice ZG (<{meja_A:.0f}mm)')
                ax.axhspan(meja_A, y_max,  alpha=0.06, color='steelblue',
                           label=f'Posodica ZG (>{meja_A:.0f}mm)')
                ax.axhline(meja_A, color='gray', lw=1.5, ls='--', alpha=0.7,
                           label=f'Meja ZG ({meja_A:.0f}mm)')
            else:
                # SP: posodica = Y < meja_B, luknjice = Y > meja_B
                ax.axhspan(y_min,  meja_B, alpha=0.06, color='steelblue',
                           label=f'Posodica SP (<{meja_B:.0f}mm)')
                ax.axhspan(meja_B, y_max,  alpha=0.06, color='seagreen',
                           label=f'Luknjice SP (>{meja_B:.0f}mm)')
                ax.axhline(meja_B, color='gray', lw=1.5, ls='--', alpha=0.7,
                           label=f'Meja SP ({meja_B:.0f}mm)')

            # Obarvaj obiske
            for o in obiski:
                barva = 'steelblue' if o['cona'] == CONA_POS else 'seagreen'
                ax.axvspan(o['t_start'], o['t_end'], alpha=0.15, color=barva)
                # Označi ekstrem
                ax.plot(o['t_ekstrem'], o['y_ekstrem'],
                        'v' if self._ekstrem_cone(o['cona']) == 'min' else '^',
                        color='darkorange', ms=7, zorder=5)

            ax.plot(casi, ys_mm, color='lightsteelblue', alpha=0.4, lw=0.8,
                    label='Y surovi')
            ax.plot(casi, ys_gl, color='steelblue', lw=1.5, label='Y zglajen')

            # Cikli
            for c in rezultat.get('cicli_vstavljanje', []):
                ax.axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.2, color='green')
                ax.axvline(c['pickup_start'],    color='green', lw=1, alpha=0.7)
                ax.axvline(c['insert_complete'], color='lime',  lw=1, alpha=0.7)

            for c in rezultat.get('cicli_pospravljanje', []):
                ax.axvspan(c['pickup_start'], c['insert_complete'],
                           alpha=0.2, color='tomato')
                ax.axvline(c['pickup_start'],    color='tomato', lw=1, alpha=0.7)
                ax.axvline(c['insert_complete'], color='red',    lw=1, alpha=0.7)

        ax.invert_yaxis()
        ax.set_ylabel('Y [mm]')
        n_v = len(rezultat.get('cicli_vstavljanje', []))
        n_p = len(rezultat.get('cicli_pospravljanje', []))
        ax.set_title(f'Y koordinata roke [Zone v2, {mreza}] — V:{n_v}  P:{n_p}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

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
        ax.set_xlabel('Čas [s]')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        if izhod_pot:
            plt.savefig(izhod_pot, dpi=130)
            plt.close()
            print(f"[AnalizatorZone v2] Graf: {izhod_pot}")
        else:
            plt.show()


# ===== INTEGRACIJA =====

def analiziraj_iz_rezultatov(rezultati_combined, verbose=True):
    casi_hom = rezultati_combined.get('casi_hom', np.array([]))
    traj_mm  = rezultati_combined.get('traj_mm', [])

    if len(casi_hom) == 0 or len(traj_mm) == 0:
        print("[AnalizatorZone v2] Ni podatkov")
        return None

    xs_mm = np.array([p[0] for p in traj_mm])
    ys_mm = np.array([p[1] for p in traj_mm])

    led       = rezultati_combined.get('led_stroj')
    t_zacetek = None
    t_preklop = None
    t_konec   = None

    analizator = AnalizatorYOsi(fps=25.0)

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
        analizator._nastavi_meje(y_pos)
        if verbose:
            print(f"[AnalizatorZone v2] Meje: A={analizator.meja_A:.0f}  "
                  f"B={analizator.meja_B:.0f}mm  "
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
    fps  = 25.0
    casi = np.linspace(0, 35, int(35 * fps))
    t_pr = 20.0
    np.random.seed(42)

    ys = np.ones(len(casi)) * 160
    for i in range(9):
        t_c = 2.0 + i * 1.9
        mask = (casi >= t_c) & (casi < t_c + 1.9)
        t_lok = (casi[mask] - t_c) / 1.9
        ys[mask] = 60 + 200 * np.abs(np.sin(np.pi * t_lok))

    for i in range(9):
        t_c = t_pr + 1.5 + i * 1.9
        mask = (casi >= t_c) & (casi < t_c + 1.9)
        t_lok = (casi[mask] - t_c) / 1.9
        ys[mask] = 260 - 200 * np.abs(np.sin(np.pi * t_lok))

    ys += np.random.randn(len(casi)) * 4

    a = AnalizatorYOsi(fps=fps)
    a._nastavi_meje(165.0)
    a.aktivna_mreza = 'ZG'

    r = a.analiziraj(casi, np.ones_like(ys) * 32, ys,
                     t_zacetek=1.0, t_preklop_pospravljanja=t_pr,
                     t_konec=34.0, verbose=True)

    print(f"\nV={len(r['cicli_vstavljanje'])}  P={len(r['cicli_pospravljanje'])}")
    a.narisi_graf(r, '/tmp/test_zone.png')
    print("Graf: /tmp/test_zone.png")