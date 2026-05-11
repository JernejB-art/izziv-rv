# preberi_csv.py
import csv
import numpy as np
import os
import glob

def preberi_csv_pacienta(pot_do_podatkov, id_pacienta):
    vzorec = os.path.join(pot_do_podatkov, id_pacienta, f'{id_pacienta}*.csv')
    datoteke = glob.glob(vzorec)

    if not datoteke:
        raise FileNotFoundError(f'CSV za {id_pacienta} ni najden')

    pot_csv = datoteke[0]
    print(f'Berem: {pot_csv}')

    with open(pot_csv, newline='') as f:
        bralnik = csv.reader(f)
        vrstice = list(bralnik)

    # Kumulativni časi zatičev (vrstice 1-9, stolpci 1-8)
    kumulativni = {}
    stolpci = ['post_P1','posp_P1','post_P2','posp_P2',
               'post_S1','posp_S1','post_S2','posp_S2']

    for i, ime in enumerate(stolpci):
        vrednosti = []
        for vrstica in vrstice[1:10]:  # 9 zatičev
            try:
                vrednosti.append(float(vrstica[i+1]))
            except (ValueError, IndexError):
                vrednosti.append(np.nan)
        kumulativni[ime] = np.array(vrednosti)

    # Skupni časi (vrstica 10, indeksi: P1=2, P2=4, S1=6, S2=8)
    total = vrstice[10]
    skupni_casi = {
        'P1': float(total[2]),
        'P2': float(total[4]),
        'S1': float(total[6]),
        'S2': float(total[8]),
    }

    # Metadata (vrstica 12)
    meta = vrstice[12]
    metadata = {
        'spol':     meta[4] if len(meta) > 4 else '?',
        'diagnoza': meta[5] if len(meta) > 5 else '?',
        'roka':     meta[6] if len(meta) > 6 else '?',
        'datum':    meta[7] if len(meta) > 7 else '?',
    }

    # Posamezni časi = razlike kumulativnih
    posamezni = {}
    for ime, kum in kumulativni.items():
        posamezni[ime] = np.diff(kum, prepend=0)

    return {
        'id':          id_pacienta,
        'skupni_casi': skupni_casi,
        'kumulativni': kumulativni,
        'posamezni':   posamezni,
        'metadata':    metadata,
    }

def izpisi_rezultate(r):
    print(f"\n{'='*80}")
    print(f"Pacient: {r['id']}")
    print(f"Spol: {r['metadata']['spol']},  "
          f"Diagnoza: {r['metadata']['diagnoza']},  "
          f"Roka: {r['metadata']['roka']}")
    print(f"{'='*80}")

    print(f"\nSkupni časi:")
    for k, v in r['skupni_casi'].items():
        print(f"  {k}: {v:.2f}s")

    print(f"\nPosamezni časi zatičev:")
    print(f"{'Zatič':>6} | {'post_P1':>8} | {'posp_P1':>8} | "
          f"{'post_P2':>8} | {'posp_P2':>8} | "
          f"{'post_S1':>8} | {'posp_S1':>8} | "
          f"{'post_S2':>8} | {'posp_S2':>8}")
    print("-" * 85)
    for i in range(9):
        vrs = [r['posamezni'][s][i]
               for s in ['post_P1','posp_P1','post_P2','posp_P2',
                         'post_S1','posp_S1','post_S2','posp_S2']]
        print(f"{i+1:>6} | " + " | ".join(f"{v:>8.2f}" for v in vrs))

if __name__ == "__main__":
    rezultat = preberi_csv_pacienta('/data/Data/', 'patient_167')
    izpisi_rezultate(rezultat)