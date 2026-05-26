#!/usr/bin/env python3
# evalvacija.py
# Evalvacija Y-os algoritma za zaznavo ciklov 9HPT testa.
#
# Primerja zaznane movement_time vrednosti z referenčnimi CSV podatki.
# Izračuna: MAE, RMSE, Pearsonova korelacija, Spearmanova korelacija,
#           % pravilno zaznanih ciklov, Bland-Altman diagram.
#
# Uporaba:
#   from evalvacija import evalviraj_pacienta, narisi_evalvacijo
#   ali samostojno:
#   python3 evalvacija.py --pacient patient_024

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_MAPA = "/workspace/src"
if SRC_MAPA not in sys.path:
    sys.path.insert(0, SRC_MAPA)

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

PODATKI_MAPA = "/data/Data"
IZHOD_MAPA   = "/workspace/results"


# ===== IZRAČUN METRIK =====

def izracunaj_metrike(zaznano, referenca, ime=""):
    """
    Primerja zaznane čase (Y analizator) z referenco (CSV).

    zaznano  → list ali array movement_time vrednosti [s]
    referenca → array referenčnih časov [s]

    Vrne dict z metrikami.
    """
    # Filtriraj NaN in ničle iz reference
    ref_cist = np.array([r for r in referenca
                          if not np.isnan(r) and r > 0.0])
    zaz_arr  = np.array(zaznano, dtype=float)

    n_zaz = len(zaz_arr)
    n_ref = len(ref_cist)

    metrике = {
        "n_zaznanih":   n_zaz,
        "n_referenca":  n_ref,
        "tocnost":      n_zaz / max(n_ref, 1),
        "mae":          np.nan,
        "rmse":         np.nan,
        "pearson_r":    np.nan,
        "pearson_p":    np.nan,
        "spearman_r":   np.nan,
        "spearman_p":   np.nan,
        "srednja_zaz":  float(np.mean(zaz_arr))  if n_zaz > 0 else np.nan,
        "srednja_ref":  float(np.mean(ref_cist)) if n_ref > 0 else np.nan,
    }

    if n_zaz == 0 or n_ref == 0:
        return metrике

    # Poravnaj dolžino — vzamemo minimum
    n_min = min(n_zaz, n_ref)
    z = zaz_arr[:n_min]
    r = ref_cist[:n_min]

    metrике["n_primerjanih"] = n_min
    metrике["mae"]  = float(np.mean(np.abs(z - r)))
    metrике["rmse"] = float(np.sqrt(np.mean((z - r)**2)))
    metrике["bias"] = float(np.mean(z - r))   # sistematična napaka

    if n_min >= 3 and SCIPY_OK:
        try:
            rp, pp = pearsonr(z, r)
            rs, ps = spearmanr(z, r)
            metrике["pearson_r"]  = float(rp)
            metrике["pearson_p"]  = float(pp)
            metrике["spearman_r"] = float(rs)
            metrике["spearman_p"] = float(ps)
        except Exception:
            pass

    if ime:
        print(f"  [{ime}] "
              f"n={n_min}/{n_ref}  "
              f"MAE={metrике['mae']:.3f}s  "
              f"r={metrике.get('pearson_r', float('nan')):.3f}  "
              f"bias={metrике['bias']:+.3f}s")

    return metrике


# ===== EVALVACIJA ENEGA PACIENTA =====

def evalviraj_pacienta(rezultati_json_pot, csv_podatki,
                        izhod_mapa=None, verbose=True):
    """
    Evalvira rezultate enega pacienta.

    rezultati_json_pot → pot do JSON iz pipeline.py
    csv_podatki        → dict iz csv_reader.preberi_csv_pacienta()
    izhod_mapa         → kje shraniti grafe
    """
    with open(rezultati_json_pot, encoding="utf-8") as f:
        podatki = json.load(f)

    id_pac = podatki.get("povzetek", {}).get("id_pacienta", "?")
    videi  = podatki.get("videi", [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALVACIJA: {id_pac}")
        print(f"{'='*60}")
        meta = csv_podatki.get("metadata", {})
        print(f"  Spol: {meta.get('spol','?')}  "
              f"Diagnoza: {meta.get('diagnoza','?')}  "
              f"Roka: {meta.get('roka','?')}")

    vse_metrike = []

    for r_video in videi:
        ime_videa = r_video.get("video", "?")
        mt_v = r_video.get("mt_vstavljanje_y",   [])
        mt_p = r_video.get("mt_pospravljanje_y",  [])

        # Določi CSV stolpec glede na kamero
        if "camP_0" in ime_videa or "camP_1" in ime_videa:
            kljuc_v, kljuc_p = "post_P1", "posp_P1"
        else:
            kljuc_v, kljuc_p = "post_P2", "posp_P2"

        ref_v = csv_podatki["posamezni"].get(kljuc_v, np.array([]))
        ref_p = csv_podatki["posamezni"].get(kljuc_p, np.array([]))

        m_v = izracunaj_metrike(mt_v, ref_v,
                                 ime=f"{ime_videa} vstavljanje" if verbose else "")
        m_p = izracunaj_metrike(mt_p, ref_p,
                                 ime=f"{ime_videa} pospravljanje" if verbose else "")

        vse_metrike.append({
            "video":     ime_videa,
            "vstavljanje": m_v,
            "pospravljanje": m_p,
            "mt_v_zaz": mt_v,
            "mt_p_zaz": mt_p,
            "mt_v_ref": ref_v.tolist() if hasattr(ref_v, "tolist") else list(ref_v),
            "mt_p_ref": ref_p.tolist() if hasattr(ref_p, "tolist") else list(ref_p),
        })

    # Skupne metrike
    mae_v_vsi  = [m["vstavljanje"].get("mae", np.nan)
                  for m in vse_metrike if not np.isnan(m["vstavljanje"].get("mae", np.nan))]
    mae_p_vsi  = [m["pospravljanje"].get("mae", np.nan)
                  for m in vse_metrike if not np.isnan(m["pospravljanje"].get("mae", np.nan))]
    r_v_vsi    = [m["vstavljanje"].get("pearson_r", np.nan)
                  for m in vse_metrike if not np.isnan(m["vstavljanje"].get("pearson_r", np.nan))]

    skupne = {
        "mae_vstavljanje":    float(np.mean(mae_v_vsi))  if mae_v_vsi else np.nan,
        "mae_pospravljanje":  float(np.mean(mae_p_vsi))  if mae_p_vsi else np.nan,
        "pearson_vstavljanje": float(np.mean(r_v_vsi))   if r_v_vsi   else np.nan,
    }

    if verbose:
        print(f"\n  Skupne metrike:")
        print(f"  MAE vstavljanje:   {skupne['mae_vstavljanje']:.3f}s")
        print(f"  MAE pospravljanje: {skupne['mae_pospravljanje']:.3f}s")
        print(f"  Pearson r (vst.):  {skupne['pearson_vstavljanje']:.3f}")

    # Grafi
    if izhod_mapa:
        os.makedirs(izhod_mapa, exist_ok=True)
        narisi_evalvacijo(vse_metrike, id_pac, izhod_mapa)

    return {
        "id_pacienta":  id_pac,
        "metrike":      vse_metrike,
        "skupne":       skupne,
    }


# ===== GRAFI EVALVACIJE =====

def narisi_evalvacijo(vse_metrike, id_pac, izhod_mapa):
    """
    Nariše 4 grafe za evalvacijo:
    1. Zaznano vs referenca (scatter plot) — vstavljanje
    2. Zaznano vs referenca (scatter plot) — pospravljanje
    3. Bland-Altman diagram — vstavljanje
    4. Primerjava po zatičih (bar chart)
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Evalvacija Y-os algoritma — {id_pac}", fontsize=13)

    # Združi vse vrednosti čez vse videe
    zaz_v_vse, ref_v_vse = [], []
    zaz_p_vse, ref_p_vse = [], []

    for m in vse_metrike:
        z_v = m.get("mt_v_zaz", [])
        r_v = m.get("mt_v_ref", [])
        r_v_cist = [r for r in r_v if r > 0]
        n = min(len(z_v), len(r_v_cist))
        zaz_v_vse.extend(z_v[:n])
        ref_v_vse.extend(r_v_cist[:n])

        z_p = m.get("mt_p_zaz", [])
        r_p = m.get("mt_p_ref", [])
        r_p_cist = [r for r in r_p if r > 0]
        n = min(len(z_p), len(r_p_cist))
        zaz_p_vse.extend(z_p[:n])
        ref_p_vse.extend(r_p_cist[:n])

    zaz_v = np.array(zaz_v_vse)
    ref_v = np.array(ref_v_vse)
    zaz_p = np.array(zaz_p_vse)
    ref_p = np.array(ref_p_vse)

    # ── Graf 1: Scatter vstavljanje ──────────────────────────────────────
    ax = axs[0, 0]
    if len(zaz_v) > 0 and len(ref_v) > 0:
        ax.scatter(ref_v, zaz_v, color="steelblue", alpha=0.7, s=60)
        lim = max(max(ref_v), max(zaz_v)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="idealno")
        # Regresijska premica
        if len(ref_v) >= 3 and SCIPY_OK:
            r, _ = pearsonr(ref_v, zaz_v)
            z_fit = np.polyfit(ref_v, zaz_v, 1)
            p_fit = np.poly1d(z_fit)
            xs = np.linspace(0, lim, 50)
            ax.plot(xs, p_fit(xs), "steelblue", alpha=0.6,
                    label=f"fit (r={r:.2f})")
        ax.set_xlabel("Referenca [s]")
        ax.set_ylabel("Zaznano [s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax.set_title("Vstavljanje: zaznano vs referenca")

    # ── Graf 2: Scatter pospravljanje ────────────────────────────────────
    ax = axs[0, 1]
    if len(zaz_p) > 0 and len(ref_p) > 0:
        ax.scatter(ref_p, zaz_p, color="tomato", alpha=0.7, s=60)
        lim = max(max(ref_p), max(zaz_p)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="idealno")
        if len(ref_p) >= 3 and SCIPY_OK:
            r, _ = pearsonr(ref_p, zaz_p)
            z_fit = np.polyfit(ref_p, zaz_p, 1)
            p_fit = np.poly1d(z_fit)
            xs = np.linspace(0, lim, 50)
            ax.plot(xs, p_fit(xs), "tomato", alpha=0.6,
                    label=f"fit (r={r:.2f})")
        ax.set_xlabel("Referenca [s]")
        ax.set_ylabel("Zaznano [s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax.set_title("Pospravljanje: zaznano vs referenca")

    # ── Graf 3: Bland-Altman vstavljanje ─────────────────────────────────
    ax = axs[1, 0]
    if len(zaz_v) > 1 and len(ref_v) > 1:
        sredina = (zaz_v + ref_v) / 2
        razlika = zaz_v - ref_v
        bias    = np.mean(razlika)
        std     = np.std(razlika)
        ax.scatter(sredina, razlika, color="steelblue", alpha=0.7, s=60)
        ax.axhline(bias,          color="steelblue", lw=1.5,
                   label=f"bias={bias:.3f}s")
        ax.axhline(bias + 1.96*std, color="steelblue", lw=1,
                   linestyle="--", label=f"±1.96σ={1.96*std:.3f}s")
        ax.axhline(bias - 1.96*std, color="steelblue", lw=1,
                   linestyle="--")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlabel("Sredina (zaznano+ref)/2 [s]")
        ax.set_ylabel("Razlika (zaznano-ref) [s]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax.set_title("Bland-Altman — vstavljanje")

    # ── Graf 4: Primerjava po zatičih ────────────────────────────────────
    ax = axs[1, 1]
    n_max = 9
    # Povpreči čez vse videe
    mt_v_po_zaticih = []
    mt_v_ref_po_zaticih = []
    for i in range(n_max):
        vals_zaz = []
        vals_ref = []
        for m in vse_metrike:
            zv = m.get("mt_v_zaz", [])
            rv = [r for r in m.get("mt_v_ref", []) if r > 0]
            if i < len(zv):
                vals_zaz.append(zv[i])
            if i < len(rv):
                vals_ref.append(rv[i])
        mt_v_po_zaticih.append(np.mean(vals_zaz) if vals_zaz else np.nan)
        mt_v_ref_po_zaticih.append(np.mean(vals_ref) if vals_ref else np.nan)

    x = np.arange(n_max)
    w = 0.35
    zaz_arr2 = np.array(mt_v_po_zaticih)
    ref_arr2 = np.array(mt_v_ref_po_zaticih)
    mask = ~(np.isnan(zaz_arr2) | np.isnan(ref_arr2))
    if mask.any():
        ax.bar(x[mask]-w/2, zaz_arr2[mask], w, label="Zaznano",
               color="steelblue", alpha=0.8)
        ax.bar(x[mask]+w/2, ref_arr2[mask], w, label="Referenca",
               color="darkorange", alpha=0.8)
    ax.set_xlabel("Zatič #")
    ax.set_ylabel("Movement time [s]")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(n_max)])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Movement time po zatičih — vstavljanje")

    plt.tight_layout()
    izhod_pot = os.path.join(izhod_mapa, "evalvacija.png")
    plt.savefig(izhod_pot, dpi=130)
    plt.close()
    print(f"  [Evalvacija] Graf shranjen: {izhod_pot}")


# ===== SKUPNA EVALVACIJA VSEH PACIENTOV =====

def evalviraj_vse(izhod_mapa=IZHOD_MAPA,
                  pot_podatkov=PODATKI_MAPA,
                  verbose=False):
    """
    Evalvira vse paciente za katere obstaja JSON in CSV.
    Nariše skupni scatter plot in bar chart MAE po pacientih.
    """
    from csv_reader import preberi_csv_pacienta

    mape_pac = sorted([
        d for d in os.listdir(izhod_mapa)
        if os.path.isdir(os.path.join(izhod_mapa, d))
        and d.startswith("patient_")
    ])

    vsi_rezultati = []
    for id_pac in mape_pac:
        json_pot = os.path.join(izhod_mapa, id_pac,
                                f"{id_pac}_rezultati.json")
        if not os.path.exists(json_pot):
            continue
        try:
            csv = preberi_csv_pacienta(pot_podatkov, id_pac)
            r   = evalviraj_pacienta(json_pot, csv,
                                      izhod_mapa=os.path.join(izhod_mapa, id_pac),
                                      verbose=verbose)
            vsi_rezultati.append(r)
        except Exception as e:
            print(f"  [NAPAKA] {id_pac}: {e}")

    if not vsi_rezultati:
        print("Ni podatkov za evalvacijo.")
        return

    # Skupni graf MAE po pacientih
    _narisi_skupni_mae(vsi_rezultati, izhod_mapa)

    print(f"\n{'='*60}")
    print(f"SKUPNA EVALVACIJA — {len(vsi_rezultati)} pacientov")
    print(f"{'='*60}")
    mae_v = [r["skupne"]["mae_vstavljanje"]   for r in vsi_rezultati
             if not np.isnan(r["skupne"]["mae_vstavljanje"])]
    mae_p = [r["skupne"]["mae_pospravljanje"] for r in vsi_rezultati
             if not np.isnan(r["skupne"]["mae_pospravljanje"])]
    print(f"  Povp. MAE vstavljanje:   {np.mean(mae_v):.3f}s ± {np.std(mae_v):.3f}s")
    print(f"  Povp. MAE pospravljanje: {np.mean(mae_p):.3f}s ± {np.std(mae_p):.3f}s")

    return vsi_rezultati


def _narisi_skupni_mae(vsi_rezultati, izhod_mapa):
    """Bar chart MAE po pacientih."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MAE po pacientih — Y-os algoritem", fontsize=12)

    ids  = [r["id_pacienta"]                  for r in vsi_rezultati]
    mv   = [r["skupne"]["mae_vstavljanje"]     for r in vsi_rezultati]
    mp   = [r["skupne"]["mae_pospravljanje"]   for r in vsi_rezultati]
    rv   = [r["skupne"]["pearson_vstavljanje"] for r in vsi_rezultati]

    x = np.arange(len(ids))

    axs[0].bar(x-0.2, mv, 0.35, label="Vstavljanje",   color="steelblue", alpha=0.8)
    axs[0].bar(x+0.2, mp, 0.35, label="Pospravljanje", color="tomato",    alpha=0.8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([i.replace("patient_","p") for i in ids],
                            rotation=45, ha="right")
    axs[0].set_ylabel("MAE [s]")
    axs[0].set_title("MAE po pacientih")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3, axis="y")

    axs[1].bar(x, rv, 0.5, color="steelblue", alpha=0.8)
    axs[1].axhline(0, color="gray", lw=0.8)
    axs[1].axhline(0.7, color="green", lw=1, linestyle="--",
                    label="r=0.7 (dobro)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([i.replace("patient_","p") for i in ids],
                            rotation=45, ha="right")
    axs[1].set_ylabel("Pearson r")
    axs[1].set_ylim(-1, 1)
    axs[1].set_title("Pearsonova korelacija (vstavljanje)")
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    pot = os.path.join(izhod_mapa, "skupna_evalvacija.png")
    plt.savefig(pot, dpi=130)
    plt.close()
    print(f"  [Evalvacija] Skupni graf: {pot}")


# ===== VSTOPNA TOČKA =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalvacija Y-os algoritma")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pacient", type=str)
    group.add_argument("--vsi",     action="store_true")
    parser.add_argument("--podatki", default=PODATKI_MAPA)
    parser.add_argument("--izhod",   default=IZHOD_MAPA)
    args = parser.parse_args()

    if SRC_MAPA not in sys.path:
        sys.path.insert(0, SRC_MAPA)
    from csv_reader import preberi_csv_pacienta

    if args.vsi:
        evalviraj_vse(args.izhod, args.podatki, verbose=True)
    else:
        json_pot = os.path.join(args.izhod, args.pacient,
                                f"{args.pacient}_rezultati.json")
        csv = preberi_csv_pacienta(args.podatki, args.pacient)
        izhod_pac = os.path.join(args.izhod, args.pacient)
        evalviraj_pacienta(json_pot, csv, izhod_pac, verbose=True)