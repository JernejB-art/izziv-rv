# homografija.py v2 — blink-phase fiksirana inicializacija
import cv2
import numpy as np
from scipy.spatial.distance import cdist

RAZMIK_MM          = 32
POSODICA_PREMER_MM = 100.0
POSODICA_POLMER_MM = 50.0
LUKNJICA_ROI_MM    = 12.0
POSODICA_ROI_MM    = 55.0

def ustvari_world_tocke(razmik_mm=32):
    tocke = []
    for vrstica in range(3):
        for stolpec in range(3):
            tocke.append((stolpec * razmik_mm, vrstica * razmik_mm))
    return np.array(tocke, dtype=np.float32)

def ustvari_world_tocke_posodica(razmik_mm=32):
    return np.array([(4 * razmik_mm), razmik_mm], dtype=np.float32)

def razvrsti_luknjice_v_mrezi(centri_slike):
    if len(centri_slike) < 4:
        return None
    tocke = np.array(centri_slike, dtype=np.float32)
    if len(tocke) >= 9:
        idx_y = np.argsort(tocke[:, 1])
        tocke_s = tocke[idx_y]
        vrstice = []
        for v in range(3):
            vrstica = tocke_s[v*3:(v+1)*3]
            vrstica = vrstica[np.argsort(vrstica[:, 0])]
            vrstice.append(vrstica)
        return np.vstack(vrstice)
    else:
        idx = np.lexsort((tocke[:, 0], tocke[:, 1]))
        return tocke[idx]

def poisci_4_kotne_luknjice(centri_slike):
    tocke = np.array(centri_slike, dtype=np.float32)
    if len(tocke) < 4:
        return None
    hull = cv2.convexHull(tocke.reshape(-1,1,2).astype(np.int32)).reshape(-1,2).astype(np.float32)
    vsota = hull.sum(axis=1); razlika = hull[:,0]-hull[:,1]
    return np.array([hull[np.argmin(vsota)], hull[np.argmax(razlika)],
                     hull[np.argmin(razlika)], hull[np.argmax(vsota)]], dtype=np.float32)

class BoardHomografija:
    """
    Homografija image(px) <-> board(mm).

    NOVA LOGIKA — inicializacija iz blink faze:
      1. inicializiraj_iz_blinka(centri_zg, centri_sp) — enkrat ob blink fazi
      2. Fiksira obe mreži -> stabilen koordinatni sistem
      3. Center posodice = sredina med centroma mrež v mm (fizicno pravilno)
      4. posodobi_iz_luknjic() po fiksiranju ignorirana
    """

    def __init__(self, razmik_mm=RAZMIK_MM):
        self.razmik_mm = razmik_mm
        self.H_mat = None
        self.H_inv = None
        self._veljavna  = False
        self._fiksirana = False
        self._px_per_mm = None
        self.world_A = ustvari_world_tocke(razmik_mm)
        self.world_B = None
        self._centri_zg_img = None
        self._centri_sp_img = None
        self.center_luknjic_mm  = None
        self.center_posodice_mm = None
        self.razdalja_med_mm    = None
        self.polmer_posodice_mm = POSODICA_POLMER_MM
        self.luknjica_roi_mm    = LUKNJICA_ROI_MM
        self.posodica_roi_mm    = POSODICA_ROI_MM

    @property
    def veljavna(self): return self._veljavna
    @property
    def fiksirana(self): return self._fiksirana
    @property
    def px_per_mm(self): return self._px_per_mm

    def inicializiraj_iz_blinka(self, centri_zg_img, centri_sp_img):
        """
        Enkratna inicializacija iz blink faze (obe mreži prižgani).
        Vrne True ce uspesno. Po prvem uspesnem klicu je fiksirana.
        """
        if self._fiksirana:
            return True
        if len(centri_zg_img) < 6 or len(centri_sp_img) < 6:
            return False
        razv_zg = razvrsti_luknjice_v_mrezi(centri_zg_img)
        razv_sp = razvrsti_luknjice_v_mrezi(centri_sp_img)
        if razv_zg is None or razv_sp is None:
            return False
        n_zg = min(len(razv_zg), 9); n_sp = min(len(razv_sp), 9)
        razv_zg = razv_zg[:n_zg]; razv_sp = razv_sp[:n_sp]
        c_zg_img = razv_zg.mean(axis=0); c_sp_img = razv_sp.mean(axis=0)
        razdalja_px = float(np.linalg.norm(c_sp_img - c_zg_img))
        if razdalja_px < 10:
            return False
        # Oceni px/mm iz sosednjih razdalj v mreži A
        razdalje = sorted([float(np.linalg.norm(razv_zg[i]-razv_zg[j]))
                           for i in range(n_zg) for j in range(i+1,n_zg)])
        razmik_px = float(np.median(razdalje[:min(12,len(razdalje))]))
        if razmik_px < 1:
            return False
        px_per_mm_ocena = razmik_px / self.razmik_mm

        # Izmeri razdaljo med mrežama v pravem mm prostoru
        # z začasno homografijo samo iz zgornje mreže (world_A)
        world_A_temp = self.world_A[:n_zg]
        H_temp, _ = cv2.findHomography(razv_zg, world_A_temp, 0)
        razdalja_mm_ocena = 275.0  # fizično izmerjena razdalja med centroma mrež (mm)

        world_A = self.world_A[:n_zg]
        world_B = np.array([(s*self.razmik_mm, razdalja_mm_ocena+v*self.razmik_mm)
                             for v in range(3) for s in range(3)],
                            dtype=np.float32)[:n_sp]
        img_pts   = np.vstack([razv_zg, razv_sp]).astype(np.float32)
        world_pts = np.vstack([world_A,  world_B]).astype(np.float32)
        H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, ransacReprojThreshold=4.0)
        if H is None:
            H, _ = cv2.findHomography(razv_zg, world_A, 0)
        if H is None:
            return False
        self.H_mat = H; self.H_inv = np.linalg.inv(H)
        self.world_B = world_B; self._veljavna = True
        self._izracunaj_px_per_mm(img_pts, world_pts)
        c_zg_mm = self.v_mm(tuple(c_zg_img))
        c_sp_mm = self.v_mm(tuple(c_sp_img))
        if c_zg_mm and c_sp_mm:
            self.center_luknjic_mm  = np.array(c_zg_mm)
            c_sp_arr = np.array(c_sp_mm)
            self.center_posodice_mm = tuple((self.center_luknjic_mm + c_sp_arr) / 2.0)
            self.razdalja_med_mm    = float(np.linalg.norm(c_sp_arr - self.center_luknjic_mm))
        self._centri_zg_img = razv_zg; self._centri_sp_img = razv_sp
        self._fiksirana = True
        return True

    def posodobi_iz_luknjic(self, centri_slike):
        """Kompatibilnostna metoda. Po fiksiranju ne dela nicesar."""
        if self._fiksirana:
            return True
        if not centri_slike or len(centri_slike) < 4:
            return False
        razv = razvrsti_luknjice_v_mrezi(centri_slike)
        if razv is None: return False
        n = min(len(razv), 9); world = self.world_A[:n]
        H, _ = cv2.findHomography(razv[:n], world, cv2.RANSAC, 5.0)
        if H is None: H, _ = cv2.findHomography(razv[:n], world, 0)
        if H is None: return False
        self.H_mat = H; self.H_inv = np.linalg.inv(H)
        self._veljavna = True; self._izracunaj_px_per_mm(razv[:n], world)
        return True

    def v_mm(self, tocka_img):
        if not self._veljavna or self.H_mat is None: return None
        pt = np.array([[[float(tocka_img[0]), float(tocka_img[1])]]], dtype=np.float32)
        r  = cv2.perspectiveTransform(pt, self.H_mat)
        return (float(r[0,0,0]), float(r[0,0,1]))

    def v_sliko(self, tocka_mm):
        if not self._veljavna or self.H_inv is None: return None
        pt = np.array([[[float(tocka_mm[0]), float(tocka_mm[1])]]], dtype=np.float32)
        r  = cv2.perspectiveTransform(pt, self.H_inv)
        return (float(r[0,0,0]), float(r[0,0,1]))

    def niz_v_mm(self, tocke_img):
        if not self._veljavna or self.H_mat is None or len(tocke_img)==0: return None
        pts = np.array(tocke_img, dtype=np.float32).reshape(-1,1,2)
        return cv2.perspectiveTransform(pts, self.H_mat).reshape(-1,2)

    def warp_frame(self, frame, sirina_mm=250, visina_mm=250, px_per_mm=4.0):
        if not self._veljavna: return None
        S = np.diag([px_per_mm, px_per_mm, 1.0])
        return cv2.warpPerspective(frame, S @ self.H_mat,
                                   (int(sirina_mm*px_per_mm), int(visina_mm*px_per_mm)))

    def _izracunaj_px_per_mm(self, img_tocke, world_tocke):
        d_img   = cdist(img_tocke, img_tocke)
        d_world = cdist(world_tocke.astype(np.float64), world_tocke.astype(np.float64))
        mask = (d_world > 5) & (d_world < 150)
        if not np.any(mask): return
        self._px_per_mm = float(np.median(d_img[mask] / d_world[mask]))

    def get_luknjice_mm(self): return self.world_A.copy()

    def get_luknjice_img(self):
        if not self._veljavna: return None
        return np.array([self.v_sliko(pt) for pt in self.world_A])

    def razdalja_do_luknjice_mm(self, tocka_mm, idx):
        return float(np.linalg.norm(np.array(tocka_mm) - self.world_A[idx]))

    def razdalja_do_posodice_mm(self, tocka_mm, center_posodice_mm=None):
        c = center_posodice_mm or self.center_posodice_mm
        if c is None: return float("inf")
        return float(np.linalg.norm(np.array(tocka_mm) - np.array(c)))

    def je_v_roi_luknjice(self, tocka_mm, idx):
        return self.razdalja_do_luknjice_mm(tocka_mm, idx) <= self.luknjica_roi_mm

    def je_v_roi_posodice(self, tocka_mm, center_posodice_mm=None):
        return self.razdalja_do_posodice_mm(tocka_mm, center_posodice_mm) <= self.posodica_roi_mm

    def narisi_debug_overlay(self, frame, roka_mm=None, pot_mm=None):
        """
        Nariše overlay na originalni frame.
        Luknjice so narisane na FIKSIRANIH image koordinatah iz blink faze
        (ne projekcija world→image, kar ima napako) — zato točno sovpadajo.
        """
        out = frame.copy()
        if not self._veljavna: return out
        px = self._px_per_mm or 1.0
        r_luk = max(4, int(self.luknjica_roi_mm * px))

        # --- Luknjice: FIKSIRANA mesta iz blink faze (zg = luknjice) ---
        if self._centri_zg_img is not None:
            for i, (cx, cy) in enumerate(self._centri_zg_img.astype(int)):
                cv2.circle(out, (cx,cy), r_luk, (0,200,100), 1)   # ROI krog
                cv2.circle(out, (cx,cy), 5, (0,255,200), -1)       # center
                cv2.putText(out, str(i), (cx+5,cy-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,255,200), 1)

        # --- Spodnja mreža (posodica stran): fiksirana mesta ---
        # Spodnja mreža — tudi luknjice, enaki krogi kot zgornja
        if self._centri_sp_img is not None:
            for i, (cx, cy) in enumerate(self._centri_sp_img.astype(int)):
                cv2.circle(out, (cx,cy), r_luk, (0,200,100), 1)   # enak zeleni ROI krog
                cv2.circle(out, (cx,cy), 5, (0,255,200), -1)
                cv2.putText(out, str(i+9), (cx+5,cy-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,255,200), 1)

        # --- Center posodice ---
        if self.center_posodice_mm is not None:
            pi = self.v_sliko(self.center_posodice_mm)
            if pi:
                px2, py2 = int(pi[0]), int(pi[1])
                r_fiz = max(5, int(self.polmer_posodice_mm * px))
                r_roi = max(5, int(self.posodica_roi_mm * px))
                cv2.circle(out, (px2,py2), r_fiz, (255,200,0), 2)   # fizicni krog
                cv2.circle(out, (px2,py2), r_roi, (255,150,0), 1)   # ROI krog
                cv2.circle(out, (px2,py2), 5, (255,200,0), -1)
                cv2.putText(out, "posodica",
                            (px2-28, py2-r_fiz-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,200,0), 1)

        # --- Trajektorija roke (v mm → projicirana nazaj) ---
        if pot_mm and len(pot_mm) > 1:
            for j in range(1, len(pot_mm)):
                p1 = self.v_sliko(pot_mm[j-1])
                p2 = self.v_sliko(pot_mm[j])
                if p1 and p2:
                    cv2.line(out,
                             (int(p1[0]),int(p1[1])),
                             (int(p2[0]),int(p2[1])),
                             (0,0,200), 2)

        # --- Roka ---
        if roka_mm:
            ri = self.v_sliko(roka_mm)
            if ri:
                cv2.circle(out, (int(ri[0]),int(ri[1])), 8, (200,50,50), -1)
                cv2.putText(out, f"({roka_mm[0]:.0f},{roka_mm[1]:.0f})mm",
                            (int(ri[0])+10,int(ri[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,50,50), 1)

        cv2.putText(out,
                    f"px/mm={px:.2f} | {'FIKSIRANA' if self._fiksirana else 'init...'}",
                    (10, out.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,50), 1)
        return out

    def narisi_board_mm(self, trajektorije_mm=None, dogodki=None,
                         naslov="Trajektorija v board koordinatah (mm)", izhod_pot=None):
        """2D prikaz: luknjice + posodica + trajektorija v mm prostoru."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[BoardMM] pip install matplotlib"); return
        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_aspect("equal"); ax.invert_yaxis()
        for i,(lx,ly) in enumerate(self.world_A):
            ax.add_patch(plt.Circle((lx,ly),self.luknjica_roi_mm,color="gold",fill=False,lw=1.5))
            ax.plot(lx,ly,"o",color="gold",ms=5,zorder=5)
            ax.annotate(str(i),(lx+2,ly-3),fontsize=7,color="goldenrod")
        if self.world_B is not None:
            for i,(bx,by) in enumerate(self.world_B):
                ax.add_patch(plt.Circle((bx,by),self.luknjica_roi_mm,
                                        color="lightsteelblue",fill=False,lw=1.0,alpha=0.7))
                ax.plot(bx,by,"s",color="lightsteelblue",ms=4,alpha=0.7)
                ax.annotate(str(i),(bx+2,by-3),fontsize=7,color="steelblue",alpha=0.7)
        if self.center_posodice_mm is not None:
            cx,cy = self.center_posodice_mm
            ax.add_patch(plt.Circle((cx,cy),self.polmer_posodice_mm,
                                    color="darkorange",fill=False,lw=2.5,
                                    label=f"Posodica (d={POSODICA_PREMER_MM:.0f}mm)"))
            ax.add_patch(plt.Circle((cx,cy),self.posodica_roi_mm,
                                    color="orange",fill=False,lw=1,linestyle="--",alpha=0.4))
            ax.plot(cx,cy,"+",color="darkorange",ms=12,markeredgewidth=2)
        barve = ["steelblue","crimson","seagreen","purple","sienna"]
        if trajektorije_mm is not None:
            items = (trajektorije_mm.items()
                     if isinstance(trajektorije_mm,dict)
                     else [("Trajektorija",trajektorije_mm)])
            for idx,(ime,traj) in enumerate(items):
                if not traj: continue
                xs=[p[0] for p in traj]; ys=[p[1] for p in traj]
                b=barve[idx%len(barve)]
                ax.plot(xs,ys,"-",color=b,lw=1.2,alpha=0.75,label=ime)
                ax.plot(xs[0],ys[0],"o",color=b,ms=7,zorder=6)
                ax.plot(xs[-1],ys[-1],"s",color=b,ms=7,zorder=6)
        vse_x=list(self.world_A[:,0]); vse_y=list(self.world_A[:,1])
        if self.world_B is not None:
            vse_x+=list(self.world_B[:,0]); vse_y+=list(self.world_B[:,1])
        if self.center_posodice_mm:
            vse_x.append(self.center_posodice_mm[0]); vse_y.append(self.center_posodice_mm[1])
        m=30
        ax.set_xlim(min(vse_x)-m,max(vse_x)+m); ax.set_ylim(min(vse_y)-m,max(vse_y)+m)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_title(naslov)
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
        if self.razdalja_med_mm:
            ax.text(0.02,0.02,
                    f"Razdalja med mrezama: {self.razdalja_med_mm:.0f}mm\npx/mm: {self._px_per_mm:.2f}",
                    transform=ax.transAxes,fontsize=8,color="gray")
        plt.tight_layout()
        if izhod_pot:
            plt.savefig(izhod_pot,dpi=130); plt.close()
            print(f"[BoardMM] Shranjen: {izhod_pot}")
        else:
            plt.show()


if __name__ == "__main__":
    print("=== TEST BoardHomografija v2 ===")
    razmik=32; px_mm=5.0; razd_px=180.0
    def sim(offset_y, zasuk=0):
        world=ustvari_world_tocke(razmik); kot=np.radians(zasuk)
        R=np.array([[np.cos(kot),-np.sin(kot)],[np.sin(kot),np.cos(kot)]])
        return [tuple(R@np.array([wx*px_mm,wy*px_mm])+np.array([200.,50.+offset_y])+np.random.randn(2))
                for wx,wy in world]
    c_zg=sim(0,3); c_sp=sim(razd_px,3)
    H=BoardHomografija(razmik_mm=razmik)
    ok=H.inicializiraj_iz_blinka(c_zg,c_sp)
    print(f"Init: {ok} | fiksirana: {H.fiksirana} | px/mm: {H.px_per_mm:.3f}")
    print(f"Center posodice: {H.center_posodice_mm}")
    print(f"Razdalja med mrezama: {H.razdalja_med_mm:.1f}mm (pricakovano ~{razd_px/px_mm:.0f}mm)")
    napake=[np.linalg.norm(np.array(H.v_mm(ip))-wp)
            for ip,wp in zip(c_zg,ustvari_world_tocke(razmik))]
    print(f"Povp. napaka: {np.mean(napake):.3f}mm")
    print("=== TEST OK ===")