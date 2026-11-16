#!/usr/bin/env python3
"""
Post-processing for the Ca0p002_2Daxi_plicRDF Taylor-bubble case.

Run it from the case folder (the one containing bubbleInfo):
    python3 postprocess_bubble.py

Reads:
  bubbleInfo                         written every 10 steps by the 'bubbleVol' function object
                                     columns: time, bubble volume (wedge), xG, yG, zG
  postProcessing/freeSurface/<t>/    alpha = 0.5 iso-surface written by the 'freeSurface' function object

Writes (in the case folder):
  bubble_speed.png, bubble_volume.png, film_profile.png
and prints the steady bubble speed, film thickness and the theoretical estimates.

Everything is in the case's dimensionless units: lengths / R, velocities / Ul, time / (R/Ul).
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # write image files, no window needed
import matplotlib.pyplot as plt

# Values from simSetup (change these if you change simSetup)
R, Ul, Ca_in, Re = 1.0, 1.0, 0.002, 1.0
Uf = 1.04215
LAB_UB, LAB_DELTA = 1.04215, 0.02065
mul = 2.0 * Ul * R / Re                    # liquid dynamic viscosity (rhol = 1)
sigma = mul * Ul / Ca_in                   # surface tension

STEADY_FRACTION = 0.2                      # use the last 20 % of bubbleInfo for the steady speed


def delta_bretherton(Ca):
    return 1.34 * Ca ** (2.0 / 3.0)


def delta_aq(Ca):
    return 1.34 * Ca ** (2.0 / 3.0) / (1.0 + 3.35 * Ca ** (2.0 / 3.0))


def ub_theory(delta_fn):
    """Bubble speed predicted by a theory, with Ca_b = Ca*Ub solved consistently."""
    Ub = 1.0
    for _ in range(50):
        Ub = Ul / (1.0 - delta_fn(Ca_in * Ub) / R) ** 2
    return Ub

def read_bubble_info(path="bubbleInfo"):
    if not os.path.isfile(path):
        sys.exit("No bubbleInfo file here. Run this from the case folder after the solver has started.")
    data = np.loadtxt(path, delimiter=",")
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, keep = np.unique(data[:, 0], return_index=True)   # drop repeated times (e.g. after a restart)
    return data[np.sort(keep)]


def bubble_speed(t, x, window=50):
    """Speed from a straight-line fit of xG(t) over a sliding window of samples (reduces noise)."""
    ts, us = [], []
    for i in range(0, len(t) - window, max(1, window // 5)):
        sl = slice(i, i + window)
        us.append(np.polyfit(t[sl], x[sl], 1)[0])
        ts.append(t[sl].mean())
    return np.array(ts), np.array(us)


def latest_free_surface():
    dirs = glob.glob(os.path.join("postProcessing", "freeSurface", "*"))
    times = []
    for d in dirs:
        try:
            times.append((float(os.path.basename(d)), d))
        except ValueError:
            pass
    if not times:
        return None, None
    t, d = max(times)
    raws = glob.glob(os.path.join(d, "*.raw"))
    if not raws:
        return t, None
    return t, np.loadtxt(raws[0], comments="#", ndmin=2)


def main():
    d = read_bubble_info()
    t, V, xG = d[:, 0], d[:, 1], d[:, 2]
    print(f"bubbleInfo: {len(t)} samples, t = {t[0]:.4g} to {t[-1]:.4g}")

    # 1. Bubble volume conservation (gas is treated as incompressible, so V should be constant)
    drift = 100.0 * (V[-1] - V[0]) / V[0]
    print(f"Bubble volume change since start: {drift:+.3f} %")

    # 2. Bubble speed
    window = max(5, min(50, len(t) // 10))
    ts, Ub = bubble_speed(t, xG, window)
    Ub = Ub + Uf
    n0 = int(len(t) * (1.0 - STEADY_FRACTION))
    drift = np.polyfit(t[n0:], xG[n0:], 1)[0]
    Ub_last = Uf + drift
    print(f"Bubble drift relative to the frame: {drift:+.5f} Ul;  centroid now at x = {xG[-1]:.3f} R")
    print(f"Lab-frame Ub = {Ub_last:.5f}  vs lab-frame run {LAB_UB:.5f}  ->  {100*(Ub_last-LAB_UB)/LAB_UB:+.3f} %")
    Ca_b = mul * Ub_last / sigma
    print(f"Bubble speed over last {int(STEADY_FRACTION*100)} % of the data: Ub = {Ub_last:.5f} Ul")
    print(f"Bubble capillary number Ca_b = mu_l*Ub/sigma = {Ca_b:.6f}")

    # Thin-film theory (check the papers before quoting these):
    #   Bretherton:              delta/R = 1.34 Ca^(2/3)
    #   Aussillous & Quere fit:  delta/R = 1.34 Ca^(2/3) / (1 + 3.35 Ca^(2/3))
    c23 = Ca_b ** (2.0 / 3.0)
    d_br = 1.34 * c23 * R
    d_aq = 1.34 * c23 / (1.0 + 3.35 * c23) * R
    print(f"Theory film thickness: Bretherton {d_br:.5f} R, Aussillous-Quere {d_aq:.5f} R")
    print(f"Speed implied by a stagnant film of the A-Q thickness: Ub = Ul/(1-delta/R)^2 = {Ul/(1-d_aq/R)**2:.5f} Ul")

    fig, ax = plt.subplots()
    ax.plot(ts, Ub)
    ax.set_xlabel("t  [R/Ul]"); ax.set_ylabel("Ub  [Ul]"); ax.grid(True)
    ax.set_title("Bubble speed (centroid)")
    ax.axhline(ub_theory(delta_aq), color="tab:green", ls="--", label="Aussillous-Quere")
    ax.axhline(ub_theory(delta_bretherton), color="tab:red", ls=":", label="Bretherton")
    ax.legend()
    fig.savefig("bubble_speed.png", dpi=150)

    fig, ax = plt.subplots()
    ax.plot(t, 100.0 * (V - V[0]) / V[0])
    ax.set_xlabel("t  [R/Ul]"); ax.set_ylabel("volume change  [%]"); ax.grid(True)
    ax.set_title("Bubble volume conservation")
    fig.savefig("bubble_volume.png", dpi=150)

    # 3. Film thickness from the latest alpha = 0.5 surface
    ts_surf, pts = latest_free_surface()
    if pts is None:
        print("No freeSurface output yet (it is written every 2000 steps).")
    else:
        x = pts[:, 0]
        r = np.hypot(pts[:, 1], pts[:, 2])
        order = np.argsort(x)
        x, r = x[order], r[order]
        xmid = 0.5 * (x.min() + x.max())
        mid = np.abs(x - xmid) < 0.1 * (x.max() - x.min())   # central 20 % of the bubble length
        delta = R - r[mid].max()
        print(f"Interface at t = {ts_surf:g}: bubble from x = {x.min():.3f} to {x.max():.3f}, "
              f"film thickness at mid-bubble delta = {delta:.5f} R")
        print(f"  vs lab-frame run {LAB_DELTA:.5f} R  ->  {100*(delta-LAB_DELTA)/LAB_DELTA:+.2f} %")

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(x, r, ".", ms=1)
        ax.axhline(R, color="k", lw=0.8)
        ax.set_xlabel("x  [R]"); ax.set_ylabel("r  [R]"); ax.set_ylim(0, 1.05 * R)
        ax.set_title(f"Bubble profile (alpha = 0.5) at t = {ts_surf:g}")
        ax.axhline(R - delta_aq(Ca_b) * R, color="tab:green", ls="--", lw=0.8, label="Aussillous-Quere")
        ax.axhline(R - delta_bretherton(Ca_b) * R, color="tab:red", ls=":", lw=0.8, label="Bretherton")
        ax.legend(loc="lower center")
        fig.tight_layout()
        fig.savefig("film_profile.png", dpi=150)

        Ca_range = np.logspace(-4, -1, 200)
        fig, ax = plt.subplots()
        ax.loglog(Ca_range, delta_aq(Ca_range), color="tab:green", ls="--", label="Aussillous-Quere")
        ax.loglog(Ca_range, delta_bretherton(Ca_range), color="tab:red", ls=":", label="Bretherton")
        ax.loglog(Ca_b, delta / R, "ko", label="Simulation")
        ax.set_xlabel("Ca_b"); ax.set_ylabel("delta / R"); ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_title("Film thickness against theory")
        fig.savefig("film_vs_theory.png", dpi=150)

    print("Plots written: bubble_speed.png, bubble_volume.png" + (", film_profile.png" if pts is not None else ""))


if __name__ == "__main__":
    main()
