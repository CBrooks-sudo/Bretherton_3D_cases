#!/usr/bin/env python3
"""
Verification of the v2 ADAPTIVE moving-frame run (Ca = 0.002, 2D axisymmetric).

Run from the v2 case folder after the run has finished:
    python3 frame_check.py
or, if your lab-frame case is somewhere else:
    python3 frame_check.py /path/to/lab-frame/case/bubbleInfo

Reads
  frameInfo   (written by the frameController function object every 10 steps)
              columns: t, Uf used that step, a for next step, xG (frame), v, e, lab-frame centroid X
  bubbleInfo  (this case; used for the bubble-volume check)
  postProcessing/freeSurface/<latest>/*.raw   (film thickness)
  the LAB-FRAME run's bubbleInfo (for the trajectory test)

Checks
  1. Controller convergence: Uf settles, v -> 0, a -> 0, e stays small, no growing oscillation
  2. Steady lab-frame bubble speed Ub = Uf + v, against the lab-frame run and v1
  3. Film thickness, against the lab-frame run
  4. Transient: lab-frame trajectory X(t) - X(0) against the lab-frame run's xG(t) - xG(0)
     (this is the test of the frame-acceleration force)
Writes frame_controller.png, frame_trajectory.png, frame_speed.png
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 1.0
LAB_UB, LAB_DELTA = 1.04215, 0.02065     # lab-frame run, Ca = 0.002
V1_UB, V1_DELTA = 1.04422, 0.02063       # v1 constant moving frame, same 20R domain
STEADY_FRACTION = 0.2                     # last 20 % of the run is treated as steady
DEFAULT_LAB = os.path.expanduser(
    "~/OpenFOAM/egycb7-v2106/run/bretherton/Ca0p002_2Daxi_plicRDF_fixed/Ca0p002_2Daxi_plicRDF/bubbleInfo")


def load(path, ncol):
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    if d.shape[1] != ncol:
        sys.exit(f"{path}: expected {ncol} columns, found {d.shape[1]}")
    d = d[np.argsort(d[:, 0], kind="stable")]
    _, keep = np.unique(d[:, 0], return_index=True)     # drop repeats (e.g. a restart appended rows)
    return d[np.sort(keep)]


def film_thickness():
    times = []
    for d in glob.glob(os.path.join("postProcessing", "freeSurface", "*")):
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
    p = np.loadtxt(raws[0], comments="#", ndmin=2)
    x, r = p[:, 0], np.hypot(p[:, 1], p[:, 2])
    mid = np.abs(x - 0.5 * (x.min() + x.max())) < 0.1 * (x.max() - x.min())
    return t, R - r[mid].max()


def main():
    if not os.path.isfile("frameInfo"):
        sys.exit("No frameInfo here: run this from the v2 case folder after the run.")
    f = load("frameInfo", 7)
    t, Uf, a, xG, v, e, X = f.T
    n0 = int(len(t) * (1.0 - STEADY_FRACTION))
    print(f"frameInfo: {len(t)} rows, t = {t[0]:.4g} to {t[-1]:.4g}")

    # 1. controller convergence
    print("\n1. CONTROLLER")
    print(f"   Uf: start {Uf[0]:.5f}, max {Uf.max():.5f} at t = {t[np.argmax(Uf)]:.2f}, end {Uf[-1]:.5f}")
    print(f"   end of run:  v = {v[-1]:+.2e}   a = {a[-1]:+.2e}   e = {e[-1]:+.4f} R")
    print(f"   largest |e| during the run: {np.abs(e).max():.4f} R at t = {t[np.argmax(np.abs(e))]:.2f}")
    late = e[n0:]
    crossings = int(np.sum(np.diff(np.sign(late - late.mean())) != 0))
    print(f"   last {int(STEADY_FRACTION*100)} %: Uf varies by {Uf[n0:].max()-Uf[n0:].min():.2e}, "
          f"e varies by {late.max()-late.min():.2e} R ({crossings} mean-crossings)")

    # 2. steady lab-frame speed
    Ub = Uf + v
    Ub_steady = Ub[n0:].mean()
    print("\n2. STEADY BUBBLE SPEED (lab frame, Ub = Uf + v, last "
          f"{int(STEADY_FRACTION*100)} %)")
    print(f"   Ub = {Ub_steady:.5f}   vs lab-frame run {LAB_UB:.5f} ({100*(Ub_steady-LAB_UB)/LAB_UB:+.3f} %)"
          f"   vs v1 {V1_UB:.5f} ({100*(Ub_steady-V1_UB)/V1_UB:+.3f} %)")

    # 3. film thickness
    ts, delta = film_thickness()
    print("\n3. FILM THICKNESS")
    if delta is None:
        print("   no freeSurface output found")
    else:
        print(f"   delta = {delta:.5f} R at t = {ts:g}   vs lab-frame run {LAB_DELTA:.5f} "
              f"({100*(delta-LAB_DELTA)/LAB_DELTA:+.2f} %)   vs v1 {V1_DELTA:.5f} "
              f"({100*(delta-V1_DELTA)/V1_DELTA:+.2f} %)")

    if os.path.isfile("bubbleInfo"):
        b = load("bubbleInfo", 5)
        print(f"   bubble volume change: {100*(b[-1,1]-b[0,1])/b[0,1]:+.4f} %")

    # 4. lab-frame trajectory against the lab-frame run
    lab_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LAB
    print("\n4. TRANSIENT: lab-frame trajectory vs lab-frame run")
    have_lab = os.path.isfile(lab_path)
    if not have_lab:
        print(f"   lab-frame bubbleInfo not found at {lab_path}\n   (pass its path as an argument)")
    else:
        L = load(lab_path, 5)
        tl, xl = L[:, 0], L[:, 2]
        dX = X - X[0]
        dxl = np.interp(t, tl, xl) - np.interp(t[0], tl, xl)
        diff = dX - dxl
        print(f"   displacement by t = {t[-1]:g}: v2 {dX[-1]:.4f} R, lab-frame run {dxl[-1]:.4f} R")
        for tt in (0.5, 1, 2, 4, 6, 10):
            if t[0] <= tt <= t[-1]:
                print(f"   t = {tt:>4}: difference {np.interp(tt, t, diff):+.4f} R")
        print(f"   largest |difference| over the run: {np.abs(diff).max():.4f} R at t = {t[np.argmax(np.abs(diff))]:.2f}")

    # plots
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for axi, y, lab in zip(ax, (Uf, v, e, a), ("Uf  [Ul]", "v  [Ul]", "e  [R]", "a  [Ul^2/R]")):
        axi.plot(t, y); axi.set_ylabel(lab); axi.grid(True)
    ax[0].axhline(LAB_UB, color="k", lw=0.8, ls="--", label="lab-frame Ub")
    ax[0].legend()
    ax[-1].set_xlabel("t  [R/Ul]")
    ax[0].set_title("Adaptive frame controller")
    fig.tight_layout(); fig.savefig("frame_controller.png", dpi=150)

    fig, ax = plt.subplots()
    ax.plot(t, Ub, label="v2: Uf + v")
    ax.axhline(LAB_UB, color="k", lw=0.8, ls="--", label="lab-frame run (steady)")
    ax.set_xlabel("t  [R/Ul]"); ax.set_ylabel("Ub  [Ul]"); ax.grid(True); ax.legend()
    ax.set_title("Lab-frame bubble speed")
    fig.savefig("frame_speed.png", dpi=150)

    if have_lab:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(t, dX, label="v2: X(t) - X(0)")
        ax[0].plot(t, dxl, "--", label="lab-frame run: xG(t) - xG(0)")
        ax[0].set_ylabel("displacement  [R]"); ax[0].grid(True); ax[0].legend()
        ax[1].plot(t, diff); ax[1].set_ylabel("difference  [R]"); ax[1].grid(True)
        ax[1].set_xlabel("t  [R/Ul]")
        ax[0].set_title("Lab-frame trajectory: adaptive frame vs lab-frame run")
        fig.tight_layout(); fig.savefig("frame_trajectory.png", dpi=150)

    print("\nPlots written: frame_controller.png, frame_speed.png" + (", frame_trajectory.png" if have_lab else ""))


if __name__ == "__main__":
    main()