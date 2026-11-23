#!/usr/bin/env python3

import math
import re
import sys

import numpy as np

SAMPLES = 96          # sampling resolution inside cut cells (96 reproduces the original field to ~1e-4)


def read_simsetup(path="simSetup"):
    text = re.sub(r"/\*.*?\*/", "", open(path).read(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    vals = {}
    for key in ["R", "LtoR", "hWallToR", "nx1", "nx2", "firstToLastAR", "alpha", "RbToR", "LbToR", "xb0ToR"]:
        m = re.search(r"^\s*" + key + r"\s+([-+0-9.eE]+)\s*;", text, re.M)
        if not m:
            sys.exit(f"Could not find a plain number for '{key}' in simSetup")
        vals[key] = float(m.group(1))
    return vals


def mesh(p):
    R, Rf = p["R"], p["R"] * (1.0 - p["hWallToR"])
    nx1, nx2 = int(p["nx1"]), int(p["nx2"])
    L = R * p["LtoR"]
    nz = math.floor(L / (Rf / nx1))                      # same formula as simSetup
    r_core = np.linspace(0.0, Rf, nx1 + 1)
    q = p["firstToLastAR"] ** (1.0 / (nx2 - 1))          # simpleGrading = last/first cell size
    w = q ** np.arange(nx2)
    w = w / w.sum() * (R - Rf)
    r_wall = Rf + np.concatenate([[0.0], np.cumsum(w)])
    z = np.linspace(0.0, L, nz + 1)
    return r_core, r_wall, z, nz, math.radians(p["alpha"])


def block_fractions(redges, z, a, Rb, x0, Lb, n):
    s = (np.arange(n) + 0.5) / n                         # radial (y) samples
    t = (np.arange(n // 2) + 0.5) / (n // 2) * 2.0 - 1.0 # azimuthal (z/y) samples
    u = (np.arange(n) + 0.5) / n                         # axial samples
    out = np.zeros((len(z) - 1, len(redges) - 1))
    for i in range(len(redges) - 1):
        ra, rb = redges[i], redges[i + 1]
        y = ra * math.cos(a) + (rb - ra) * math.cos(a) * s
        Y, T = np.meshgrid(y, t, indexing="ij")
        rho = np.sqrt(Y ** 2 + (T * Y * math.tan(a)) ** 2).ravel()   # distance from the axis
        wgt = (Y / Y.sum()).ravel()                                     # trapezoid: width grows with y
        for k in range(len(z) - 1):
            xa, xb = z[k], z[k + 1]
            # nearest/farthest axial distance from the cell to the bubble's rod section
            dmin = max(0.0, x0 - xb, xa - (x0 + Lb))
            if dmin ** 2 + ra ** 2 > Rb ** 2:
                continue                                 # cell entirely outside
            dmax = max(abs(xa - min(max(xa, x0), x0 + Lb)), abs(xb - min(max(xb, x0), x0 + Lb)))
            if dmax ** 2 + rb ** 2 <= Rb ** 2:
                out[k, i] = 1.0                          # cell entirely inside
                continue
            X = xa + (xb - xa) * u
            dx = X - np.clip(X, x0, x0 + Lb)
            ins = dx[:, None] ** 2 + rho[None, :] ** 2 <= Rb ** 2
            out[k, i] = (ins * wgt[None, :]).sum() / len(u)
    return out


def main():
    p = read_simsetup()
    r_core, r_wall, z, nz, a = mesh(p)
    Rb, x0, Lb = p["R"] * p["RbToR"], p["R"] * p["xb0ToR"], p["R"] * p["LbToR"]
    L = z[-1]
    if x0 - Rb <= 0 or x0 + Lb + Rb >= L:
        sys.exit("The bubble does not fit inside the domain: check xb0ToR, LbToR, RbToR and LtoR")

    f = np.concatenate([block_fractions(r_core, z, a, Rb, x0, Lb, SAMPLES).ravel(),
                        block_fractions(r_wall, z, a, Rb, x0, Lb, SAMPLES).ravel()])

    # check: bubble volume against the ideal shape (same wedge measure)
    area = lambda r: (r[1:] ** 2 - r[:-1] ** 2) * math.sin(a) * math.cos(a)
    dz = L / nz
    V = (f[: len(r_core[:-1]) * nz].reshape(nz, -1) * area(r_core)).sum() * dz \
        + (f[len(r_core[:-1]) * nz:].reshape(nz, -1) * area(r_wall)).sum() * dz
    V_ideal = (math.pi * Rb ** 2 * Lb + 4.0 / 3.0 * math.pi * Rb ** 3) / math.pi * math.sin(a) * math.cos(a)

    template = open("0.orig/alpha.bubble.org").read()
    header = template.split("dimensions")[0].replace("alpha.bubble.org;", "alpha.bubble;")
    boundary = template[template.index("boundaryField"):]
    values = "\n".join("0" if v <= 0.0 else ("1" if v >= 1.0 else repr(float(v))) for v in f)
    with open("0.orig/alpha.bubble", "w") as out:
        out.write(header)
        out.write("dimensions      [ 0 0 0 0 0 0 0 ];\n\n")
        out.write("// Written by make_alpha_bubble.py from simSetup. Rerun it if the mesh or bubble changes.\n\n")
        out.write(f"internalField   nonuniform List<scalar>\n{f.size}\n(\n{values}\n)\n;\n\n")
        out.write(boundary)

    print(f"Mesh: {f.size} cells ({int(p['nx1'])}+{int(p['nx2'])} radial x {nz} axial), domain length {L:g} R")
    print(f"Bubble: x = {x0 - Rb:g} to {x0 + Lb + Rb:g} R, radius {Rb:g} R")
    print(f"Cells cut by the interface: {int(((f > 0) & (f < 1)).sum())}")
    print(f"Bubble volume / reference: {V / V_ideal:.5f}  (expect about 1.0013: the wedge reference uses a flat-sided slice)")
    print("Written: 0.orig/alpha.bubble")


if __name__ == "__main__":
    main()