"""
IN BLOOM - an ASCII rose that grows and opens.

An original, from-scratch implementation of the ideas listed in this folder's
README (3D parametric surfaces -> perspective projection -> z-buffer on a grid
of characters -> brightness shading + outline detection). It is NOT the
artist's paid source code.

Run
    python rose.py                    open the animated window
    python rose.py --shot 16 out.png  render the frame at t=16s to a PNG (no window)

Keys
    SPACE pause / resume    R restart    S save screenshot    ESC quit

Requires
    pip install numpy pygame-ce      (plain "pygame" works too)
"""
import argparse
import math
import os
import random
import sys

import numpy as np

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
W, H = 1080, 960
FPS = 30
FONT_SIZE = 14
CAM_D = 7.0            # camera distance from the origin
FOCAL = 1150.0         # perspective scale in pixels
SCREEN_CY = 430        # pixel row where world y=0 (at camera depth) lands
CAM_TILT = 0.28        # look slightly down onto the flower (radians)
EDGE_T = 0.32          # depth jump that counts as an outline
RAMPS = [".:-=+*tfjzrcxvunoeaQ0&8B%$#@"]   # character set(s), dark -> bright
BG = (6, 6, 9)

GROUND_Y = -2.15
STEM_TOP = 0.35
HEAD_SCALE = 1.45

# timeline (seconds)
T_STEM = 4.0           # stem drawn up by the travelling light
T_BUD0, T_BUD1 = 4.0, 6.5
T_BLOOM0, T_BLOOM1 = 6.5, 14.0


# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def surface_normals(P):
    """Normals of a (nu, nv, 3) grid surface from its two parameter directions."""
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    return normalize(np.cross(du, dv))


# ----------------------------------------------------------------------------
# point batch: everything drawn in a frame is a cloud of points with normals
# ----------------------------------------------------------------------------
class Batch:
    """gain scales a part's brightness, ramp picks which character set it uses."""
    def __init__(self):
        self.parts = []

    def add(self, P, N, color, spec=0.3, mask=None, gain=1.0, ramp=0):
        P = np.asarray(P, float).reshape(-1, 3)
        N = np.asarray(N, float).reshape(-1, 3)
        color = np.asarray(color, float)
        C = np.broadcast_to(color.reshape(-1, 3) if color.ndim > 1 else color, P.shape).reshape(-1, 3)
        S = np.broadcast_to(np.asarray(spec, float).reshape(-1), (len(P),))
        G = np.broadcast_to(np.asarray(gain, float).reshape(-1), (len(P),))
        K = np.full(len(P), int(ramp))
        if mask is not None:
            m = np.asarray(mask, bool).reshape(-1)
            P, N, C, S, G, K = P[m], N[m], C[m], S[m], G[m], K[m]
        if len(P):
            self.parts.append((P, N, C, S, G, K))

    def arrays(self):
        if not self.parts:
            z, e = np.zeros((0, 3)), np.zeros(0)
            return z, z, z, e, e, e.astype(int)
        return tuple(np.concatenate(a) for a in zip(*self.parts))


def shade(P, N, C, S, lights, ambient=0.24):
    """Two-sided diffuse + specular lighting in camera space.
    Returns brightness (0..1+), an RGB colour and the specular term per point."""
    V = normalize(np.array([0.0, 0.0, -CAM_D]) - P)
    facing = np.sum(N * V, axis=1)
    back = facing < 0
    N = np.where(back[:, None], -N, N)
    diff = np.zeros(len(P))
    spec = np.zeros(len(P))
    for Ldir, strength, power in lights:
        L = normalize(np.asarray(Ldir, float))[None, :]
        d = np.clip(np.sum(N * L, axis=1), 0, 1)
        Hh = normalize(L + V)
        sp = np.clip(np.sum(N * Hh, axis=1), 0, 1) ** power
        diff += strength * d
        spec += strength * sp
    diff = np.where(back, diff * 0.55, diff)
    rim = (1.0 - np.abs(facing)) ** 3 * 0.18
    bright = ambient + 0.72 * diff + 0.8 * S * spec + rim
    rgb = C * (0.42 + 0.85 * diff + rim)[:, None] + (255 * S * spec)[:, None]
    return np.clip(bright, 0, 1.2), np.clip(rgb, 0, 255), spec


# ----------------------------------------------------------------------------
# ASCII renderer
# ----------------------------------------------------------------------------
def pick_font(pygame, size, bold=False):
    for name in ("consolas", "menlo", "dejavusansmono", "couriernew", "monospace"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size + 4)


class AsciiRenderer:
    def __init__(self, pygame):
        self.pg = pygame
        self.font = pick_font(pygame, FONT_SIZE)
        self.hud_font = pick_font(pygame, FONT_SIZE + 2)
        self.cw = self.font.size("M")[0]
        self.ch = self.font.get_height()
        self.cols, self.rows = W // self.cw, H // self.ch
        self.cache = {}
        self.layer = pygame.Surface((W, H))
        self.depth = None
        self.chars = None
        self.colors = None
        self.letters = 0

    # glyph cache -----------------------------------------------------------
    def glyph(self, ch, rgb, font=None):
        key = (ch, rgb, font is not None)
        g = self.cache.get(key)
        if g is None:
            if len(self.cache) > 30000:
                self.cache.clear()
            g = (font or self.font).render(ch, True, rgb)
            self.cache[key] = g
        return g

    @staticmethod
    def q(rgb):
        """Quantise a colour so the glyph cache stays small."""
        return tuple(int(min(255, max(0, round(c / 12.0) * 12))) for c in rgb)

    def project(self, P):
        zc = P[:, 2] + CAM_D
        zc = np.maximum(zc, 1e-3)
        sx = W / 2 + FOCAL * P[:, 0] / zc
        sy = SCREEN_CY - FOCAL * P[:, 1] / zc
        return sx, sy, zc

    # main rasteriser -------------------------------------------------------
    def rasterize(self, P, bright, rgb, ramp_id=None):
        rows, cols = self.rows, self.cols
        sx, sy, zc = self.project(P)
        c = np.floor(sx / self.cw).astype(int)
        r = np.floor(sy / self.ch).astype(int)
        ok = (c >= 0) & (c < cols) & (r >= 0) & (r < rows) & (P[:, 2] + CAM_D > 0.2)
        idx = (r * cols + c)[ok]
        zc, bright, rgb = zc[ok], bright[ok], rgb[ok]
        kid = np.zeros(len(idx), int) if ramp_id is None else np.asarray(ramp_id)[ok]

        depth = np.full(rows * cols, np.inf)
        B = np.zeros(rows * cols)
        RGB = np.zeros((rows * cols, 3))
        Kg = np.zeros(rows * cols, int)
        if len(idx):
            order = np.lexsort((zc, idx))            # nearest point first per cell
            ids = idx[order]
            first = np.ones(len(ids), bool)
            first[1:] = ids[1:] != ids[:-1]
            sel = order[first]
            depth[idx[sel]] = zc[sel]
            B[idx[sel]] = bright[sel]
            RGB[idx[sel]] = rgb[sel]
            Kg[idx[sel]] = kid[sel]

        D = depth.reshape(rows, cols)
        B = B.reshape(rows, cols)
        Kg = Kg.reshape(rows, cols)
        RGB = RGB.reshape(rows, cols, 3)
        occ = np.isfinite(D)

        # outline detection on a "farness" field (empty cells are very far)
        far = (D[occ].max() + 3.0) if occ.any() else 10.0
        F = np.where(occ, D, far)
        Fp = np.pad(F, 1, mode="edge")
        Fp[0, :] = Fp[-1, :] = Fp[:, 0] = Fp[:, -1] = far
        nbr = np.maximum.reduce([Fp[:-2, 1:-1], Fp[2:, 1:-1], Fp[1:-1, :-2], Fp[1:-1, 2:]])
        edge = occ & (nbr - F > EDGE_T)
        gx = (2 * (Fp[1:-1, 2:] - Fp[1:-1, :-2]) + (Fp[:-2, 2:] - Fp[:-2, :-2])
              + (Fp[2:, 2:] - Fp[2:, :-2]))
        gy = (2 * (Fp[2:, 1:-1] - Fp[:-2, 1:-1]) + (Fp[2:, :-2] - Fp[:-2, :-2])
              + (Fp[2:, 2:] - Fp[:-2, 2:]))
        px, py = gx / self.cw, -gy / self.ch           # outward direction, y up
        ang = np.degrees(np.arctan2(py, px)) % 180.0

        chars = np.full((rows, cols), " ", dtype="<U1")
        for k, ramp in enumerate(RAMPS):
            m = occ & (Kg == k)
            if m.any():
                arr = np.array(list(ramp))
                ri = np.clip((np.clip(B[m], 0, 1) ** 1.1 * len(ramp)).astype(int), 0, len(ramp) - 1)
                chars[m] = arr[ri]
        ec = np.where((ang < 22.5) | (ang >= 157.5), "|",
             np.where(ang < 67.5, "\\",
             np.where(ang < 112.5, np.where(py < 0, "_", "-"), "/")))
        chars[edge] = ec[edge]
        RGB = np.where(edge[..., None], np.clip(RGB * 1.3 + 25, 0, 255), RGB)

        self.depth, self.chars, self.colors, self.occ = D, chars, RGB, occ
        self.letters = int(occ.sum())

    def draw_grid(self):
        rr, cc = np.nonzero(self.occ)
        blits = []
        for r, c in zip(rr.tolist(), cc.tolist()):
            blits.append((self.glyph(self.chars[r, c], self.q(self.colors[r, c])),
                          (c * self.cw, r * self.ch)))
        self.layer.blits(blits, doreturn=False)

    def draw_reflection(self, ground_row, n=9, strength=0.22):
        """Faint mirror image of the rows just above the ground."""
        blits = []
        for k in range(1, n + 1):
            src, dst = ground_row - k, ground_row + k - 1
            if src < 0 or dst >= self.rows:
                continue
            f = strength * (1 - k / (n + 1))
            for c in np.nonzero(self.occ[src])[0].tolist():
                ch = self.chars[src, c]
                ch = {"/": "\\", "\\": "/", "_": "-"}.get(ch, ch)
                blits.append((self.glyph(ch, self.q(self.colors[src, c] * f)),
                              (c * self.cw, dst * self.ch)))
        self.layer.blits(blits, doreturn=False)

    def draw_points(self, P, chars, rgbs, occlude=True):
        """Particles: snap to the character grid, hidden behind nearer geometry."""
        if len(P) == 0:
            return
        sx, sy, zc = self.project(P)
        blits = []
        for x, y, z, ch, rgb in zip(sx, sy, zc, chars, rgbs):
            c, r = int(x // self.cw), int(y // self.ch)
            if not (0 <= c < self.cols and 0 <= r < self.rows):
                continue
            if occlude and self.depth is not None and self.depth[r, c] < z:
                continue
            blits.append((self.glyph(ch, self.q(rgb)), (c * self.cw, r * self.ch)))
        self.layer.blits(blits, doreturn=False)

    def draw_flare(self, surf, x, y, size, rgb, a):
        """A star with little rays made of characters."""
        pts = [("*", 0, 0, 1.0)]
        for k in range(1, size + 1):
            f = (1 - k / (size + 1)) ** 1.5
            pts += [("-", k * 2, 0, f), ("-", -k * 2, 0, f),
                    ("|", 0, k, f), ("|", 0, -k, f)]
            if k <= size // 2 + 1:
                pts += [("\\", -k, -k, f * .7), ("\\", k, k, f * .7),
                        ("/", k, -k, f * .7), ("/", -k, k, f * .7)]
        blits = []
        for ch, dc, dr, f in pts:
            col = self.q(np.asarray(rgb) * a * f)
            if max(col) < 12:
                continue
            blits.append((self.glyph(ch, col), (x + dc * self.cw, y + dr * self.ch)))
        surf.blits(blits, doreturn=False)

    def draw_rays(self, surf, x, y, t, rgb, a, n=9, length=10, spread=60.0, direction=90.0):
        """A fan of flickering light rays made of line characters."""
        blits = []
        for i in range(n):
            ang = math.radians(direction + spread * (i / max(1, n - 1) - 0.5))
            fl = 0.35 + 0.65 * max(0.0, math.sin(t * 2.1 + i * 1.93)) ** 2
            L = max(2, int(length * (0.55 + 0.45 * fl)))
            deg = math.degrees(ang) % 180
            ch = "|" if 72 <= deg <= 108 else "/" if 18 <= deg < 72 else "\\" if 108 < deg <= 162 else "-"
            dx, dy = math.cos(ang), -math.sin(ang)
            for k in range(1, L + 1):
                col = self.q(np.asarray(rgb) * a * fl * (1 - k / (L + 1)) ** 1.2)
                if max(col) < 12:
                    continue
                cx = int((x + dx * k * self.ch * 0.8) // self.cw) * self.cw
                cy = int((y + dy * k * self.ch * 0.8) // self.ch) * self.ch
                blits.append((self.glyph(ch, col), (cx, cy)))
        surf.blits(blits, doreturn=False)

    def composite(self, screen):
        pg = self.pg
        small = pg.transform.smoothscale(self.layer, (W // 5, H // 5))
        glow = pg.transform.smoothscale(small, (W, H))
        screen.blit(glow, (0, 0), special_flags=pg.BLEND_ADD)
        screen.blit(glow, (0, 0), special_flags=pg.BLEND_ADD)
        screen.blit(self.layer, (0, 0), special_flags=pg.BLEND_ADD)


class Starfield:
    def __init__(self, rng, n=260, palette=None, chars="..'+*", flares=9, flare_size=(2, 4),
                 flare_a=0.55, flare_rgb=(200, 200, 215)):
        self.stars = []
        palette = palette or [(235, 235, 245), (255, 150, 175), (150, 190, 255), (255, 255, 255)]
        for _ in range(n):
            self.stars.append((rng.uniform(0, W), rng.uniform(0, H),
                               rng.choice(chars), rng.choice(palette),
                               rng.uniform(0, 6.28), rng.uniform(0.6, 2.4)))
        self.flares = [(rng.uniform(40, W - 40), rng.uniform(40, H - 140),
                        rng.uniform(0, 6.28), rng.uniform(0.15, 0.4), rng.randint(*flare_size))
                       for _ in range(flares)]
        self.flare_a, self.flare_rgb = flare_a, flare_rgb

    def draw(self, R, surf, t):
        blits = []
        for x, y, ch, rgb, ph, sp in self.stars:
            a = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * sp + ph)) ** 2
            col = R.q(np.asarray(rgb) * a * 0.8)
            if max(col) > 10:
                blits.append((R.glyph(ch, col), (x, y)))
        surf.blits(blits, doreturn=False)
        for x, y, ph, sp, size in self.flares:
            a = max(0.0, math.sin(t * sp + ph)) ** 2 * self.flare_a
            if a > 0.03:
                R.draw_flare(surf, x, y, size, self.flare_rgb, a)


# ----------------------------------------------------------------------------
# the rose
# ----------------------------------------------------------------------------
# A single spiral surface: x runs from the centre of a petal to its tip,
# theta winds around the flower (outer petals at small theta).
TH_MIN, TH_MAX = -2 * math.pi, 15 * math.pi
NX, NT = 30, 820
_x = np.linspace(0, 1, NX)[:, None]
_th = np.linspace(TH_MIN, TH_MAX, NT)[None, :]
_tn = (_th - TH_MIN) / (TH_MAX - TH_MIN)
_petal = 1 - 0.5 * (1.25 * (1 - np.mod(3.6 * _th, 2 * np.pi) / np.pi) ** 2 - 0.25) ** 2
_phi_open = np.minimum((np.pi / 2) * np.exp(-_th / (8 * np.pi)), 1.5)
_phi_bud = np.minimum(_phi_open, 0.22 + 0.22 * np.exp(-_th / (8 * np.pi)))
PETAL_COL = (np.array([185, 14, 48])[None, None, :] * (1 - _x[..., None])
             + np.array([255, 80, 118])[None, None, :] * _x[..., None]) * np.ones((1, NT, 1))


def rose_head(bloom):
    # outer petals open first: each turn of the spiral has its own delay
    local = smoothstep(0, 1, np.clip(bloom * 1.6 - _tn * 0.6, 0, 1))
    phi = _phi_bud + (_phi_open - _phi_bud) * local
    curl = 1.95653 * _x ** 2 * (1.27689 * _x - 1) ** 2 * np.sin(phi)
    r = _petal * (_x * np.sin(phi) + curl * np.cos(phi))
    h = _petal * (_x * np.cos(phi) - curl * np.sin(phi))
    P = np.stack([r * np.sin(_th), h, r * np.cos(_th)], -1)
    # closed bud: a rounded, egg-shaped wrap of the same spiral
    psi = np.pi * (0.06 + 0.82 * _x)                      # bottom pole -> near top pole
    rb = 0.37 * np.sin(psi) * (1 - 0.38 * _tn) * (0.88 + 0.12 * _petal)
    hb = 0.46 * (1 - np.cos(psi)) * (1 - 0.10 * _tn)
    Pb = np.stack([rb * np.sin(_th), hb, rb * np.cos(_th)], -1)
    w = local[..., None]
    return Pb * (1 - w) + P * w


def stem_curve(s):
    s = np.asarray(s, float)
    y = GROUND_Y + (STEM_TOP - GROUND_Y) * s
    x = 0.07 * np.sin(s * 2.8) - 0.03 * s
    z = 0.05 * np.sin(s * 4.2)
    return np.stack([x, y, z], -1)


def tube(curve_pts, radius, nv=12):
    T = normalize(np.gradient(curve_pts, axis=0))
    ref = np.array([0.0, 0.0, 1.0])
    A = normalize(np.cross(T, ref))
    Bv = np.cross(T, A)
    a = np.linspace(0, 2 * np.pi, nv, endpoint=False)[None, :, None]
    Nrm = np.cos(a) * A[:, None, :] + np.sin(a) * Bv[:, None, :]
    rad = np.asarray(radius, float).reshape(-1, 1, 1) if np.ndim(radius) else radius
    return curve_pts[:, None, :] + rad * Nrm, Nrm


def leaf(base, yaw, pitch, length, width, grow, droop=0.35, nu=44, nv=15, roll=0.0, fold=0.35):
    """A leaf as a folded, drooping surface. grow 0..1 unfurls it.
    roll turns the blade so its face looks toward the viewer."""
    u = np.linspace(0, 1, nu)[:, None, None]
    v = np.linspace(-1, 1, nv)[None, :, None]
    d = np.array([math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch)])
    side = normalize(np.cross(d, [0.0, 1.0, 0.0]))
    upv = normalize(np.cross(side, d))
    if roll:
        r = roll * (1.0 if side[2] >= 0 else -1.0)
        side, upv = side * math.cos(r) + upv * math.sin(r), upv * math.cos(r) - side * math.sin(r)
    L = length * grow
    wid = width * grow * np.sin(np.pi * u) ** 0.85 * (1 - 0.25 * u)
    center = base + L * u * d - L * droop * u ** 2 * np.array([0, 1.0, 0])
    P = center + side * v * wid * (0.3 + 0.7 * grow) + upv * fold * wid * v ** 2
    P = np.broadcast_to(P, (nu, nv, 3))
    shade_var = 0.7 + 0.55 * (1 - np.abs(v)) ** 1.5 * np.sin(np.pi * u) ** 0.6
    shade_var = shade_var - 0.25 * (np.abs(v) < 0.12)          # midrib line
    return P, surface_normals(P), np.broadcast_to(shade_var[..., 0], (nu, nv))


LEAVES = [  # (stem position, yaw, pitch, length, width)
    (0.26, math.pi + 0.22, 0.18, 1.05, 0.46),
    (0.50, 0.18, 0.30, 0.85, 0.38),
    (0.72, math.pi - 0.35, 0.50, 0.42, 0.16),
]


class Rose:
    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.dew = []        # [pos(3), vel(3), age, hang]
        self.splash = []     # [pos(3), vel(3), age, life, char]
        self.sparks = []     # light-trail particles
        self.dew_timer = 0.0
        self.head_pts = None

    # -- world transform: sway (petals trail) then turn ---------------------
    def to_world(self, P, N):
        t = self.t
        hgt = np.clip((P[..., 1] - GROUND_Y) / (STEM_TOP + 1.0 - GROUND_Y), 0, 1.3)
        amp = 0.12
        dx = amp * hgt ** 2 * np.sin(0.9 * t - 1.3 * hgt)
        dz = amp * 0.6 * hgt ** 2 * np.sin(0.7 * t - 1.3 * hgt + 1.0)
        P = P.copy()
        P[..., 0] += dx
        P[..., 2] += dz
        R = rot_y(self.yaw)
        return P @ R.T, N @ R.T

    @property
    def yaw(self):
        return math.radians(24.0 * math.sin(self.t * 0.33))

    @property
    def bloom(self):
        return float(smoothstep(T_BLOOM0, T_BLOOM1, self.t))

    def build(self):
        """Return (Batch in world space, light-head position or None)."""
        t = self.t
        b = Batch()
        g = float(smoothstep(0.0, T_STEM, t)) if t < T_STEM else 1.0
        light = None

        # stem ------------------------------------------------------------
        s = np.linspace(0, 1, 220)
        cur = stem_curve(s)
        rad = 0.042 - 0.012 * s
        P, N = tube(cur, rad, nv=12)
        vis = np.broadcast_to(((s <= g) & (s < 0.97))[:, None], P.shape[:2])
        stem_col = np.array([40, 200, 70])
        b.add(P, N, stem_col, spec=0.08, mask=vis, gain=0.85)
        if t < T_BUD1:
            light = stem_curve(min(g, 1.0))
            if g >= 1.0:
                light = light + np.array([0, (t - T_STEM) / (T_BUD1 - T_STEM) * 0.8, 0])

        # leaves ----------------------------------------------------------
        for sa, yaw, pitch, L, Wd in LEAVES:
            t0 = sa * T_STEM
            gr = float(smoothstep(t0, t0 + 1.8, t))
            if gr <= 0.01:
                continue
            base = stem_curve(sa)
            # a short stalk from the stem out to the leaf
            out = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            tip = base + out * 0.2 * gr + np.array([0, 0.07 * gr, 0])
            sp = base + (tip - base) * np.linspace(0, 1, 24)[:, None]
            P, N = tube(sp, 0.022, nv=8)
            b.add(P, N, np.array([40, 200, 80]), spec=0.1, gain=0.9)
            P, N, var = leaf(tip, yaw, pitch, L, Wd, gr, droop=0.2, roll=0.8, fold=0.12)
            b.add(P, N, np.array([55, 230, 100]), spec=0.25, gain=1.15 * var)

        # sepals + head ---------------------------------------------------
        top = stem_curve(1.0)
        fill = float(smoothstep(T_BUD0, T_BUD1, t))
        if fill > 0.01:
            for k in range(5):
                gr = fill * (0.6 + 0.4 * self.bloom)
                P, N, _ = leaf(top + [0, -0.04, 0], k * 2 * math.pi / 5 + 0.3,
                               -0.35 - 0.75 * self.bloom, 0.30, 0.07, gr, droop=0.2, nu=16, nv=7)
                b.add(P, N, np.array([45, 200, 80]), spec=0.2, gain=1.1)

            P = rose_head(self.bloom) * (HEAD_SCALE * (0.55 + 0.45 * fill))
            P = P + top + np.array([0, 0.0, 0])
            N = surface_normals(P)
            reveal = np.broadcast_to(_x <= fill * 1.02, P.shape[:2])
            b.add(P, N, PETAL_COL, spec=0.3, mask=reveal, gain=1.2)
            self.head_pts = (P, reveal)
        else:
            self.head_pts = None
        return b, light

    # -- particles ----------------------------------------------------------
    def update_particles(self, dt, light_world):
        rng = self.rng
        grav = np.array([0, -3.2, 0])
        if light_world is not None:
            for _ in range(3):
                v = np.array([rng.uniform(-.4, .4), rng.uniform(-.2, .6), rng.uniform(-.4, .4)])
                self.sparks.append([light_world.copy(), v, 0.0, rng.uniform(.4, 1.0)])
        for p in self.sparks:
            p[1] = p[1] * 0.96 + grav * 0.15 * dt
            p[0] = p[0] + p[1] * dt
            p[2] += dt
        self.sparks = [p for p in self.sparks if p[2] < p[3]]

        # dew drops gather on the petals once the rose is open, then fall
        if self.bloom > 0.55 and self.head_pts is not None:
            self.dew_timer -= dt
            if self.dew_timer <= 0:
                self.dew_timer = rng.uniform(0.35, 1.1)
                P, mask = self.head_pts
                i = rng.randrange(int(NX * 0.55), NX)
                j = rng.randrange(0, int(NT * 0.45))
                if mask[i, j]:
                    w, _ = self.to_world(P[i, j][None], np.zeros((1, 3)))
                    self.dew.append([w[0], np.zeros(3), 0.0, rng.uniform(0.5, 1.4)])
        alive = []
        for d in self.dew:
            d[2] += dt
            if d[2] > d[3]:
                d[1] = d[1] + grav * dt
                d[0] = d[0] + d[1] * dt
            if d[0][1] <= GROUND_Y:
                for _ in range(rng.randint(5, 8)):
                    a = rng.uniform(0, 2 * math.pi)
                    v = np.array([math.cos(a) * rng.uniform(.3, .9), rng.uniform(.8, 1.6),
                                  math.sin(a) * rng.uniform(.3, .9)])
                    self.splash.append([np.array([d[0][0], GROUND_Y, d[0][2]]), v, 0.0,
                                        rng.uniform(.35, .7), rng.choice("'`.,")])
            else:
                alive.append(d)
        self.dew = alive
        for p in self.splash:
            p[1] = p[1] + grav * dt
            p[0] = p[0] + p[1] * dt
            p[2] += dt
        self.splash = [p for p in self.splash if p[2] < p[3] and p[0][1] >= GROUND_Y - 0.05]

    def step(self, dt):
        self.t += dt


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
def render_frame(pg, screen, R, stars, rose, extra_hud=True):
    t = rose.t
    batch, light_local = rose.build()
    P, N, C, S, G, K = batch.arrays()
    cam = rot_x(CAM_TILT)
    if len(P):
        Pw, Nw = rose.to_world(P, N)
        Pc, Nc = Pw @ cam.T, Nw @ cam.T
        mlight = (math.cos(t * 0.8) * 0.8, 0.5, -0.6 + 0.4 * math.sin(t * 0.8))
        lights = [((-0.55, 0.75, -0.65), 0.95, 18), (mlight, 0.45, 40), ((0.4, -0.3, 0.9), 0.25, 8)]
        bright, rgb, _ = shade(Pc, Nc, C, S, lights)
        R.rasterize(Pc, bright * G, rgb * np.minimum(G, 1.15)[:, None], K)
    else:
        R.depth = None
        R.occ = np.zeros((R.rows, R.cols), bool)
        R.letters = 0

    screen.fill(BG)
    stars.draw(R, screen, t)
    R.layer.fill((0, 0, 0))
    if len(P):
        R.draw_grid()
        gy = R.project((np.array([[0, GROUND_Y, 0]]) @ cam.T))[1][0]
        R.draw_reflection(int(gy // R.ch) + 1)

    # particles (already in world space)
    pts, chs, cols = [], [], []
    for p in rose.sparks:
        f = 1 - p[2] / p[3]
        pts.append(p[0]); chs.append("*" if f > .6 else "+" if f > .3 else ".")
        cols.append(np.array([170, 255, 190]) * f)
    for d in rose.dew:
        grow = min(1.0, d[2] / d[3])
        pts.append(d[0]); chs.append("o" if grow > .6 else "." if grow < .3 else "°")
        cols.append(np.array([190, 225, 255]) * (0.5 + 0.5 * grow))
    for p in rose.splash:
        f = 1 - p[2] / p[3]
        pts.append(p[0]); chs.append(p[4]); cols.append(np.array([170, 210, 255]) * f)
    if pts:
        R.draw_points(np.array(pts) @ cam.T, chs, cols)

    # travelling point of light
    if light_local is not None and t < T_BUD1:
        lw, _ = rose.to_world(np.asarray(light_local)[None], np.zeros((1, 3)))
        sx, sy, _ = R.project(lw @ cam.T)
        x = int(sx[0] // R.cw) * R.cw
        y = int(sy[0] // R.ch) * R.ch
        R.draw_flare(R.layer, x, y, 3, (220, 255, 225), 1.0)

    R.composite(screen)

    # hud
    grey = (125, 125, 135)
    hud = [f"IN BLOOM  .  {2026}",
           f"bloom {rose.bloom * 100:5.1f}%   sway {math.degrees(rose.yaw):6.2f}°",
           f"letters {R.letters:,}"]
    for i, line in enumerate(hud):
        screen.blit(R.hud_font.render(line, True, grey), (64, H - 100 + i * 26))
    screen.blit(R.hud_font.render(f"● t {t:05.2f}s", True, (255, 110, 160)), (W - 155, 30))


def main():
    ap = argparse.ArgumentParser(description="IN BLOOM - ASCII rose")
    ap.add_argument("--shot", nargs=2, metavar=("SECONDS", "FILE"),
                    help="render one frame at the given time to an image and exit")
    args = ap.parse_args()

    if args.shot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("IN BLOOM - ASCII rose")
    R = AsciiRenderer(pg)
    stars = Starfield(random.Random(3), n=300, flares=12, flare_size=(3, 6), flare_a=0.75)
    rose = Rose()
    dt = 1.0 / FPS

    if args.shot:
        target = float(args.shot[0])
        while rose.t < target - 1e-6:
            _, light = rose.build()         # particles need the geometry
            lw = rose.to_world(np.asarray(light)[None], np.zeros((1, 3)))[0][0] if light is not None else None
            rose.update_particles(dt, lw)
            rose.step(dt)
        render_frame(pg, screen, R, stars, rose)
        pg.image.save(screen, args.shot[1])
        print(f"saved {args.shot[1]}  (t={rose.t:.2f}s, letters={R.letters})")
        return

    clock = pg.time.Clock()
    paused = False
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                pg.quit()
                return
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key == pg.K_r:
                    rose.reset()
                elif e.key == pg.K_s:
                    name = f"rose_{rose.t:05.2f}s.png"
                    pg.image.save(screen, name)
                    print("saved", name)
        if not paused:
            _, light = rose.build()
            lw = rose.to_world(np.asarray(light)[None], np.zeros((1, 3)))[0][0] if light is not None else None
            rose.update_particles(dt, lw)
            rose.step(dt)
        render_frame(pg, screen, R, stars, rose)
        pg.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
