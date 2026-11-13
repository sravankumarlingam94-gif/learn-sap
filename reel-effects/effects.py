# ============================================================================
# effects.py — motion-graphics effects for vertical (9:16) talking-head clips
# Requires: python3, ffmpeg, numpy, opencv-python-headless, pillow
# Optional (better subject cut-out): onnxruntime + onnx (model fetched from GitHub)
# ============================================================================
import argparse, datetime, functools, json, math, os, subprocess, sys, types, urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
S = 2                                   # supersampling factor for vector layers
CACHE = os.path.expanduser("~/.cache/vfx-effects")
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
cv2.setNumThreads(max(1, os.cpu_count() or 1))

# ----------------------------------------------------------------- fonts ----
FONT_PREFS = {
    "black":   ["Inter-Black.otf", "InterDisplay-Black.otf", "Inter-ExtraBold.otf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "xbold":   ["Inter-ExtraBold.otf", "Inter-Bold.otf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "bold":    ["Inter-Bold.otf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf"],
    "semi":    ["Inter-SemiBold.otf", "Poppins-Medium.ttf", "DejaVuSans-Bold.ttf"],
    "medium":  ["Inter-Medium.otf", "Poppins-Medium.ttf", "DejaVuSans.ttf"],
    "regular": ["Inter-Regular.otf", "Poppins-Regular.ttf", "DejaVuSans.ttf"],
    "mono":    ["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "FreeMonoBold.ttf"],
    "round":   ["Poppins-Bold.ttf", "Inter-Black.otf", "DejaVuSans-Bold.ttf"],
    "serif":   ["Lora-Variable.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"],
    "serifi":  ["Lora-Italic-Variable.ttf", "DejaVuSerif-Italic.ttf", "LiberationSerif-Italic.ttf"],
    "serifb":  ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "Lora-Variable.ttf"],
}
_FONT_INDEX = None

def _font_index():
    global _FONT_INDEX
    if _FONT_INDEX is None:
        _FONT_INDEX = {}
        for root in ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                     os.path.expanduser("~/.local/share/fonts"), "/Library/Fonts", "/System/Library/Fonts"]:
            for dp, _, fs in os.walk(root):
                for f in fs:
                    _FONT_INDEX.setdefault(f, os.path.join(dp, f))
    return _FONT_INDEX

@functools.lru_cache(maxsize=None)
def font(kind, size):
    idx = _font_index()
    for name in FONT_PREFS.get(kind, FONT_PREFS["bold"]):
        if name in idx:
            try:
                return ImageFont.truetype(idx[name], max(4, int(size)))
            except Exception:
                pass
    return ImageFont.load_default(size=max(4, int(size)))

def text_w(s, kind, size):
    return font(kind, size * S).getlength(s) / S

def fit_size(lines, kind, max_w, start, lo=20):
    size = start
    while size > lo and max(text_w(l, kind, size) for l in lines) > max_w:
        size -= 2
    return size

# ------------------------------------------------------------ easing ----
def clamp01(x): return 0.0 if x < 0 else 1.0 if x > 1 else float(x)
def seg(t, a, d): return clamp01((t - a) / d) if d > 0 else float(t >= a)
def eo(x): x = clamp01(x); return 1 - (1 - x) ** 3
def ei(x): x = clamp01(x); return x ** 3
def eio(x): x = clamp01(x); return x * x * (3 - 2 * x)
def eback(x, s=1.9): x = clamp01(x) - 1; return x * x * ((s + 1) * x + s) + 1
def lerp(a, b, t): return a + (b - a) * t
def rgb(h):
    if isinstance(h, (list, tuple)): return tuple(int(v) for v in h[:3])
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
def rgba(c, a=255): return rgb(c) + (int(a),)
def fmt_num(v):
    v = float(v)
    if v >= 1e6: return f"{v/1e6:.1f}M".replace(".0M", "M")
    if v >= 1e4: return f"{v/1e3:.0f}K"
    if v >= 1e3: return f"{v/1e3:.1f}K".replace(".0K", "K")
    return f"{int(round(v))}"
def commas(v): return f"{int(round(v)):,}"

# ------------------------------------------------------------- layers ----
class Layer:
    """Vector canvas drawn at S x resolution; .np() returns float32 (h,w,4) straight alpha in 0..1."""
    def __init__(self, w, h):
        self.w, self.h = max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))
        self.im = Image.new("RGBA", (self.w * S, self.h * S), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
    @staticmethod
    def _b(box): return [v * S for v in box]
    def rrect(self, box, r, fill=None, outline=None, width=1):
        x0, y0, x1, y1 = box
        if x1 - x0 < 1 or y1 - y0 < 1: return
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        self.d.rounded_rectangle(self._b(box), radius=r * S, fill=fill, outline=outline, width=max(1, int(width * S)))
    def rect(self, box, fill): self.d.rectangle(self._b(box), fill=fill)
    def ellipse(self, box, fill=None, outline=None, width=1):
        self.d.ellipse(self._b(box), fill=fill, outline=outline, width=max(1, int(width * S)))
    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.ellipse((cx - r, cy - r, cx + r, cy + r), fill, outline, width)
    def arc(self, box, a0, a1, fill, width):
        if a1 - a0 > 0.5: self.d.arc(self._b(box), a0, a1, fill=fill, width=max(1, int(width * S)))
    def line(self, pts, fill, width=2, joint="curve"):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=max(1, int(width * S)), joint=joint)
    def poly(self, pts, fill=None, outline=None):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline)
    def text(self, xy, s, kind, size, fill, anchor="lt", stroke=0, stroke_fill=None, track=0):
        f = font(kind, size * S)
        if not track:
            self.d.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor,
                        stroke_width=int(stroke * S), stroke_fill=stroke_fill)
            return
        total = sum(f.getlength(ch) for ch in s) / S + track * (len(s) - 1)
        x = xy[0] - (total / 2 if anchor[0] == "m" else total if anchor[0] == "r" else 0)
        for ch in s:
            self.d.text((x * S, xy[1] * S), ch, font=f, fill=fill, anchor="l" + anchor[1],
                        stroke_width=int(stroke * S), stroke_fill=stroke_fill)
            x += f.getlength(ch) / S + track
    def paste(self, arr, x, y, mask_r=0):
        """Paste an RGB uint8/float array (h,w,3) at x,y (layer units), optional rounded corners."""
        a = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(a).resize((a.shape[1] * S, a.shape[0] * S), Image.BILINEAR).convert("RGBA")
        if mask_r:
            mk = Image.new("L", im.size, 0)
            ImageDraw.Draw(mk).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=mask_r * S, fill=255)
            im.putalpha(mk)
        self.im.alpha_composite(im, (int(x * S), int(y * S)))
    def np(self):
        a = np.asarray(self.im.resize((self.w, self.h), Image.BOX) if S > 1 else self.im)
        return a.astype(np.float32) / 255.0

def over(F, L, x, y, op=1.0):
    """Alpha-composite layer L (h,w,4 in 0..1) onto float frame F (0..255) with top-left at x,y."""
    if L is None or op <= 0.003: return
    h, w = L.shape[:2]; x, y = int(round(x)), int(round(y))
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(F.shape[1], x + w), min(F.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0: return
    s = L[Y0 - y:Y1 - y, X0 - x:X1 - x]
    a = s[..., 3:4] * op
    if F.shape[2] == 4:   # compositing onto another RGBA layer (0..1)
        d = F[Y0:Y1, X0:X1]; da_ = d[..., 3:4]
        oa = a + da_ * (1 - a)
        d[..., :3] = np.where(oa > 1e-5, (s[..., :3] * a + d[..., :3] * da_ * (1 - a)) / np.maximum(oa, 1e-5), 0)
        d[..., 3:4] = oa
        return
    F[Y0:Y1, X0:X1] = F[Y0:Y1, X0:X1] * (1 - a) + s[..., :3] * (255.0 * a)

def add(F, G, x, y, op=1.0):
    """Additive blend of premultiplied glow G (h,w,3 in 0..255 units) onto F."""
    if op <= 0.003: return
    h, w = G.shape[:2]; x, y = int(round(x)), int(round(y))
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(F.shape[1], x + w), min(F.shape[0], y + h)
    if X1 <= X0 or Y1 <= Y0: return
    F[Y0:Y1, X0:X1] += G[Y0 - y:Y1 - y, X0 - x:X1 - x] * op

def xform(L, scale=1.0, angle=0.0, sx=None):
    """Scale / rotate a layer about its centre (premultiplied resampling). sx = extra x-squash."""
    if abs(scale - 1) < 1e-3 and abs(angle) < 1e-2 and sx is None: return L
    h, w = L.shape[:2]
    P = L.copy(); P[..., :3] *= P[..., 3:4]
    if sx is not None:
        nw = max(1, int(w * sx)); P = cv2.resize(P, (nw, h), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR); w = nw
    if abs(scale - 1) < 1e-3 and abs(angle) < 1e-2:
        R = P
    else:
        ca, sa = abs(math.cos(math.radians(angle)) * scale), abs(math.sin(math.radians(angle)) * scale)
        nw, nh = int(w * ca + h * sa) + 2, int(w * sa + h * ca) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        R = cv2.warpAffine(P, M, (max(1, nw), max(1, nh)), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = R[..., 3:4]
    R[..., :3] = np.where(a > 1e-4, R[..., :3] / np.maximum(a, 1e-4), 0)
    return R

def over_c(F, L, cx, cy, op=1.0, scale=1.0, angle=0.0, sx=None):
    if L is None or op <= 0.003 or scale <= 0.01: return
    R = xform(L, scale, angle, sx)
    over(F, R, cx - R.shape[1] / 2, cy - R.shape[0] / 2, op)

def glow_from(L, sigma, gain=1.0, color=None):
    """Blurred premultiplied glow (h,w,3, 0..255) from a layer's alpha."""
    a = cv2.GaussianBlur(L[..., 3], (0, 0), sigma)
    col = np.array(color if color is not None else (255, 255, 255), np.float32)
    if color is None:
        rgbp = cv2.GaussianBlur(L[..., :3] * L[..., 3:4], (0, 0), sigma) * 255
        return rgbp * gain
    return a[..., None] * col * gain

@functools.lru_cache(maxsize=512)
def rmask(w, h, r):
    L = Layer(w, h); L.rrect((0, 0, w, h), r, fill=(255, 255, 255, 255)); return L.np()[..., 3]

@functools.lru_cache(maxsize=512)
def rborder(w, h, r, width=1.5):
    L = Layer(w, h); L.rrect((width / 2, width / 2, w - width / 2, h - width / 2), r, outline=(255, 255, 255, 255), width=width)
    return L.np()[..., 3]

@functools.lru_cache(maxsize=512)
def rshadow(w, h, r, pad=44):
    m = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    m[pad:pad + h, pad:pad + w] = rmask(w, h, r)
    return cv2.GaussianBlur(m, (0, 0), pad / 2.6)

def shadow(F, x, y, w, h, r=30, op=1.0, strength=0.45, dy=16, pad=44):
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w < 4 or h < 4 or op <= 0: return
    sm = rshadow(w, h, int(r), pad)
    sx, sy = x - pad, y - pad + dy; FH, FW = F.shape[:2]
    X0, Y0, X1, Y1 = max(0, sx), max(0, sy), min(FW, sx + sm.shape[1]), min(FH, sy + sm.shape[0])
    if X1 <= X0 or Y1 <= Y0: return
    F[Y0:Y1, X0:X1] *= (1 - sm[Y0 - sy:Y1 - sy, X0 - sx:X1 - sx] * strength * op)[..., None]

def fast_blur(reg, sigma):
    if sigma < 1: return reg
    k = 4 if sigma >= 8 else 2
    h, w = reg.shape[:2]
    sm = cv2.resize(reg, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / k)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)

def glass(F, x, y, w, h, r=36, op=1.0, tone="dark", blur=26, border=0.28, shade=True, tint=None):
    """Frosted-glass panel: blurs what is behind it, tints it, adds sheen + hairline border + soft shadow."""
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if op <= 0.003 or w < 6 or h < 6: return
    r = int(min(r, w / 2, h / 2))
    if shade: shadow(F, x, y, w, h, r, op, strength=0.35 if tone == "dark" else 0.22)
    pad = int(blur * 1.5); FH, FW = F.shape[:2]
    X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(FW, x + w + pad), min(FH, y + h + pad)
    cx0, cy0, cx1, cy1 = max(0, x), max(0, y), min(FW, x + w), min(FH, y + h)
    if cx1 <= cx0 or cy1 <= cy0: return
    b = fast_blur(F[Y0:Y1, X0:X1], blur)[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0]
    if tone == "dark":
        t = b * 0.52 + 6
    elif tone == "light":
        t = b * 0.32 + 255 * 0.68
    else:  # "clear"
        t = b * 0.86 + 255 * 0.10
    if tint is not None:
        t = t * 0.8 + np.array(rgb(tint), np.float32) * 0.2
    gy = np.clip(1 - (np.arange(cy0 - y, cy1 - y, dtype=np.float32) / max(1, h * 0.45)), 0, 1)
    t = t + (255 - t) * (gy[:, None, None] * (0.10 if tone != "light" else 0.25))
    m = rmask(w, h, r)[cy0 - y:cy1 - y, cx0 - x:cx1 - x][..., None] * op
    reg = F[cy0:cy1, cx0:cx1]
    reg[:] = reg * (1 - m) + t * m
    if border > 0:
        bm = rborder(w, h, r)[cy0 - y:cy1 - y, cx0 - x:cx1 - x][..., None] * (border * op)
        reg[:] = reg * (1 - bm) + 255 * bm

def grad(w, h, c1, c2, vertical=True, angle=None):
    """RGB float gradient array (h,w,3)."""
    c1, c2 = np.array(rgb(c1), np.float32), np.array(rgb(c2), np.float32)
    if angle is not None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = math.radians(angle); t = (xx * math.cos(a) + yy * math.sin(a))
        t = (t - t.min()) / max(1e-6, t.max() - t.min())
    else:
        t = np.linspace(0, 1, h if vertical else w, dtype=np.float32)
        t = t[:, None] if vertical else t[None, :]
        t = np.broadcast_to(t, (h, w))
    return c1 * (1 - t[..., None]) + c2 * t[..., None]

def to_layer(arr, alpha=None):
    a = np.ones(arr.shape[:2], np.float32) if alpha is None else alpha.astype(np.float32)
    return np.dstack([np.clip(arr, 0, 255).astype(np.float32) / 255.0, a])

def rounded_frame(arr, r, border=0, border_col=(255, 255, 255)):
    """RGB array -> RGBA layer with rounded corners and optional border."""
    h, w = arr.shape[:2]
    L = to_layer(arr, rmask(w, h, int(r)))
    if border:
        bm = rborder(w, h, int(r), border)[..., None]
        L[..., :3] = L[..., :3] * (1 - bm) + np.array(border_col, np.float32) / 255 * bm
    return L

def vignette(strength=0.5, cx=0.5, cy=0.45):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx / W - cx) ** 2 * 1.6 + (yy / H - cy) ** 2) / 0.35
    return (1 - strength * np.clip(d, 0, 1) ** 1.3)[..., None]

def grade(O, sat=1.0, contrast=1.0, bright=1.0, tint=(1, 1, 1), lift=0.0):
    g = O @ np.array([0.299, 0.587, 0.114], np.float32)
    X = g[..., None] + (O - g[..., None]) * sat
    X = (X - 128) * contrast + 128
    return np.clip(X * bright * np.array(tint, np.float32) + lift, 0, 255)

def shift_img(A, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), borderMode=cv2.BORDER_REPLICATE)

def crop_cover(A, w, h, cx=None, cy=None, zoom=1.0):
    """Crop+scale array A to exactly (w,h) like CSS object-fit: cover, centred on cx,cy."""
    ah, aw = A.shape[:2]
    s = max(w / aw, h / ah) * zoom
    cw, ch = w / s, h / s
    cx = aw / 2 if cx is None else cx; cy = ah / 2 if cy is None else cy
    x0 = int(min(max(cx - cw / 2, 0), aw - cw)); y0 = int(min(max(cy - ch / 2, 0), ah - ch))
    sub = A[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(sub, (int(w), int(h)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

def star_pts(cx, cy, r_out, r_in, n=4, rot=0):
    pts = []
    for k in range(n * 2):
        r = r_out if k % 2 == 0 else r_in
        a = math.pi * k / n + rot - math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def check_path(L, cx, cy, s, p, fill, width):
    """Draw a check mark progressively (p 0..1)."""
    pts = [(cx - 0.45 * s, cy + 0.02 * s), (cx - 0.12 * s, cy + 0.32 * s), (cx + 0.48 * s, cy - 0.30 * s)]
    l1 = math.dist(pts[0], pts[1]); l2 = math.dist(pts[1], pts[2]); L_ = (l1 + l2) * p
    if L_ <= 0: return
    if L_ <= l1:
        t = L_ / l1; L.line([pts[0], (lerp(pts[0][0], pts[1][0], t), lerp(pts[0][1], pts[1][1], t))], fill, width)
    else:
        t = (L_ - l1) / l2; L.line([pts[0], pts[1], (lerp(pts[1][0], pts[2][0], t), lerp(pts[1][1], pts[2][1], t))], fill, width)

PALETTE = [(255, 90, 54), (255, 176, 46), (64, 196, 255), (150, 110, 255), (60, 214, 150), (255, 72, 150), (250, 220, 70)]

# ------------------------------------------------------- subject mask ----
@functools.lru_cache(maxsize=1)
def silhouette():
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (W // 2, int(H * 0.34)), (int(W * 0.17), int(H * 0.125)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(m, (W // 2, int(H * 1.02)), (int(W * 0.40), int(H * 0.52)), 0, 180, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 14)

def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path + ".part")
    except Exception:
        subprocess.run(["curl", "-sSL", "-m", "120", "-o", path + ".part", url], check=True)
    os.replace(path + ".part", path)

class Segmenter:
    """Person / main-subject matte. AI model (u2netp) when available, otherwise a soft silhouette."""
    def __init__(self, mode="auto"):
        self.sess, self.prev, self.ai, self.every = None, None, False, 2
        self.size = (320, 320)
        if mode in ("auto", "ai"):
            try:
                import onnxruntime as ort
                base = os.path.join(CACHE, "u2netp.onnx")
                if not os.path.exists(base): _download(MODEL_URL, base)
                path = base
                try:   # make input size dynamic -> much faster at 9:16
                    dyn = os.path.join(CACHE, "u2netp_dyn.onnx")
                    if not os.path.exists(dyn):
                        import onnx
                        mdl = onnx.load(base)
                        dd = mdl.graph.input[0].type.tensor_type.shape.dim; dd[2].dim_param = "h"; dd[3].dim_param = "w"
                        for o in mdl.graph.output:
                            od = o.type.tensor_type.shape.dim
                            if len(od) == 4: od[2].dim_param = "h"; od[3].dim_param = "w"
                        onnx.save(mdl, dyn)
                    path, self.size = dyn, (160, 288)
                except Exception:
                    self.every = 3
                so = ort.SessionOptions(); so.log_severity_level = 3
                so.intra_op_num_threads = max(1, os.cpu_count() or 1)
                self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                self.inp = self.sess.get_inputs()[0].name
                self.ai = True
            except Exception as e:
                if mode == "ai": raise
                print(f"[effects] AI subject mask unavailable ({e}); using soft silhouette.", file=sys.stderr)
        self.n = 0
    def __call__(self, frame):
        if not self.ai: return silhouette()
        self.n += 1
        if self.prev is not None and (self.n - 1) % self.every: return self.prev
        x = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        x /= max(float(x.max()), 1e-6)
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        y = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        y = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
        m = np.clip((cv2.resize(y, (W, H), interpolation=cv2.INTER_LINEAR) - 0.28) / 0.44, 0, 1).astype(np.float32)
        if self.prev is not None: m = 0.6 * m + 0.4 * self.prev
        self.prev = m
        return m

class Subject:
    """Smoothed head position / size + body centroid estimated from the matte."""
    def __init__(self):
        self.hx, self.hy, self.r, self.cx, self.cy, self.top = W / 2, H * 0.34, 170.0, W / 2, H * 0.62, H * 0.22
        self.init = False
    def update(self, m):
        k = 8; sm = cv2.resize(m, (W // k, H // k), interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(sm > 0.5)
        if len(ys) < 40: return
        top = ys.min(); bh = max(4, int(0.10 * sm.shape[0]))
        band = sm[top:top + bh] > 0.5
        cols = np.nonzero(band.mean(0) > 0.35)[0]
        if len(cols) == 0: return
        hx = cols.mean() * k; r = float(np.clip((cols.max() - cols.min() + 1) * k / 2, 70, 300))
        est = (hx, top * k + r * 1.05, r, xs.mean() * k, ys.mean() * k, top * k)
        a = 1.0 if not self.init else 0.25
        self.hx, self.hy, self.r, self.cx, self.cy, self.top = [lerp(o, n, a) for o, n in
                                                                 zip((self.hx, self.hy, self.r, self.cx, self.cy, self.top), est)]
        self.init = True

# ------------------------------------------------------ media helpers ----
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    dur = float(j.get("format", {}).get("duration", 0) or 0)
    has_a = any(s.get("codec_type") == "audio" for s in j.get("streams", []))
    fps = 30.0
    for s in j.get("streams", []):
        if s.get("codec_type") == "video":
            try:
                n_, d_ = s["r_frame_rate"].split("/"); fps = float(n_) / float(d_)
            except Exception: pass
    return dur, has_a, fps

def audio_env(path, start, dur, fps, n):
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", path, "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        if a.size < 4000: raise ValueError("no audio")
        hop = 16000 / fps
        idx = (np.arange(n + 1) * hop).astype(int); idx = np.clip(idx, 0, a.size)
        e = np.array([np.sqrt(np.mean(a[idx[i]:idx[i + 1]] ** 2)) if idx[i + 1] > idx[i] else 0 for i in range(n)])
        e = e / max(np.percentile(e, 95), 1e-4)
        return np.clip(e, 0, 1.2).astype(np.float32)
    except Exception:
        t = np.arange(n) / fps
        return (0.45 + 0.35 * np.sin(t * 9.1) * np.sin(t * 2.3)).astype(np.float32)

def grab_frames(path, start, dur, count, tw, th):
    vf = f"fps={max(count, 1) / max(dur, 0.1):.5f},scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", path, "-vf", vf,
                          "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    sz = tw * th * 3; k = len(raw) // sz
    if k == 0: return [np.full((th, tw, 3), 60, np.uint8)] * count
    fr = [np.frombuffer(raw[i * sz:(i + 1) * sz], np.uint8).reshape(th, tw, 3) for i in range(k)]
    while len(fr) < count: fr.append(fr[len(fr) % k])
    return fr

# ------------------------------------------------------------ registry ----
EFFECTS = {}
def effect(name, desc, mask=False, defaults=None, main=None):
    def deco(fn):
        EFFECTS[name] = types.SimpleNamespace(fn=fn, desc=desc, mask=mask, defaults=defaults or {}, main=main)
        return fn
    return deco

class Ctx(types.SimpleNamespace):
    def frames(self, count, tw, th, key="thumbs"):
        """`count` evenly spaced frames from the current effect segment (cached)."""
        k = (key, count, tw, th, self.seg_a)
        if k not in self.cache:
            self.cache[k] = grab_frames(self.src, self.start + self.seg_a, self.D, count, tw, th)
        return self.cache[k]
    def st(self, name):
        return self.state.setdefault((name, self.seg_a), {})



LUMA = np.array([0.299, 0.587, 0.114], np.float32)
WHITE = (255, 255, 255, 255)

def wa(a): return (255, 255, 255, int(a))

# ---------------------------------------------------------------- shared UI bits ----
def phone_screen(k, w, h):
    themes = [((124, 58, 237), (236, 72, 153)), ((15, 23, 42), (30, 58, 138)), ((246, 246, 248), (232, 233, 238)),
              ((251, 146, 60), (225, 52, 72)), ((255, 255, 255), (242, 242, 246)), ((22, 22, 26), (40, 40, 48))]
    k %= 6; c1, c2 = themes[k]
    L = Layer(w, h); L.paste(grad(w, h, c1, c2), 0, 0)
    dark = k in (0, 1, 3, 5)
    fg = WHITE if dark else (22, 22, 26, 255); fg2 = fg[:3] + (150,)
    L.text((22, 64), ["Reels", "Analytics", "Messages", "Now playing", "Calendar", "Gallery"][k], "bold", 28, fg)
    if k == 0:
        L.circle(w / 2, h / 2, 48, fill=(255, 255, 255, 80))
        L.poly([(w / 2 - 14, h / 2 - 22), (w / 2 - 14, h / 2 + 22), (w / 2 + 24, h / 2)], fill=WHITE)
        L.rrect((22, h - 64, w - 22, h - 58), 3, fill=(255, 255, 255, 90)); L.rrect((22, h - 64, w * 0.55, h - 58), 3, fill=WHITE)
        L.text((22, h - 110), "@studio", "semi", 22, fg)
    elif k == 1:
        L.text((22, 112), "+38%", "black", 54, fg); L.text((22, 176), "reach this week", "regular", 20, fg2)
        pts = [(22 + i * (w - 44) / 7, 380 - v) for i, v in enumerate([20, 60, 40, 90, 70, 120, 105, 150])]
        L.line(pts, (96, 165, 250, 255), 5)
        for i in range(6): L.rrect((22 + i * 42, 520 - 30 - i * 12, 52 + i * 42, 520), 6, fill=(96, 165, 250, 120 + i * 20))
    elif k == 2:
        y = 120
        for j, (wd, me) in enumerate([(170, 0), (140, 1), (200, 0), (120, 1), (160, 1)]):
            x0 = w - 22 - wd if me else 22
            L.rrect((x0, y, x0 + wd, y + 52), 24, fill=(10, 132, 255, 255) if me else (229, 229, 234, 255)); y += 70
    elif k == 3:
        L.paste(grad(w - 60, w - 60, (255, 214, 102), (255, 94, 98), angle=45), 30, 110, mask_r=18)
        L.text((30, w + 70), "Golden hour", "bold", 26, fg); L.text((30, w + 104), "Studio mix", "regular", 20, fg2)
        L.rrect((30, w + 150, w - 30, w + 156), 3, fill=(255, 255, 255, 90)); L.rrect((30, w + 150, w * 0.6, w + 156), 3, fill=WHITE)
        for j, r_ in enumerate([18, 28, 18]): L.circle(w / 2 + (j - 1) * 70, w + 210, r_, fill=WHITE)
    elif k == 4:
        L.text((22, 104), "13", "black", 84, (230, 57, 70, 255))
        for j, cc in enumerate(PALETTE[:4]):
            y = 230 + j * 78; L.rrect((22, y, 28, y + 58), 3, fill=cc + (255,))
            L.rrect((40, y + 6, 40 + 150 - j * 18, y + 22), 6, fill=(22, 22, 26, 200)); L.rrect((40, y + 32, 140, y + 44), 6, fill=(22, 22, 26, 90))
    else:
        cw = (w - 44 - 12) / 3
        for j in range(12):
            r_, q = divmod(j, 3); cc = PALETTE[(j * 3) % len(PALETTE)]
            L.rrect((22 + q * (cw + 6), 110 + r_ * (cw + 6), 22 + q * (cw + 6) + cw, 110 + r_ * (cw + 6) + cw), 8, fill=cc + (255,))
    return (L.np()[..., :3] * 255)

def load_img(path):
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None

def phone_mockup(k, images=(), w=300, h=620, screen=None):
    L = Layer(w, h)
    L.rrect((0, 0, w, h), 50, fill=(14, 14, 18, 255))
    L.rrect((2, 2, w - 2, h - 2), 48, outline=(110, 110, 122, 255), width=2)
    sw, sh = w - 24, h - 24
    img = screen
    if img is None and images:
        img = load_img(images[k % len(images)])
    scr = crop_cover(img.astype(np.float32), sw, sh) if img is not None else phone_screen(k, sw, sh)
    L.paste(scr, 12, 12, mask_r=38)
    L.rrect((w / 2 - 44, 26, w / 2 + 44, 52), 13, fill=(0, 0, 0, 255))
    return L.np()

# ================================================================== 1. glass-widgets ====
def _gw(kind, w, h, tl, c, acc):
    p = c.cfg; L = Layer(w, h)
    if kind == "counter":
        L.text((28, 26), p.counter_kicker, "semi", 22, wa(170), track=3)
        L.text((24, 56), str(int(round(p.counter * eo(seg(tl, 0.15, 1.2))))), "black", 118, WHITE)
        L.text((28, 200), p.counter_label, "medium", 28, wa(205))
    elif kind == "status":
        pu = 0.5 + 0.5 * math.sin(tl * 6)
        L.circle(40, 46, 15 + 9 * pu, fill=acc + (int(80 * (1 - pu)),)); L.circle(40, 46, 10, fill=acc + (255,))
        L.text((66, 28), p.status + "." * (int(tl * 3) % 4), "semi", 34, WHITE)
        L.text((28, 86), p.status_sub, "regular", 24, wa(175))
    elif kind == "ring":
        cx, cy, R = w / 2, 140, 100; g = eo(seg(tl, 0.2, 1.4)); pr = p.ring_pct / 100 * g
        L.arc((cx - R, cy - R, cx + R, cy + R), 0, 360, wa(45), 18)
        L.arc((cx - R, cy - R, cx + R, cy + R), -90, -90 + 360 * pr, acc + (255,), 18)
        L.text((cx, cy), f"{int(round(p.ring_pct * g))}%", "black", 58, WHITE, anchor="mm")
        L.text((cx, 290), p.ring_label, "medium", 28, wa(205), anchor="mm")
    elif kind == "bars":
        L.text((28, 26), p.bars_title, "semi", 26, wa(205))
        vals = p.bars; n = len(vals); mx = max(vals) or 1; bw = (w - 56) / n
        for j, v in enumerate(vals):
            g = eo(seg(tl, 0.15 + 0.08 * j, 0.6)); hh = max(6, 150 * v / mx * g)
            x0, x1 = 28 + j * bw + bw * 0.18, 28 + (j + 1) * bw - bw * 0.18
            L.rrect((x0, 236 - hh, x1, 236), 7, fill=acc + (255,) if j == n - 1 else wa(110))
            L.text(((x0 + x1) / 2, 266), "MTWTFSS"[j % 7], "medium", 20, wa(150), anchor="mm")
    elif kind == "follow":
        for j, cc in enumerate(PALETTE[:3]): L.circle(46 + j * 36, 60, 27, fill=cc + (255,), outline=wa(230), width=3)
        L.text((168, 20), "+" + commas(p.followers * eo(seg(tl, 0.2, 1.2))), "black", 42, WHITE)
        L.text((170, 72), p.followers_label, "medium", 24, wa(195))
    elif kind == "audio":
        L.rrect((42, 38, 72, 90), 15, fill=wa(235)); L.arc((32, 50, 82, 102), 0, 180, wa(235), 4)
        L.line([(57, 102), (57, 116)], wa(235), 4)
        nb, x0, x1 = 46, 110, w - 40; bw = (x1 - x0) / nb; g = eo(seg(tl, 0.1, 0.5))
        for j in range(nb):
            idx = c.i - (nb - 1 - j); v = c.env[idx] if 0 <= idx < len(c.env) else 0.0
            hh = 8 + 92 * min(1.0, v) * g
            L.rrect((x0 + j * bw + 2, 75 - hh / 2, x0 + (j + 1) * bw - 2, 75 + hh / 2), 3, fill=wa(110 + 145 * j / nb))
    return L.np()

@effect("glass-widgets", "Frosted-glass stat widgets (counter, status, ring meter, bar chart, followers, live audio) pop in around the speaker.",
        defaults=dict(counter=36, counter_kicker="LIBRARY", counter_label="effects ready", status="Rendering", status_sub="4K · 60 fps · HDR",
                      ring_pct=84, ring_label="Engagement", bars_title="Views this week", bars=[32, 45, 38, 60, 52, 78, 96],
                      followers=2431, followers_label="new followers", accent="#7CF5C4"), main="counter_label")
def fx_glass_widgets(F, O, c):
    acc = rgb(c.cfg.accent)
    cards = [("counter", 48, 190, 330, 250, 0.10), ("status", 702, 190, 330, 130, 0.25), ("ring", 702, 350, 330, 330, 0.40),
             ("bars", 48, 470, 330, 300, 0.55), ("follow", 48, 1360, 470, 120, 0.70), ("audio", 120, 1570, 840, 150, 0.85)]
    for kind, x, y, w, h, dl in cards:
        a = seg(c.lt, dl, 0.5)
        if a <= 0: continue
        s = 0.8 + 0.2 * eback(a); op = eo(a * 1.4); cx, cy = x + w / 2, y + h / 2
        glass(F, cx - w * s / 2, cy - h * s / 2, w * s, h * s, r=34 * s, op=op)
        over_c(F, _gw(kind, w, h, c.lt - dl, c, acc), cx, cy, op=op, scale=s)
    return F

# ================================================================== 2. documentary ====
def _doc_circle(r, p, col):
    rx, ry, pad = r * 1.45, r * 1.62, 30
    L = Layer(2 * rx + 2 * pad, 2 * ry + 2 * pad)
    L.arc((pad, pad, pad + 2 * rx, pad + 2 * ry), 200, 200 + 360 * p, col + (255,), 10)
    if p > 0.55:
        L.arc((pad + 12, pad - 4, pad + 2 * rx + 6, pad + 2 * ry - 14), 215, 215 + 300 * (p - 0.55) / 0.45, col + (190,), 5)
    return L.np()

@effect("documentary", "Black-and-white documentary poster: huge headline behind the speaker, white cut-out outline, red hand-drawn circle, chapter lower-third, film grain.",
        mask=True, defaults=dict(kicker="AN ORIGINAL DOCUMENTARY", headline="THE UNTOLD\nSTORY", chapter="CHAPTER 01",
                                 subtitle="How it all started", accent="#E5322D"), main="headline")
def fx_documentary(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("doc"); acc = rgb(p.accent)
    if "vig" not in st:
        st["vig"] = vignette(0.55)
        rng = np.random.default_rng(3)
        st["grain"] = [cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))[..., None] for _ in range(4)]
        lines = [l.upper() for l in p.headline.split("\n")]
        size = fit_size(lines, "black", 990, 240); lh = size * 0.9
        L = Layer(W, lh * len(lines) + 50)
        for j, l in enumerate(lines): L.text((W / 2, j * lh), l, "black", size, (246, 244, 238, 255), anchor="mt")
        st["hl"] = L.np()
        K = Layer(W, 44); K.text((W / 2, 4), p.kicker, "semi", 26, wa(215), anchor="mt", track=6); st["kick"] = K.np()
        T = Layer(860, 140)
        T.rect((0, 8, 10, 122), fill=acc + (255,))
        T.text((34, 4), p.chapter, "bold", 30, acc + (255,), track=4)
        T.text((34, 50), p.subtitle, "semi", 52, WHITE, stroke=1, stroke_fill=(0, 0, 0, 120))
        st["lt"] = T.np(); st["circ"] = {}
    m = c.mask[..., None]
    g = np.clip(((O @ LUMA) / 255 - 0.5) * 1.45 + 0.5, 0, 1); g = g * g * (3 - 2 * g)
    G = (g * 255)[..., None]
    F = G * np.array([0.48, 0.50, 0.54], np.float32) * st["vig"]
    a = seg(lt, 0.05, 0.7)
    over(F, st["kick"], 0, 112 + 30 * (1 - eo(a)), eo(a))
    over(F, st["hl"], 0, 168 + 60 * (1 - eo(a)), eo(a))
    F = F * (1 - m) + G * np.array([1.03, 1.0, 0.97], np.float32) * m
    if c.ai:
        ms = cv2.resize(c.mask, (W // 2, H // 2))
        ring = cv2.resize(np.clip(cv2.dilate(ms, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) - ms, 0, 1), (W, H))[..., None]
        F = F * (1 - ring * 0.95) + 255 * ring * 0.95
    pc = eo(seg(lt, 0.7, 0.75))
    if pc > 0:
        key = (int(c.subj.r / 6), int(pc * 40))
        if key not in st["circ"]:
            if len(st["circ"]) > 64: st["circ"].clear()
            st["circ"][key] = _doc_circle(key[0] * 6, key[1] / 40, acc)
        over_c(F, st["circ"][key], c.subj.hx, c.subj.hy - c.subj.r * 0.05, angle=-8)
    b = eo(seg(lt, 1.0, 0.6))
    over(F, st["lt"], 60 - 560 * (1 - b), 1590, b)
    F += st["grain"][c.i % 4] * 11
    return F

# ================================================================== 3. orbit ====
@effect("orbit", "Six phone screens orbit around the speaker in 3D, passing behind and in front of them.", mask=True,
        defaults=dict(images=[], speed=0.75))
def fx_orbit(F, O, c):
    st = c.st("orbit"); lt = c.lt
    if "ph" not in st: st["ph"] = [phone_mockup(k, c.cfg.images) for k in range(6)]
    m = c.mask[..., None]
    F *= 1 - 0.30 * (1 - m)
    s = c.subj; cx, cy = s.hx, min(H - 500, s.hy + s.r * 1.9)
    grow = eo(seg(lt, 0, 1.0)); rx, ry = 40 + 440 * grow, 30 + 150 * grow
    items = []
    for k in range(6):
        th = lt * c.cfg.speed + k * math.pi / 3 + 0.3; z = math.sin(th)
        items.append((z, cx + rx * math.cos(th), cy + ry * z, (0.58 + 0.16 * (z + 1)) * (0.5 + 0.5 * grow), k, th))
    items.sort()
    op = eo(grow * 1.5)
    for z, x, y, sc, k, th in items:
        if z >= 0: continue
        ph = st["ph"][k].copy(); ph[..., :3] *= 0.6
        over_c(F, ph, x, y, op=op, scale=sc, sx=0.55 + 0.45 * abs(math.sin(th)))
    F = F * (1 - m) + O * m
    for z, x, y, sc, k, th in items:
        if z < 0: continue
        over_c(F, st["ph"][k], x, y, op=op, scale=sc, sx=0.55 + 0.45 * abs(math.sin(th)))
    return F

# ================================================================== 4. split-chart ====
@effect("split-chart", "Split screen: an animated 'top reels by views' bar chart on top, the speaker below.", mask=True,
        defaults=dict(title="Top reels by views", subtitle="Last 30 days",
                      data=[["Morning routine", 1240000], ["Desk setup tour", 982000], ["5 AI tools I use", 871000],
                            ["Budget tips", 640000], ["Day in my life", 412000]]), main="title")
def fx_split_chart(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("split"); TH = 860
    th = int(TH * eo(seg(lt, 0, 0.55))); vh = H - th
    y0 = int(np.clip(c.subj.hy - 0.30 * vh, 0, H - vh))
    F[th:] = O[y0:y0 + vh]
    if th <= 0: return F
    if "bg" not in st:
        bg = grad(W, TH, (18, 19, 25), (9, 10, 13)); Lg = Layer(W, TH)
        for gx in range(130, 1000, 152): Lg.rect((gx, 225, gx + 1, TH - 50), fill=wa(14))
        Lg.text((60, 70), p.title, "bold", 56, WHITE); Lg.text((60, 142), p.subtitle, "regular", 30, wa(140))
        over(bg, Lg.np(), 0, 0); st["bg"] = bg
    P = st["bg"].copy()
    data = sorted(p.data, key=lambda d: -d[1])[:6]; mx = data[0][1] or 1
    L = Layer(W, 116 * len(data))
    for j, (lab, val) in enumerate(data):
        y = j * 116; g = eo(seg(lt, 0.35 + 0.12 * j, 0.9)); col = PALETTE[j % len(PALETTE)]
        L.circle(84, y + 58, 24, fill=col + (255,)); L.text((84, y + 58), str(j + 1), "black", 26, (16, 16, 20, 255), anchor="mm")
        L.text((130, y + 6), str(lab), "semi", 32, wa(235))
        bw = max(36, 740 * val / mx * g)
        L.rrect((130, y + 58, 130 + bw, y + 94), 18, fill=col + (255,))
        L.text((130 + bw + 16, y + 76), fmt_num(val * g), "bold", 30, WHITE, anchor="lm")
    over(P, L.np(), 0, 222)
    F[:th] = P[TH - th:]
    F[th:th + 5] = F[th:th + 5] * 0.15 + np.array([255, 90, 54], np.float32) * 0.85
    return F

# ================================================================== 5. spotlight ====
def neon_flicker(t):
    if t < 0: return 0.0
    for a, b, v in [(0, .06, 1), (.06, .14, .08), (.14, .2, 1), (.2, .3, .2), (.3, .36, .9), (.36, .42, .15)]:
        if a <= t < b: return v
    return 0.96 + 0.04 * math.sin(t * 50)

@effect("spotlight", "Room goes dark, a stage spotlight hits the speaker and a flickering neon title glows behind them.", mask=True,
        defaults=dict(title="SPOTLIGHT", color="#FF3FB4"), main="title")
def fx_spotlight(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("spot"); col = np.array(rgb(p.color), np.float32)
    if "core" not in st:
        t = p.title.upper(); size = fit_size([t], "black", 980, 300); pad = 110
        L = Layer(text_w(t, "black", size) + 2 * pad, size * 1.2 + 2 * pad)
        L.text((pad, pad), t, "black", size, (0, 0, 0, 0), stroke=5, stroke_fill=WHITE)
        N = L.np(); st["g1"] = glow_from(N, 5, 1.5, col); st["g2"] = glow_from(N, 20, 1.3, col); st["g3"] = glow_from(N, 55, 0.9, col)
        N[..., :3] = col / 255 * 0.35 + 0.65; st["core"] = N
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        hw = 80 + 0.26 * yy
        cone = np.clip(1 - np.abs(xx - W / 2) / hw, 0, 1) ** 1.6 * (0.55 + 0.45 * (1 - yy / H))
        st["cone"] = cv2.GaussianBlur(cone, (0, 0), 25); st["cones"] = {}
    pp = eo(seg(lt, 0, 0.6))
    bg = O * (1 - 0.80 * pp) * np.array([0.82, 0.88, 1.08], np.float32)
    dx = int((c.subj.hx - W / 2) / 8) * 8
    if dx not in st["cones"]:
        st["cones"][dx] = cv2.warpAffine(st["cone"], np.float32([[1, 0, dx], [0, 1, 0]]), (W, H))[..., None] * np.array([255, 236, 210], np.float32)
    bg += st["cones"][dx] * 0.30 * pp
    fl = neon_flicker(lt - 0.35)
    if fl > 0:
        N = st["core"]; nx = W / 2 - N.shape[1] / 2
        ny = max(-80, c.subj.hy - c.subj.r * 0.6 - N.shape[0] / 2)
        add(bg, st["g3"], nx, ny, 0.9 * fl); add(bg, st["g2"], nx, ny, fl); add(bg, st["g1"], nx, ny, fl)
        over(bg, N, nx, ny, fl)
    m = c.mask[..., None]
    F = bg * (1 - m) + O * 1.04 * m
    if c.ai:
        ms = cv2.resize(c.mask, (W // 4, H // 4)); ring = np.clip(cv2.GaussianBlur(ms, (0, 0), 6) - ms, 0, 1) * 2.2
        F += cv2.resize(ring, (W, H))[..., None] * np.array([255, 214, 170], np.float32) * 0.55 * pp
    return F

# ================================================================== 6. checklist ====
@effect("checklist", "Glass checklist card ticks off editing steps one by one with spinners and a live progress bar.",
        defaults=dict(title="Editing your clip", items=["Cutting silences", "Color grading", "Adding captions", "Mixing audio", "Exporting in 4K"],
                      accent="#34C759"), main="items")
def fx_checklist(F, O, c):
    p, lt = c.cfg, c.lt; acc = rgb(p.accent); items = p.items; n = len(items)
    w, h = 900, 150 + 80 * n + 80; x, y = 90, H - h - 150
    a = seg(lt, 0, 0.5); yy = y + 90 * (1 - eo(a)); op = eo(a)
    glass(F, x, yy, w, h, r=42, op=op)
    t0 = 0.6; per = max(0.3, (c.D - t0 - 0.7) / n)
    done = sum(1 for j in range(n) if lt >= t0 + per * (j + 1))
    L = Layer(w, h)
    L.text((44, 40), p.title, "bold", 44, WHITE); L.text((w - 44, 50), f"{done}/{n}", "semi", 30, wa(170), anchor="rt")
    for j, it in enumerate(items):
        ry = 132 + j * 80; cx, cy = 66, ry + 28; ts, te = t0 + per * j, t0 + per * (j + 1)
        if lt >= te:
            L.circle(cx, cy, 20 * eback(seg(lt, te, 0.3)), fill=acc + (255,))
            check_path(L, cx, cy, 26, seg(lt, te + 0.05, 0.25), WHITE, 4); tc = wa(215)
        elif lt >= ts:
            ang = (lt * 576) % 360
            L.circle(cx, cy, 20, outline=wa(50), width=4); L.arc((cx - 20, cy - 20, cx + 20, cy + 20), ang, ang + 100, acc + (255,), 4); tc = WHITE
        else:
            L.circle(cx, cy, 20, outline=wa(90), width=3); tc = wa(120)
        L.text((106, ry + 10), it, "semi", 36, tc)
    prog = clamp01((lt - t0) / (per * n)); bx0, bx1, by = 44, w - 150, h - 62
    L.rrect((bx0, by, bx1, by + 12), 6, fill=wa(45)); L.rrect((bx0, by, bx0 + (bx1 - bx0) * prog, by + 12), 6, fill=acc + (255,))
    L.text((w - 44, by + 6), f"{int(prog * 100)}%", "bold", 32, WHITE, anchor="rm")
    over(F, L.np(), x, yy, op)
    return F

# ================================================================== 7. timeline ====
def timeline_static(c, w, TX0, TW, thumbs_y, thumbs_h, wave_y, wave_h, clips, nsec):
    """Shared editor-timeline drawing used by timeline / pull-back. Returns (static RGBA, bright-wave RGBA)."""
    L = Layer(w, wave_y + wave_h + 10); Hi = Layer(TW, wave_h)
    for s_ in range(nsec + 1):
        gx = TX0 + TW * s_ / max(1, nsec)
        L.rect((gx, thumbs_y - 120, gx + 2, thumbs_y - 104), fill=wa(120))
        L.text((gx + 6, thumbs_y - 126), f"{s_}s", "medium", 18, wa(140))
    ty = thumbs_y - 90
    spans = [(0.02, 0.30), (0.36, 0.66), (0.72, 0.97)]
    for j, (a, b) in enumerate(spans):
        cc = [(150, 110, 255), (255, 176, 46), (60, 214, 150)][j]
        L.rrect((TX0 + TW * a, ty, TX0 + TW * b, ty + 60), 12, fill=cc + (235,))
        L.text((TX0 + TW * a + 16, ty + 30), clips[j % len(clips)] if clips else "", "semi", 22, (20, 20, 24, 255), anchor="lm")
    th = c.frames(8, int(TW / 8), thumbs_h)
    for j, im in enumerate(th): L.paste(im, TX0 + j * TW / 8, thumbs_y)
    L.rrect((TX0, thumbs_y, TX0 + TW, thumbs_y + thumbs_h), 10, outline=(255, 214, 10, 255), width=3)
    i0 = int(c.seg_a * c.fps); ev = c.env[i0:i0 + max(2, int(c.D * c.fps))]
    nb = 120; bw = TW / nb
    for j in range(nb):
        v = float(ev[min(len(ev) - 1, int(j * len(ev) / nb))]) if len(ev) else 0.3
        hh = 6 + (wave_h - 16) * min(1, v)
        box = (j * bw + 1, wave_h / 2 - hh / 2, (j + 1) * bw - 2, wave_h / 2 + hh / 2)
        L.rrect((TX0 + box[0], wave_y + box[1], TX0 + box[2], wave_y + box[3]), 2, fill=(60, 214, 150, 110))
        Hi.rrect(box, 2, fill=(90, 240, 170, 255))
    for lab, yy in [("T1", ty + 30), ("V1", thumbs_y + thumbs_h / 2), ("A1", wave_y + wave_h / 2)]:
        L.rrect((TX0 - 84, yy - 22, TX0 - 24, yy + 22), 10, fill=wa(28)); L.text((TX0 - 54, yy), lab, "bold", 20, wa(200), anchor="mm")
    return L.np(), Hi.np()

@effect("timeline", "Video-editor timeline UI slides in on top: text, thumbnail and waveform tracks with a moving playhead and render progress.",
        defaults=dict(project="my-reel-final.mp4", clips=["Hook", "Title card", "Call to action"], tools=["Cut", "Captions", "Color", "Audio", "Export"]),
        main="project")
def fx_timeline(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("tl")
    x, y, w, h = 40, 110, 1000, 650; TX0, TW = 130, 840
    a = eo(seg(lt, 0, 0.55)); yy = y - 780 * (1 - a)
    if "static" not in st:
        hd = Layer(w, 200)
        for j in range(3): hd.circle(36 + j * 28, 38, 9, fill=wa(70 + j * 30))
        hd.text((140, 38), p.project, "semi", 28, wa(230), anchor="lm")
        hd.text((w - 36, 132), f"1080 × 1920 · {int(c.fps)} fps", "regular", 22, wa(140), anchor="rm")
        st["hd"] = hd.np()
        st["static"], st["hi"] = timeline_static(c, w, TX0, TW, 330, 110, 460, 90, p.clips, max(1, int(round(c.D))))
    glass(F, x, yy, w, h, r=36, op=a)
    over(F, st["hd"], x, yy, a); over(F, st["static"], x, yy, a)
    ph = clamp01(lt / c.D); px = TX0 + TW * ph
    hi = st["hi"][:, :max(1, int(TW * ph))]
    over(F, hi, x + TX0, yy + 460, a)
    D = Layer(w, 120)
    fr = int((lt % 1) * c.fps)
    D.text((36, 70), f"00:00:{int(lt):02d}:{fr:02d}", "mono", 44, WHITE)
    pct = int(100 * ph)
    D.rrect((w - 290, 16, w - 30, 60), 22, fill=wa(30))
    D.rrect((w - 290, 16, w - 290 + 260 * ph, 60), 22, fill=(124, 245, 196, 90))
    D.text((w - 160, 38), f"Rendering {pct}%", "semi", 22, WHITE, anchor="mm")
    over(F, D.np(), x, yy, a)
    T = Layer(w, 70); xx = 36; act = int(lt * 1.3) % len(p.tools)
    for j, tl in enumerate(p.tools):
        tw_ = text_w(tl, "semi", 22) + 40
        T.rrect((xx, 6, xx + tw_, 58), 26, fill=(124, 245, 196, 255) if j == act else wa(26))
        T.text((xx + tw_ / 2, 32), tl, "semi", 22, (16, 16, 20, 255) if j == act else wa(210), anchor="mm"); xx += tw_ + 14
    over(F, T.np(), x, yy + 568, a)
    PH = Layer(30, 360); PH.rrect((3, 0, 27, 30), 8, fill=(255, 69, 58, 255)); PH.rect((13.5, 24, 16.5, 360), fill=(255, 69, 58, 255))
    over(F, PH.np(), x + px - 15, yy + 192, a)
    return F


DARK = (28, 28, 30, 255)
def da(a): return (28, 28, 30, int(a))

def cover_art(k, n=88):
    c1 = PALETTE[k % len(PALETTE)]; c2 = PALETTE[(k * 3 + 2) % len(PALETTE)]
    return grad(n, n, c1, c2, angle=35 + 40 * (k % 3))

def grad_ring_layer(r, width, c1=(255, 204, 0), c2=(255, 0, 128)):
    size = int(2 * r + 4)
    g = grad(size, size, c1, c2, angle=225)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32); d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    a = np.clip(r - d + 0.5, 0, 1) * np.clip(d - (r - width) + 0.5, 0, 1)
    return to_layer(g, a)

def round_img(arr, d):
    im = cv2.resize(arr.astype(np.float32), (d, d), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32); r = d / 2
    return to_layer(im, np.clip(r - np.sqrt((xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2) + 0.5, 0, 1))

# ================================================================== 8. playlist-scroll ====
@effect("playlist-scroll", "Music-app style 'Up next' playlist card scrolls through tracks; the playing track gets live equaliser bars.",
        defaults=dict(kicker="UP NEXT", title="Late night edits", accent="#B388FF",
                      tracks=[["Golden hour", "Studio mix", "3:12"], ["Night drive", "Lo-fi tapes", "2:48"], ["Paper planes", "The Weekenders", "3:41"],
                              ["Slow mornings", "Coastline", "2:59"], ["Neon rain", "Midnight FM", "3:20"], ["First light", "Northbound", "4:02"],
                              ["Static hearts", "Analog Club", "3:05"], ["Blue hour", "Studio mix", "2:37"], ["Daydreamer", "Polaroid", "3:33"],
                              ["Afterglow", "Late Bloom", "3:18"]]), main="title")
def fx_playlist(F, O, c):
    p, lt = c.cfg, c.lt; acc = rgb(p.accent); st = c.st("pl"); tr = p.tracks; n = len(tr)
    x, y, w, h = 60, 1060, 960, 770
    a = seg(lt, 0, 0.5); yy = y + 110 * (1 - eo(a)); op = eo(a)
    if "covers" not in st:
        st["covers"] = [cover_art(k, 64) for k in range(n)]
        Hd = Layer(w, 150)
        Hd.text((44, 34), p.kicker, "semi", 22, wa(160), track=4); Hd.text((44, 66), p.title, "bold", 46, WHITE)
        Hd.text((w - 44, 84), f"{n} tracks", "regular", 24, wa(150), anchor="rm"); st["hd"] = Hd.np()
    glass(F, x, yy, w, h, r=44, op=op)
    over(F, st["hd"], x, yy, op)
    VH, RH = h - 170, 98
    off = max(0.0, lt - 0.4) * 0.85 * RH
    L = Layer(w, VH)
    j0 = int(off // RH) - 1
    cur = int((off + VH * 0.3) // RH)
    for j in range(j0, j0 + VH // RH + 3):
        ry = j * RH - off
        if ry < -RH or ry > VH: continue
        k = j % n; t_, ar, du = tr[k]
        is_cur = j == cur
        if is_cur: L.rrect((24, ry + 6, w - 24, ry + RH - 6), 20, fill=wa(22))
        if is_cur:
            for b in range(3):
                v = c.env[max(0, c.i - b * 3)] if len(c.env) else 0.5
                bh = 10 + 30 * min(1, v) * (0.6 + 0.4 * math.sin(lt * 9 + b * 2))
                L.rrect((46 + b * 11, ry + RH / 2 + 18 - bh, 53 + b * 11, ry + RH / 2 + 18), 3, fill=acc + (255,))
        else:
            L.text((62, ry + RH / 2), str(k + 1), "medium", 26, wa(150), anchor="mm")
        L.paste(st["covers"][k], 100, ry + 17, mask_r=12)
        L.text((190, ry + 20), t_, "semi", 32, acc + (255,) if is_cur else WHITE)
        L.text((190, ry + 58), ar, "regular", 24, wa(160))
        L.text((w - 48, ry + RH / 2), du, "regular", 24, wa(150), anchor="rm")
    V = L.np()
    ramp = np.ones(VH, np.float32); f_ = 70
    ramp[:f_] = np.linspace(0, 1, f_); ramp[-f_:] = np.linspace(1, 0, f_)
    V[..., 3] *= ramp[:, None]
    over(F, V, x, yy + 150, op)
    return F

# ================================================================== 9. tech-widgets ====
@effect("tech-widgets", "Phone home-screen style widgets: calendar, reel views with sparkline, new followers and tappable quick-action buttons.",
        defaults=dict(date="", event="Shoot reel", event_time="6:00 PM", views=128000, views_change="24% this week",
                      followers=1204, followers_label="new followers since yesterday", buttons=["Record", "Edit", "Post"],
                      accent="#0A84FF"), main="event")
def fx_tech_widgets(F, O, c):
    p, lt = c.cfg, c.lt; acc = rgb(p.accent)
    try: d = datetime.date.fromisoformat(p.date) if p.date else datetime.date.today()
    except Exception: d = datetime.date.today()
    def pop(dl):
        a = seg(lt, dl, 0.5); return a, 0.8 + 0.2 * eback(a), eo(a * 1.4)
    # calendar
    a, s, op = pop(0.1)
    if a > 0:
        x, y, w, h = 50, 170, 310, 310; glass(F, x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s, r=48 * s, op=op, tone="light")
        L = Layer(w, h)
        L.text((30, 30), d.strftime("%A").upper(), "bold", 24, (255, 59, 48, 255), track=1)
        L.text((26, 60), str(d.day), "black", 112, DARK)
        L.rrect((30, 210, 36, 270), 3, fill=(150, 110, 255, 255))
        L.text((50, 208), p.event, "semi", 26, DARK); L.text((50, 244), p.event_time, "regular", 22, da(140))
        over_c(F, L.np(), x + w / 2, y + h / 2, op, s)
    # views
    a, s, op = pop(0.3)
    if a > 0:
        x, y, w, h = 720, 170, 310, 310; glass(F, x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s, r=48 * s, op=op, tone="light")
        L = Layer(w, h); g = eo(seg(lt, 0.4, 1.3))
        L.text((30, 30), "Reel views", "semi", 24, da(150))
        L.text((26, 62), fmt_num(p.views * g), "black", 76, DARK)
        L.poly([(30, 182), (46, 182), (38, 168)], fill=(52, 199, 89, 255))
        L.text((54, 164), p.views_change, "semi", 22, (52, 199, 89, 255))
        vals = [12, 18, 15, 26, 22, 34, 31, 46]; pts = [(30 + i * (w - 60) / 7, 285 - v * 1.5) for i, v in enumerate(vals)]
        k = g * (len(pts) - 1); ii = int(k); fr = k - ii
        sub = pts[:ii + 1] + ([(lerp(pts[ii][0], pts[ii + 1][0], fr), lerp(pts[ii][1], pts[ii + 1][1], fr))] if ii + 1 < len(pts) else [])
        if len(sub) >= 2: L.line(sub, acc + (255,), 5)
        over_c(F, L.np(), x + w / 2, y + h / 2, op, s)
    # followers
    a, s, op = pop(0.6)
    if a > 0:
        x, y, w, h = 90, 1290, 900, 200; glass(F, x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s, r=48 * s, op=op, tone="light")
        L = Layer(w, h)
        for j, ch in enumerate("ARSKM"):
            cx = 70 + j * 40; L.circle(cx, 100, 38, fill=PALETTE[j] + (255,), outline=(255, 255, 255, 255), width=4)
            L.text((cx, 100), ch, "bold", 28, WHITE, anchor="mm")
        L.text((300, 46), "+" + commas(p.followers * eo(seg(lt, 0.7, 1.3))), "black", 52, DARK)
        L.text((302, 118), p.followers_label, "regular", 26, da(150))
        over_c(F, L.np(), x + w / 2, y + h / 2, op, s)
    # buttons
    for j, lab in enumerate(p.buttons[:3]):
        a, s, op = pop(0.9 + 0.12 * j)
        if a <= 0: continue
        tp = 1.6 + 0.45 * j; press = 1 - 0.08 * math.sin(math.pi * seg(lt, tp, 0.18)) if lt < tp + 0.18 else 1
        on = lt >= tp + 0.09
        x, y, w, h = 90 + j * 315, 1540, 270, 100; s *= press
        glass(F, x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s, r=50 * s, op=op, tone="light", tint=p.accent if on else None)
        L = Layer(w, h)
        if on: L.rrect((0, 0, w, h), 50, fill=acc + (235,))
        L.circle(52, 50, 18, fill=WHITE if on else acc + (255,))
        L.text((86, 50), lab, "semi", 30, WHITE if on else DARK, anchor="lm")
        over_c(F, L.np(), x + w / 2, y + h / 2, op, s)
    return F

# ================================================================== 10. glass-tunnel ====
@effect("glass-tunnel", "The speaker stands inside a 3D tunnel of glass video panels flying toward the camera.", mask=True,
        defaults=dict(rings=6, per_ring=10, speed=1.1))
def fx_glass_tunnel(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("tun")
    if "panels" not in st:
        fr = c.frames(8, 240, 320); st["panels"] = []
        for k, im in enumerate(fr):
            g = im.astype(np.float32) * 0.85 + 30
            st["panels"].append(rounded_frame(g, 26, 3, (255, 255, 255)))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        st["glow"] = np.clip(1 - np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.45) / H) ** 2) * 2.2, 0, 1)[..., None]
    m = c.mask[..., None]; a = eo(seg(lt, 0, 0.6))
    bg = O * (1 - 0.72 * a) + st["glow"] * np.array([90, 120, 255], np.float32) * 0.35 * a
    cx, cy = c.subj.hx, c.subj.hy + c.subj.r * 1.2
    R, span = 700, 5.4
    items = []
    for ring in range(p.rings):
        z = ((ring * span / p.rings - lt * p.speed) % span) + 0.55
        for k in range(p.per_ring):
            ang = k * 2 * math.pi / p.per_ring + ring * 0.35 + lt * 0.12
            items.append((z, ang, (ring * p.per_ring + k) % 8))
    items.sort(key=lambda t: -t[0])
    for z, ang, pk in items:
        op = clamp01((span + 0.3 - z) / 1.6) * clamp01((z - 0.55) / 0.5) * a
        if op < 0.02: continue
        x = cx + R * math.cos(ang) / z; y = cy + R * 1.45 * math.sin(ang) / z; sc = 0.95 / z
        if not (-400 < x < W + 400 and -500 < y < H + 500): continue
        dg = (-math.degrees(ang)) % 180
        if dg > 90: dg -= 180
        over_c(bg, st["panels"][pk], x, y, op=op * 0.92, scale=sc, angle=dg)
    return bg * (1 - m) + O * m

# ================================================================== 11. notes ====
@effect("notes", "Notes-app style to-do card ('Tomorrow's reel') checks itself off while you talk.",
        defaults=dict(app="Notes", title="Tomorrow's reel", date="", items=["Write the hook", "Film in natural light", "Add captions", "Post at 7 PM"],
                      accent="#FFB000"), main="items")
def fx_notes(F, O, c):
    p, lt = c.cfg, c.lt; acc = rgb(p.accent); n = len(p.items)
    try: d = datetime.date.fromisoformat(p.date) if p.date else datetime.date.today()
    except Exception: d = datetime.date.today()
    w, h = 900, 236 + 86 * n + 30; x, y = 90, H - h - 170
    a = seg(lt, 0, 0.6); e = eback(a); op = eo(a * 1.5)
    yy = y + 260 * (1 - e)
    if op > 0.05: shadow(F, x, yy, w, h, 44, op, strength=0.45, dy=22)
    t0 = 0.8; per = max(0.3, (c.D - t0 - 0.5) / n)
    L = Layer(w, h)
    L.rrect((0, 0, w, h), 44, fill=(252, 251, 248, 255))
    L.text((40, 32), "‹ " + p.app, "semi", 28, acc + (255,))
    L.text((44, 86), p.title, "bold", 54, DARK)
    L.text((44, 160), f"{d.day} {d.strftime('%B %Y')}", "regular", 26, da(130))
    for j, it in enumerate(p.items):
        ry = 230 + j * 86; cx, cy = 70, ry + 30; te = t0 + per * (j + 1) - 0.2
        tw_ = text_w(it, "regular", 38)
        if lt >= te:
            k = eback(seg(lt, te, 0.3))
            L.circle(cx, cy, 21 * k, fill=acc + (255,)); check_path(L, cx, cy, 26, seg(lt, te + 0.05, 0.25), WHITE, 4)
            L.text((110, ry + 8), it, "regular", 38, da(120))
            sp = eo(seg(lt, te + 0.1, 0.35))
            if sp > 0: L.rect((108, ry + 32, 108 + (tw_ + 6) * sp, ry + 35), fill=da(150))
        else:
            L.circle(cx, cy, 20, outline=acc + (255,), width=3)
            L.text((110, ry + 8), it, "regular", 38, DARK)
    over_c(F, L.np(), x + w / 2, yy + h / 2, op, angle=-4 * (1 - eo(a)))
    return F

# ================================================================== 12. story-portal ====
@effect("story-portal", "Social 'stories' row appears, a story bubble is tapped and expands like a portal into the next shot.",
        defaults=dict(names=["you", "studio", "travel", "food", "daily"], pick=2, tap_at=0.9), main="names")
def fx_story_portal(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("portal"); n = min(5, len(p.names))
    xs = [W / 2 + (j - (n - 1) / 2) * 206 for j in range(n)]; sy = 250; R = 78
    if "ui" not in st:
        av = c.frames(n, 200, 200, key="avatars")
        L = Layer(W, 420)
        L.text((60, 70), "Stories", "bold", 44, WHITE)
        for j in range(n): L.text((xs[j], sy + R + 34), p.names[j], "medium", 26, WHITE, anchor="mm")
        UI = L.np()
        ring = grad_ring_layer(R, 7)
        for j in range(n):
            over_c(UI, ring, xs[j], sy - 0)
            over_c(UI, round_img(av[j], int(2 * R - 22)), xs[j], sy)
        st["ui"] = UI; st["ring"] = ring
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); st["xx"], st["yy"] = xx, yy
        st["av"] = round_img(av[min(p.pick, n - 1)], 64)
    ta = p.tap_at; q = seg(lt, ta + 0.12, 0.95); qe = q ** 1.7
    B = fast_blur(O, 22) * 0.42
    over(B, st["ui"], 0, 0, eo(seg(lt, 0, 0.4)) * (1 - eo(q * 1.6)))
    px, py = xs[min(p.pick, n - 1)], sy
    if lt < ta + 0.12:
        F = B
        if lt > ta - 0.35:
            k = seg(lt, ta - 0.35, 0.35)
            Cu = Layer(90, 110); Cu.circle(45, 45, 34 * (1 - 0.25 * math.sin(math.pi * seg(lt, ta, 0.2))), fill=wa(200)); C = Cu.np()
            over_c(F, C, lerp(px + 200, px, eo(k)), lerp(py + 380, py + 10, eo(k)))
        if lt >= ta:
            rr = Layer(260, 260); rp = seg(lt, ta, 0.4); rr.circle(130, 130, 40 + 80 * rp, outline=wa(255 * (1 - rp)), width=4)
            over_c(F, rr.np(), px, py)
        return F
    rad = lerp(R - 8, 2300, qe)
    d = np.sqrt((st["xx"] - px) ** 2 + (st["yy"] - py) ** 2)
    mk = np.clip(rad - d + 0.5, 0, 1)[..., None]
    F = B * (1 - mk) + O * mk
    if rad < 2100:
        ring = grad_ring_layer(rad + 10, 14) if rad < 900 else None
        if ring is not None: over_c(F, ring, px, py, op=1 - q)
    hp = seg(lt, ta + 1.0, 0.4)
    if hp > 0:
        Hd = Layer(W, 180)
        for j in range(3):
            x0 = 30 + j * 345; Hd.rrect((x0, 30, x0 + 330, 36), 3, fill=wa(90))
            fill = 1.0 if j == 0 else clamp01((lt - ta - 1.0) / 3.0) if j == 1 else 0
            if fill > 0: Hd.rrect((x0, 30, x0 + 330 * fill, 36), 3, fill=WHITE)
        Hd.text((118, 92), p.names[min(p.pick, n - 1)], "semi", 30, WHITE, anchor="lm", stroke=1, stroke_fill=(0, 0, 0, 90))
        Hd.text((118 + text_w(p.names[min(p.pick, n - 1)], "semi", 30) + 14, 92), "2h", "regular", 28, wa(200), anchor="lm")
        Hp = Hd.np(); over(F, Hp, 0, 0, eo(hp)); over_c(F, st["av"], 72, 92, eo(hp))
    return F

# ================================================================== 13. headline ====
def sticker_pill(text, size, bg, fg):
    tw_ = text_w(text, "black", size); pw, ph = tw_ + size * 0.9, size * 1.45; pad = 16
    L = Layer(pw + pad * 2 + 10, ph + pad * 2 + 12)
    L.rrect((pad + 10, pad + 12, pad + 10 + pw, pad + 12 + ph), 26, fill=(18, 18, 22, 255))
    L.rrect((pad - 6, pad - 6, pad + pw + 6, pad + ph + 6), 32, fill=WHITE)
    L.rrect((pad, pad, pad + pw, pad + ph), 26, fill=rgba(bg))
    L.text((pad + pw / 2, pad + ph / 2), text, "black", size, rgba(fg), anchor="mm")
    return L.np()

def shape_layer(kind, size, col):
    L = Layer(size, size); s = size; colA = rgba(col)
    if kind == "star": L.poly(star_pts(s / 2, s / 2, s / 2, s * 0.14, 4), fill=colA)
    elif kind == "heart":
        L.circle(s * 0.32, s * 0.36, s * 0.22, fill=colA); L.circle(s * 0.68, s * 0.36, s * 0.22, fill=colA)
        L.poly([(s * 0.12, s * 0.44), (s * 0.88, s * 0.44), (s * 0.5, s * 0.9)], fill=colA)
    elif kind == "bolt":
        L.poly([(s * .58, s * .02), (s * .18, s * .56), (s * .46, s * .56), (s * .36, s * .98), (s * .82, s * .40), (s * .54, s * .40)], fill=colA)
    elif kind == "burst":
        L.poly(star_pts(s / 2, s / 2, s / 2, s * 0.38, 12), fill=colA)
        L.text((s / 2, s / 2), "NEW", "black", s * 0.26, (18, 18, 22, 255), anchor="mm")
    elif kind == "cursor":
        pts = [(s * .1, s * .05), (s * .1, s * .82), (s * .3, s * .64), (s * .45, s * .95), (s * .58, s * .89), (s * .44, s * .58), (s * .72, s * .58)]
        L.poly(pts, fill=(18, 18, 22, 255)); L.poly([(x + (s * .5 - x) * 0.18, y + (s * .5 - y) * 0.12) for x, y in pts], fill=WHITE)
    return L.np()

@effect("headline", "Loud sticker-style title cards pop in with sparkles, stickers and a clicking cursor (yap-style content).",
        defaults=dict(lines=["HOT TAKE", "watch till the end"], colors=[["#FF4D8D", "#FFFFFF"], ["#FFD60A", "#141414"], ["#5AC8FA", "#0A1428"]]),
        main="lines")
def fx_headline(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("yap")
    if "pills" not in st:
        st["pills"] = []
        for j, ln in enumerate(p.lines[:3]):
            size = fit_size([ln], "black", 820, 96 if j == 0 else 64)
            bg, fg = p.colors[j % len(p.colors)]; st["pills"].append(sticker_pill(ln, size, bg, fg))
        st["shapes"] = [shape_layer(k, s_, col) for k, s_, col in
                        [("star", 70, (255, 230, 90)), ("star", 46, (255, 255, 255)), ("heart", 84, (255, 77, 141)),
                         ("bolt", 90, (255, 214, 10)), ("burst", 130, (124, 245, 196)), ("star", 56, (90, 200, 250))]]
        st["cursor"] = shape_layer("cursor", 90, (0, 0, 0))
    cy = 300; angs = [-4, 3, -2]; pos = []
    for j, pl in enumerate(st["pills"]):
        pos.append((W / 2 + (-30 if j % 2 == 0 else 40), cy)); cy += pl.shape[0] * 0.78
    click = 1.15
    starts = [0.15, click, click + 0.35]
    for j, pl in enumerate(st["pills"]):
        a = seg(lt, starts[j], 0.45)
        if a <= 0: continue
        sc = 0.2 + 0.8 * eback(a, 2.6); ang = angs[j % 3] * (1 + 2 * (1 - eo(a))) + 1.2 * math.sin(lt * 2.2 + j)
        over_c(F, pl, pos[j][0], pos[j][1], op=eo(a * 2), scale=sc, angle=ang)
    sp = [(150, 170, 0, 0.5), (930, 250, 1, 0.6), (120, 520, 3, 0.8), (960, 560, 2, 1.0), (880, 110, 4, 0.4), (250, 640, 5, 1.2)]
    for x, y, k, dl in sp:
        a = seg(lt, dl, 0.4)
        if a <= 0: continue
        tw_ = 0.75 + 0.25 * math.sin(lt * 4 + k) if k in (0, 1, 5) else 1
        over_c(F, st["shapes"][k], x, y + 8 * math.sin(lt * 2 + k), op=eo(a), scale=eback(a) * tw_, angle=10 * math.sin(lt * 1.5 + k))
    tx, ty = pos[1] if len(pos) > 1 else pos[0]; tx += 120; ty += 10
    if lt < click + 1.2:
        k = eo(seg(lt, 0.35, click - 0.35))
        cx_, cy_ = lerp(W + 80, tx, k), lerp(H * 0.62, ty, k)
        press = 1 - 0.18 * math.sin(math.pi * seg(lt, click - 0.05, 0.2))
        over_c(F, st["cursor"], cx_ + 30, cy_ + 34, op=1 - seg(lt, click + 0.8, 0.4), scale=press)
        if lt > click:
            rr = Layer(200, 200); q = seg(lt, click, 0.45)
            rr.circle(100, 100, 20 + 70 * q, outline=wa(255 * (1 - q)), width=5); over_c(F, rr.np(), tx, ty)
    return F

# ================================================================== 14. pull-back ====
@effect("pull-back", "Camera pulls back to reveal the clip is being edited inside an editor dashboard (tools, layers, timeline).",
        defaults=dict(status="Auto edit in progress", project="my-reel-final.mp4", tools=["Trim", "Text", "Color", "Audio", "FX"],
                      clips=["Hook", "Title card", "Call to action"], accent="#7CF5C4"), main="status")
def fx_pull_back(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("pb"); acc = rgb(p.accent)
    sc_end = 0.6; vw, vh = int(W * sc_end), int(H * sc_end); vx, vy = (W - vw) // 2, 150
    if "ui" not in st:
        U = grad(W, H, (16, 17, 22), (8, 8, 11))
        L = Layer(W, 1400)
        L.text((W / 2, vy + vh + 40), p.project, "medium", 24, wa(150), anchor="mt")
        for j, t_ in enumerate(p.tools[:5]):
            y = 230 + j * 170
            L.rrect((50, y, 160, y + 110), 26, fill=wa(18), outline=wa(40), width=2)
            cxx, cyy = 105, y + 50; col = PALETTE[j % len(PALETTE)] + (255,)
            [lambda: L.rect((cxx - 26, cyy - 4, cxx + 26, cyy + 4), col), lambda: L.text((cxx, cyy), "T", "black", 50, col, anchor="mm"),
             lambda: L.circle(cxx, cyy, 24, fill=col), lambda: [L.rrect((cxx - 24 + b * 14, cyy - 6 - b * 7, cxx - 14 + b * 14, cyy + 18), 3, fill=col) for b in range(4)],
             lambda: L.poly(star_pts(cxx, cyy, 28, 11, 5), fill=col)][j % 5]()
            L.text((105, y + 92), t_, "medium", 18, wa(170), anchor="mm")
        L.text((W - 105, 232), "Layers", "semi", 22, wa(170), anchor="mm")
        U2 = U.copy(); over(U2, L.np(), 0, 0)
        for j, im in enumerate(c.frames(3, 150, 190, key="layers")):
            over(U2, rounded_frame(im.astype(np.float32), 14, 2, (255, 255, 255)), W - 180, 270 + j * 230)
            Ll = Layer(160, 30); Ll.text((80, 0), f"Layer {j + 1}", "medium", 18, wa(150), anchor="mt"); over(U2, Ll.np(), W - 185, 470 + j * 230)
        tl, hi = timeline_static(c, W - 40, 120, W - 200, 170, 100, 290, 80, p.clips, max(1, int(round(c.D))))
        over(U2, tl, 20, 1400)
        st["ui"], st["hi"] = U2, hi
    q = eio(seg(lt, 0.3, 1.0)); s = lerp(1.0, sc_end, q)
    cw, ch = int(W * s), int(H * s); cx = W / 2; cy = lerp(H / 2, vy + vh / 2, q)
    F = st["ui"].copy()
    ph = clamp01(lt / c.D); TW = W - 200
    over(F, st["hi"][:, :max(1, int(TW * ph))], 20 + 120, 1400 + 290)
    D = Layer(W, 130)
    pu = 0.5 + 0.5 * math.sin(lt * 6)
    D.rrect((36, 40, 36 + text_w(p.status, "semi", 26) + 90, 104), 32, fill=wa(22))
    D.circle(70, 72, 10 + 5 * pu, fill=acc + (int(90 * (1 - pu)),)); D.circle(70, 72, 9, fill=acc + (255,))
    D.text((92, 72), p.status, "semi", 26, WHITE, anchor="lm")
    D.rrect((W - 190, 44, W - 36, 104), 30, fill=acc + (255,)); D.text((W - 113, 74), "Export", "bold", 26, (16, 16, 20, 255), anchor="mm")
    over(F, D.np(), 0, 0)
    PH = Layer(26, 320); PH.rrect((2, 0, 24, 26), 7, fill=(255, 69, 58, 255)); PH.rect((11.5, 20, 14.5, 320), fill=(255, 69, 58, 255))
    over(F, PH.np(), 20 + 120 + TW * ph - 13, 1400 + 50)
    V = cv2.resize(O, (cw, ch), interpolation=cv2.INTER_AREA)
    r = int(lerp(0, 36, q))
    if q > 0.01: shadow(F, cx - cw / 2, cy - ch / 2, cw, ch, r, q, strength=0.6, dy=20)
    over(F, rounded_frame(V, max(1, r), 3 if q > 0.5 else 0, acc), cx - cw / 2, cy - ch / 2)
    return F

# ================================================================== 15. carousel-spin ====
@effect("carousel-spin", "A ring of video cards spins behind the speaker like a carousel and eases to a stop.", mask=True,
        defaults=dict(cards=8, images=[]))
def fx_carousel(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("car"); n = p.cards
    if "cards" not in st:
        if p.images:
            imgs = [load_img(f) for f in p.images]; imgs = [crop_cover(i.astype(np.float32), 280, 380) for i in imgs if i is not None]
        else:
            imgs = [im.astype(np.float32) for im in c.frames(n, 280, 380)]
        st["cards"] = [rounded_frame(imgs[k % len(imgs)], 28, 4, (255, 255, 255)) for k in range(n)]
    m = c.mask[..., None]; a = eo(seg(lt, 0, 0.6))
    bg = O * (1 - 0.45 * a)
    cx, cy = c.subj.hx, c.subj.hy + c.subj.r * 0.6
    spin = 7.5 * eo(seg(lt, 0, 2.6)) + 0.25 * lt
    items = []
    for k in range(n):
        th = spin + k * 2 * math.pi / n; z = math.sin(th)
        items.append((z, cx + 500 * math.cos(th), cy - 80 * z, k, th))
    items.sort()
    for z, x, y, k, th in items:
        card = st["cards"][k]
        sc = (0.72 + 0.38 * (z + 1) / 2) * (0.6 + 0.4 * a)
        if z < 0: card = card.copy(); card[..., :3] *= 0.55 + 0.45 * (z + 1)
        over_c(bg, card, x, y, op=a, scale=sc, sx=max(0.12, abs(math.sin(th))))
    return bg * (1 - m) + O * m

# ================================================================== 16. dark-stage ====
@effect("dark-stage", "Background becomes a dark concert stage: sweeping colour beams, haze particles and coloured rim light on the speaker.", mask=True,
        defaults=dict(color1="#8B5CF6", color2="#22D3EE"))
def fx_dark_stage(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("stage"); c1, c2 = np.array(rgb(p.color1), np.float32), np.array(rgb(p.color2), np.float32)
    if "base" not in st:
        st["base"] = grad(W, H, (6, 6, 10), (18, 14, 26))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        fl = np.clip(1 - np.sqrt(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H * 0.98) / (H * 0.16)) ** 2), 0, 1)
        st["floor"] = fl[..., None]
        L = 2600; beam = np.zeros((H, W), np.float32)
        pts = np.array([[W / 2 - 20, 0], [W / 2 + 20, 0], [W / 2 + 260, L], [W / 2 - 260, L]], np.int32)
        cv2.fillConvexPoly(beam, pts, 1.0); beam = cv2.GaussianBlur(beam, (0, 0), 30)
        ramp = np.clip(1 - yy / (H * 1.2), 0, 1); st["beam"] = beam * ramp
        rng = np.random.default_rng(5); st["pts"] = rng.random((90, 4))
    a = eo(seg(lt, 0, 0.6))
    keep = 0.10 if c.ai else 0.45
    bg = st["base"] * (1 - keep) + O * keep * 0.5
    for j, (col, ox, ph) in enumerate([(c1, -W * 0.38, 0.0), (c2, W * 0.38, 1.7)]):
        ang = 22 * math.sin(lt * 0.9 + ph) + (18 if j == 0 else -18)
        M = cv2.getRotationMatrix2D((W / 2, 0), ang, 1.0); M[0, 2] += ox
        b = cv2.warpAffine(st["beam"], M, (W, H))
        bg += b[..., None] * col * 0.55
    bg += st["floor"] * (c1 * 0.5 + c2 * 0.5) * 0.35
    P = np.zeros((H // 2, W // 2), np.float32)
    for x0, y0, sp, br in st["pts"]:
        x = (x0 * W / 2 + lt * 18 * (sp - 0.5)) % (W / 2); y = (y0 * H / 2 - lt * 22 * (0.4 + sp)) % (H / 2)
        cv2.circle(P, (int(x), int(y)), 1 + int(br * 2), 0.5 + 0.5 * br, -1, lineType=cv2.LINE_AA)
    bg += cv2.resize(cv2.GaussianBlur(P, (0, 0), 1.2), (W, H))[..., None] * 160
    m = c.mask[..., None]
    out = bg * (1 - m) + O * 0.92 * m
    if c.ai:
        ms = cv2.resize(c.mask, (W // 4, H // 4)); ring = np.clip(cv2.GaussianBlur(ms, (0, 0), 5) - ms, 0, 1) * 2.4
        ring = cv2.resize(ring, (W, H))[..., None]
        xr = (np.arange(W, dtype=np.float32) / W)[None, :, None]
        out += ring * (c1 * (1 - xr) + c2 * xr) * 0.9
    return O * (1 - a) + out * a

# ================================================================== 17. match-cut ====
@effect("match-cut", "Punch-in match cuts on the beat: hard cuts between wide and tight framings with a flash and camera shake.", mask=True,
        defaults=dict(beat=0.6, zooms=[1.0, 1.45, 1.2, 1.65]))
def fx_match_cut(F, O, c):
    p, lt = c.cfg, c.lt
    k = int(lt / p.beat); since = lt - k * p.beat
    z = p.zooms[k % len(p.zooms)]
    sh = 14 * math.sin(lt * 75) * math.exp(-since * 14) if k > 0 else 0
    F = crop_cover(O, W, H, c.subj.hx + sh, c.subj.hy + c.subj.r * 0.4, z)
    if k > 0:
        f = max(0.0, 1 - since / 0.09)
        if f > 0: F = F + (255 - F) * 0.55 * f
    return F


def smooth_curve(vals, n):
    xs = np.linspace(0, len(vals) - 1, n); i0 = np.floor(xs).astype(int); i1 = np.minimum(i0 + 1, len(vals) - 1)
    t = xs - i0; t = t * t * (3 - 2 * t)
    v = np.array(vals, np.float32)
    return v[i0] * (1 - t) + v[i1] * t

def wrap_text(s, kind, size, max_w):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if text_w(t, kind, size) <= max_w or not cur: cur = t
        else: lines.append(cur); cur = w_
    if cur: lines.append(cur)
    return lines

# ================================================================== 18. liquid-glass-graph ====
@effect("liquid-glass-graph", "Glass card with a growth graph that fills like liquid: wavy surface, rising bubbles, count-up number.",
        defaults=dict(title="Channel growth", value="+248%", sub="followers in 90 days", values=[8, 12, 10, 18, 24, 22, 34, 41, 56, 72],
                      accent="#5AC8FA"), main="title")
def fx_liquid_graph(F, O, c):
    p, lt = c.cfg, c.lt; acc = np.array(rgb(p.accent), np.float32); st = c.st("lg")
    x, y, w, h = 70, 1090, 940, 640; cw, ch, gx, gy = 860, 330, 40, 260
    a = seg(lt, 0, 0.5); op = eo(a); yy = y + 100 * (1 - eo(a))
    glass(F, x, yy, w, h, r=44, op=op)
    L = Layer(w, 250)
    L.text((44, 40), p.title, "semi", 30, wa(190))
    num = p.value
    try:
        tgt = float(''.join(ch_ for ch_ in num if ch_.isdigit() or ch_ == '.')); g = eo(seg(lt, 0.3, 1.6))
        num = num.replace(str(int(tgt)) if tgt == int(tgt) else str(tgt), str(int(tgt * g)))
    except Exception: pass
    L.text((40, 84), num, "black", 96, WHITE); L.text((44, 200), p.sub, "medium", 28, wa(180))
    over(F, L.np(), x, yy, op)
    if "yy" not in st:
        st["yy"], st["xx"] = np.mgrid[0:ch, 0:cw].astype(np.float32)
        rng = np.random.default_rng(2); st["bub"] = rng.random((18, 3))
    vals = smooth_curve(p.values, cw); mn, mx = min(p.values), max(p.values)
    base = ch - 20 - (vals - mn) / max(1e-6, mx - mn) * (ch - 60)
    prog = eo(seg(lt, 0.3, 1.8)); px = cw * prog
    xs = np.arange(cw, dtype=np.float32)
    surf = base + 5 * np.sin(xs / 38 + lt * 5) * eo(seg(lt, 0.4, 0.6))
    yyg, xxg = st["yy"], st["xx"]
    inside = ((yyg >= surf[None, :]) & (xxg <= px)).astype(np.float32)
    depth = np.clip((yyg - surf[None, :]) / ch, 0, 1)
    alpha = inside * (0.62 - 0.5 * depth)
    hl = inside * np.clip(1 - (yyg - surf[None, :]) / 14, 0, 1) * 0.6
    G = np.zeros((ch, cw, 4), np.float32)
    col = acc / 255
    G[..., :3] = col * (1 - hl[..., None]) + hl[..., None]
    G[..., 3] = np.clip(alpha + hl, 0, 1)
    for bx, by, bs in st["bub"]:
        bxp = bx * cw
        if bxp > px: continue
        r_ = 3 + bs * 6; byp = ch - ((lt * (40 + 60 * bs) + by * ch) % ch)
        if byp < surf[min(cw - 1, int(bxp))] + r_: continue
        cv2.circle(G, (int(bxp), int(byp)), int(r_), (1, 1, 1, 0.55), 2, lineType=cv2.LINE_AA)
    lm = np.zeros((ch, cw), np.float32)
    n_ = max(2, int(px))
    pts = np.stack([xs[:n_], surf[:n_]], 1).astype(np.int32)
    cv2.polylines(lm, [pts], False, 1.0, 4, lineType=cv2.LINE_AA)
    G[..., :3] = G[..., :3] * (1 - lm[..., None]) + lm[..., None]; G[..., 3] = np.maximum(G[..., 3], lm)
    over(F, G, x + gx, yy + gy, op)
    if prog > 0.02:
        ex, ey = x + gx + px, yy + gy + surf[min(cw - 1, int(px))]
        D = Layer(80, 80); pu = (lt * 1.5) % 1
        D.circle(40, 40, 12 + 26 * pu, fill=rgba(p.accent, 120 * (1 - pu))); D.circle(40, 40, 11, fill=WHITE)
        over_c(F, D.np(), ex, ey, op)
    return F

# ================================================================== 19. scrapbook ====
def paper_bg(seed=1):
    rng = np.random.default_rng(seed)
    base = grad(W, H, (236, 224, 200), (222, 205, 176), angle=60)
    n = cv2.resize(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (W, H))[..., None]
    fib = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None]
    return np.clip(base + n * 7 + fib * 5, 0, 255)

def tape(w, h, col):
    L = Layer(w, h); zz = []
    for i in range(9): zz.append((0 if i % 2 == 0 else 6, i * h / 8))
    pts = [(x_, y_) for x_, y_ in zz] + [(w - (0 if i % 2 == 0 else 6), (8 - i) * h / 8) for i in range(9)]
    L.poly(pts, fill=rgba(col, 175)); return L.np()

@effect("scrapbook", "Scrapbook collage: the clip plays inside a tilted polaroid on paper, with tape, extra snapshots, doodles and a handwritten caption.",
        defaults=dict(caption="behind the scenes", note="day 01", tape_colors=["#F7A8B8", "#9ED9C3", "#FFD66B"]), main="caption")
def fx_scrapbook(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("scrap")
    pw, ph, m_, mb = 700, 840, 34, 150
    if "bg" not in st:
        bg = paper_bg()
        snaps = c.frames(2, 300, 300, key="snaps")
        for j, (sx, sy, ang) in enumerate([(240, 330, 12), (860, 1580, -10)]):
            Lp = Layer(360, 420); Lp.rrect((0, 0, 360, 420), 6, fill=(250, 248, 242, 255)); Lp.paste(snaps[j], 30, 30)
            shadow(bg, sx - 180, sy - 210, 360, 420, 6, 1, 0.3, 14)
            over_c(bg, Lp.np(), sx, sy, angle=ang)
            over_c(bg, tape(150, 46, p.tape_colors[(j + 1) % 3]), sx + (-110 if j == 0 else 110), sy - 200, angle=-30 if j == 0 else 25)
        D = Layer(W, H)
        D.line([(760, 260), (840, 230), (900, 290), (860, 350)], (40, 40, 44, 230), 5)
        D.poly([(846, 362), (874, 336), (880, 372)], fill=(40, 40, 44, 230))
        sp_ = star_pts(170, 1660, 46, 18, 5, 0.2); D.line(sp_ + [sp_[0]], (230, 70, 90, 255), 5)
        D.text((150, 1500), p.note, "serifi", 54, (40, 40, 44, 255))
        over(bg, D.np(), 0, 0)
        st["bg"] = bg
        Fr = Layer(pw + 2 * m_, ph + m_ + mb); Fr.rrect((0, 0, pw + 2 * m_, ph + m_ + mb), 6, fill=(252, 250, 245, 255))
        Fr.text(((pw + 2 * m_) / 2, ph + m_ + mb / 2 - 4), p.caption, "serifi", 58, (38, 38, 44, 255), anchor="mm")
        st["frame"] = Fr.np(); st["t1"] = tape(180, 54, p.tape_colors[0]); st["t2"] = tape(180, 54, p.tape_colors[2])
    F = st["bg"].copy()
    a = seg(lt, 0, 0.7); ang = lerp(-14, -3, eback(a)) + 1.2 * math.sin(lt * 1.3)
    cx, cy = W / 2 + 10, lerp(-700, 900, eo(a))
    P = st["frame"].copy()
    vid = crop_cover(O, pw, ph, c.subj.hx, c.subj.hy + c.subj.r * 0.9) / 255.0
    P[m_:m_ + ph, m_:m_ + pw, :3] = vid; P[m_:m_ + ph, m_:m_ + pw, 3] = 1
    over(P, st["t1"], P.shape[1] / 2 - 90, -16)
    shadow(F, cx - P.shape[1] / 2, cy - P.shape[0] / 2, P.shape[1], P.shape[0], 8, eo(a), 0.4, 26)
    over_c(F, P, cx, cy, angle=ang)
    return F

# ================================================================== 20. messages ====
@effect("messages", "Chat bubbles pop in one by one (with a typing indicator) above the speaker, like a text conversation.",
        defaults=dict(thread=[["in", "are you posting today?"], ["out", "yes! editing it right now"], ["in", "what's it about?"],
                              ["out", "how I plan a week of content in one hour"], ["in", "send me the link when it's live"]],
                      out_color="#0A84FF", in_color="#E9E9EB"), main="thread")
def fx_messages(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("msg"); th = p.thread; n = len(th)
    if "b" not in st:
        st["b"] = []
        for side, txt in th:
            lines = wrap_text(txt, "medium", 36, 560)
            bw = max(text_w(l, "medium", 36) for l in lines) + 56; bh = 28 + 46 * len(lines)
            L = Layer(bw + 20, bh + 10); out = side == "out"
            L.rrect((0, 0, bw, bh), 30, fill=rgba(p.out_color if out else p.in_color))
            for j, l in enumerate(lines): L.text((28, 18 + j * 46), l, "medium", 36, WHITE if out else (20, 20, 22, 255))
            st["b"].append((L.np(), out))
        T = Layer(140, 76); T.rrect((0, 0, 140, 76), 38, fill=rgba(p.in_color)); st["typing"] = T
    per = max(0.5, (c.D - 0.6) / n)
    times = [0.3 + j * per + (0.45 if th[j][0] == "in" else 0) for j in range(n)]
    shown = [j for j in range(n) if lt >= times[j]]
    typing = None
    for j in range(n):
        if th[j][0] == "in" and times[j] - 0.45 <= lt < times[j]: typing = j
    stack = [st["b"][j][0].shape[0] + 16 for j in shown] + ([92] if typing is not None else [])
    total = sum(stack); y = max(150, 980 - total)
    for idx, j in enumerate(shown):
        L, out = st["b"][j]; a = seg(lt, times[j], 0.35)
        x = W - 60 - L.shape[1] + 20 if out else 60
        ox = x + (L.shape[1] if out else 0)
        sc = 0.6 + 0.4 * eback(a)
        cx = lerp(ox, x + L.shape[1] / 2, sc); cy = y + L.shape[0] / 2
        over_c(F, L, cx, cy, op=eo(a * 1.6), scale=sc)
        y += stack[idx]
    if typing is not None:
        T = Layer(140, 76); T.rrect((0, 0, 140, 76), 38, fill=rgba(p.in_color))
        for d_ in range(3):
            bb = 6 * math.sin(lt * 12 - d_ * 0.9)
            T.circle(38 + d_ * 32, 38 - max(0, bb), 9, fill=(140, 140, 146, 255))
        over(F, T.np(), 60, y)
    return F

# ================================================================== 21. location-pin ====
@effect("location-pin", "A map pin drops onto the speaker with a bounce and radar pulses, and a location card slides in.", mask=True,
        defaults=dict(place="My studio", detail="Hyderabad · Now", color="#FF3B30"), main="place")
def fx_location_pin(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("pin"); col = rgb(p.color)
    if "pin" not in st:
        L = Layer(120, 160)
        L.circle(60, 56, 52, fill=col + (255,)); L.poly([(16, 84), (104, 84), (60, 156)], fill=col + (255,))
        L.circle(60, 56, 20, fill=WHITE); st["pin"] = L.np()
    tipx = c.subj.hx; tipy = max(170, c.subj.top - 14)
    a = seg(lt, 0.1, 0.7)
    fall = -700 * (1 - ei(a))
    q_ = seg(lt, 0.8, 0.5); bounce = -60 * abs(math.sin(math.pi * q_ * 2)) * (1 - q_)
    y_tip = tipy + fall + bounce
    for k in range(3):
        q = ((lt - 0.8) * 0.8 + k / 3) % 1
        if lt < 0.8: break
        R_ = Layer(400, 140); R_.ellipse((200 - 180 * q, 70 - 60 * q, 200 + 180 * q, 70 + 60 * q), outline=col + (int(220 * (1 - q)),), width=4)
        over_c(F, R_.np(), tipx, tipy)
    if lt > 0.75:
        Sd = Layer(120, 40); Sd.ellipse((20, 10, 100, 30), fill=(0, 0, 0, 90)); over_c(F, Sd.np(), tipx, tipy + 4)
    over(F, st["pin"], tipx - 60, y_tip - 156, eo(seg(lt, 0.1, 0.2)))
    b = seg(lt, 1.0, 0.5)
    if b > 0:
        w, h = 760, 150; x = (W - w) / 2; y = 1560 + 120 * (1 - eo(b))
        glass(F, x, y, w, h, r=40, op=eo(b), tone="light")
        L = Layer(w, h); L.circle(80, 75, 40, fill=col + (255,)); L.circle(80, 68, 14, fill=WHITE)
        L.poly([(70, 78), (90, 78), (80, 100)], fill=WHITE)
        L.text((146, 42), p.place, "bold", 40, DARK); L.text((148, 96), p.detail, "regular", 28, da(150))
        over(F, L.np(), x, y, eo(b))
    return F

# ================================================================== 22. glass-type ====
@effect("glass-type", "Giant frosted-glass lettering behind the speaker that refracts the room, with a moving light glint.", mask=True,
        defaults=dict(text="CREATE"), main="text")
def fx_glass_type(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("gt")
    if "M" not in st:
        t = p.text.upper(); size = fit_size([t], "black", 1010, 330)
        L = Layer(W, size * 1.25); L.text((W / 2, size * 0.62), t, "black", size, WHITE, anchor="mm")
        st["M"] = L.np()[..., 3]; st["h"] = st["M"].shape[0]
        er = cv2.erode(st["M"], np.ones((5, 5), np.uint8)); st["edge"] = np.clip(st["M"] - er, 0, 1)
        st["inner"] = np.clip(st["M"] - shift_img(st["M"], 6, 10), 0, 1)
        xx = np.arange(W, dtype=np.float32)[None, :]; yy = np.arange(st["h"], dtype=np.float32)[:, None]
        st["diag"] = xx + yy * 0.6
    a = seg(lt, 0, 0.8); hh = st["h"]
    y0 = int(np.clip(c.subj.hy - c.subj.r * 0.4 - hh / 2 + 80 * (1 - eo(a)), 0, H - hh))
    reg = F[y0:y0 + hh]
    B = fast_blur(O[y0:y0 + hh], 16) * 1.12 + 28
    M = st["M"][..., None] * eo(a)
    G = B * (1 - st["inner"][..., None] * 0.35)
    gx = (lt * 700) % (W + 900) - 450
    glint = np.clip(1 - np.abs(st["diag"] - gx) / 60, 0, 1)[..., None] * 0.55
    G = G + (255 - G) * glint
    G = G * (1 - st["edge"][..., None] * 0.85) + 255 * st["edge"][..., None] * 0.85
    reg[:] = reg * (1 - M) + G * M
    m = c.mask[..., None]
    return F * (1 - m) + O * m

# ================================================================== 23. place-in-grid ====
@effect("place-in-grid", "The clip shrinks into a tile of a profile grid, revealing the profile header and other posts.", mask=True,
        defaults=dict(name="Your Name", handle="@yourhandle", posts="128", followers="24.6K", following="312",
                      bio="Short videos about making short videos.", cell=1), main="name")
def fx_place_in_grid(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("grid")
    gap, cw = 4, (W - 8) / 3; chh = cw * 4 / 3; gy = 700
    cells = [(j % 3 * (cw + gap), gy + j // 3 * (chh + gap)) for j in range(9)]
    if "ui" not in st:
        U = np.zeros((H, W, 3), np.float32) + 6
        L = Layer(W, gy)
        L.text((W / 2, 70), p.handle, "bold", 34, WHITE, anchor="mm")
        for k_, (v, lab) in enumerate([(p.posts, "posts"), (p.followers, "followers"), (p.following, "following")]):
            xx = 450 + k_ * 205; L.text((xx, 210), v, "bold", 40, WHITE, anchor="mm"); L.text((xx, 256), lab, "regular", 26, wa(190), anchor="mm")
        L.text((60, 360), p.name, "semi", 32, WHITE); L.text((60, 406), p.bio, "regular", 28, wa(210))
        for k_, lab in enumerate(["Following", "Message"]):
            x0 = 60 + k_ * 490; L.rrect((x0, 470, x0 + 470, 540), 16, fill=wa(40)); L.text((x0 + 235, 505), lab, "semi", 28, WHITE, anchor="mm")
        for gx_ in (W / 2 - 160, W / 2 + 160): L.rect((gx_ - 20, 630, gx_ + 20, 634), fill=wa(255 if gx_ < W / 2 else 90))
        L.rect((0, 660, W / 2, 663), fill=WHITE)
        over(U, L.np(), 0, 0)
        av = c.frames(1, 240, 240, key="av")[0]
        over_c(U, grad_ring_layer(100, 7), 190, 230); over_c(U, round_img(av, 176), 190, 230)
        th = c.frames(9, int(cw), int(chh), key="grid")
        for j, (x0, y0) in enumerate(cells):
            if j == p.cell: continue
            im = th[j].astype(np.float32)
            im = grade(im, sat=0.8 + 0.1 * (j % 3), bright=0.9 + 0.05 * (j % 2))
            Y1 = min(H, int(y0 + chh)); U[int(y0):Y1, int(x0):int(x0) + int(cw)] = im[:Y1 - int(y0), :min(int(cw), W - int(x0))]
        st["ui"] = U
    q = eio(seg(lt, 0.3, 1.0))
    tx, ty = cells[p.cell]
    rx, ry, rw, rh = lerp(0, tx, q), lerp(0, ty, q), lerp(W, cw, q), lerp(H, chh, q)
    F = st["ui"] * eo(q) + O * (1 - eo(q)) * 0.3
    V = crop_cover(O, int(rw), int(rh), c.subj.hx, c.subj.hy + c.subj.r * lerp(0, 1.1, q))
    X0, Y0 = int(rx), int(ry); X1, Y1 = min(W, X0 + V.shape[1]), min(H, Y0 + V.shape[0])
    F[Y0:Y1, X0:X1] = V[:Y1 - Y0, :X1 - X0]
    if q > 0.95:
        k = seg(lt, 1.35, 0.3)
        over_c(F, shape_layer("heart", 150, (255, 255, 255)), tx + cw / 2, ty + chh / 2, op=eo(k) * (1 - seg(lt, 1.9, 0.3)), scale=eback(k))
    return F

# ================================================================== 24. over-the-shoulder ====
@effect("over-the-shoulder", "Over-the-shoulder shot: a phone screen in the foreground plays your post with live likes, with an out-of-focus shoulder.", mask=False,
        defaults=dict(handle="@yourhandle", caption="new video is up!", likes=4820), main="caption")
def fx_ots(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("ots")
    sw, sh = 640, 900
    if "sh" not in st:
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (60, H + 140), (520, 700), -18, 0, 360, 1.0, -1)
        st["sh"] = cv2.GaussianBlur(m, (0, 0), 50)[..., None]
        st["dst"] = np.float32([[300, 900], [1020, 960], [1040, 1960], [190, 1940]])
    a = eo(seg(lt, 0, 0.7))
    L = Layer(sw, sh); L.rrect((0, 0, sw, sh), 40, fill=(10, 10, 12, 255))
    L.text((40, 40), p.handle, "bold", 30, WHITE)
    vid = crop_cover(O, sw - 40, 560, c.subj.hx, c.subj.hy + c.subj.r * 0.8)
    L.paste(vid, 20, 96, mask_r=18)
    likes = int(p.likes * eo(seg(lt, 0.4, 2.5)))
    L.poly([(40 + 18, 712), (40, 690), (40 + 36, 690)], fill=(255, 55, 95, 255)); L.circle(49, 684, 9, fill=(255, 55, 95, 255)); L.circle(67, 684, 9, fill=(255, 55, 95, 255))
    L.text((96, 680), f"{commas(likes)} likes", "bold", 30, WHITE)
    L.text((40, 740), p.handle.lstrip("@") + "  " + p.caption, "regular", 28, wa(220))
    L.rrect((40, 800, sw - 40, 806), 3, fill=wa(60))
    S_ = L.np()
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = st["dst"] + np.float32([[0, 400 * (1 - a)]] * 4)
    M = cv2.getPerspectiveTransform(src, dst)
    P = S_.copy(); P[..., :3] *= P[..., 3:4]
    Wp = cv2.warpPerspective(P, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    Wp = cv2.GaussianBlur(Wp, (0, 0), 1.2)
    al = Wp[..., 3:4]
    bgd = fast_blur(O, 6) * 0.5 + O * 0.5
    F = bgd * (1 - al) + Wp[..., :3] * 255
    F = F * (1 - st["sh"] * 0.92 * a) + np.array([24, 20, 20], np.float32) * st["sh"] * 0.92 * a
    return F

# ================================================================== 25. editorial-notes ====
@effect("editorial-notes", "Magazine editorial look: serif masthead, warm film grade, thin frame and numbered annotation call-outs that draw onto the speaker.", mask=True,
        defaults=dict(masthead="The Edit", issue="NO. 07", notes=["Natural light, no ring light", "Linen shirt, earthy tones", "One take, no script"]),
        main="notes")
def fx_editorial(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("ed")
    if "top" not in st:
        T = Layer(W, 260)
        T.text((60, 70), p.masthead, "serifi", 120, (250, 246, 238, 255))
        T.text((W - 60, 96), p.issue, "semi", 24, (250, 246, 238, 220), anchor="rt", track=4)
        T.text((W - 60, 132), datetime.date.today().strftime("%B %Y").upper(), "regular", 20, (250, 246, 238, 190), anchor="rt", track=3)
        st["top"] = T.np()
        Fr = Layer(W, H); Fr.rect((30, 30, W - 30, 32), fill=(250, 246, 238, 220)); Fr.rect((30, H - 32, W - 30, H - 30), fill=(250, 246, 238, 220))
        Fr.rect((30, 30, 32, H - 30), fill=(250, 246, 238, 220)); Fr.rect((W - 32, 30, W - 30, H - 30), fill=(250, 246, 238, 220))
        st["fr"] = Fr.np()
    F = grade(O, sat=0.85, contrast=1.06, tint=(1.05, 1.0, 0.9), lift=6)
    a = eo(seg(lt, 0, 0.6))
    if "dark" not in st:
        yv = np.arange(H, dtype=np.float32)
        st["dark"] = (1 - 0.5 * np.clip(1 - yv / 460, 0, 1) - 0.35 * np.clip((yv - (H - 360)) / 360, 0, 1))[:, None, None]
    F *= 1 - (1 - st["dark"]) * a
    over(F, st["fr"], 0, 0, a); over(F, st["top"], 0, 40 * (1 - a), a)
    s = c.subj; hx, hy, r = s.hx, s.hy, s.r
    anchors = [((hx - r * 0.25, hy - r * 0.35), (90, hy - r * 1.6), "l"), ((hx + r * 1.35, hy + r * 2.3), (W - 90, hy + r * 1.4), "r"),
               ((hx - r * 0.7, hy + r * 3.6), (90, hy + r * 4.3), "l")]
    for j, note in enumerate(p.notes[:3]):
        (ax, ay), (lx, ly), side = anchors[j]
        ly = float(np.clip(ly, 330, H - 220)); k = seg(lt, 0.5 + 0.5 * j, 0.6)
        if k <= 0: continue
        ry0, ry1 = int(max(0, ly - 30)), int(min(H, ly + 130))
        rx0, rx1 = (int(lx), int(min(W, lx + 380))) if side == "l" else (int(max(0, lx - 380)), int(lx))
        lum = float((F[ry0:ry1, rx0:rx1] @ LUMA).mean()) if ry1 > ry0 and rx1 > rx0 else 100.0
        ink, halo = ((24, 22, 20), (255, 250, 240)) if lum > 135 else ((250, 246, 238), (0, 0, 0))
        L = Layer(W, H)
        L.circle(ax, ay, 7, fill=ink + (255,)); L.circle(ax, ay, 14 * eo(k), outline=ink + (200,), width=2)
        ex = lerp(ax, lx, eo(k)); ey = lerp(ay, ly, eo(k))
        L.line([(ax, ay), (ex, ey)], ink + (235,), 2)
        if k > 0.6:
            tk = seg(k, 0.6, 0.4)
            lines = wrap_text(note, "serifi", 42, 360)
            L.circle(lx + (20 if side == "l" else -20), ly - 46, 20, outline=ink + (255,), width=2)
            L.text((lx + (20 if side == "l" else -20), ly - 46), str(j + 1), "serif", 22, ink + (255,), anchor="mm")
            for i_, ln in enumerate(lines):
                L.text((lx, ly + 6 + i_ * 48), ln, "serifi", 42, ink + (int(255 * tk),), anchor="lt" if side == "l" else "rt",
                       stroke=2, stroke_fill=halo + (int(70 * tk),))
        over(F, L.np(), 0, 0)
    return F

# ================================================================== 26. the-mill ====
@effect("the-mill", "Content mill: video cards ride a conveyor belt under the speaker, get stamped 'POSTED' and a shipped counter ticks up.",
        defaults=dict(stamp="POSTED", counter_label="Reels shipped", start_count=127, accent="#34C759"), main="stamp")
def fx_mill(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("mill"); acc = rgb(p.accent)
    by0, by1 = 1470, 1820; speed, spacing = 260, 300; cw_, ch_ = 200, 260
    if "cards" not in st:
        st["cards"] = [rounded_frame(im.astype(np.float32), 20, 4, (255, 255, 255)) for im in c.frames(6, cw_, ch_, key="mill")]
        S_ = Layer(260, 120); S_.rrect((6, 6, 254, 114), 14, outline=acc + (255,), width=8)
        S_.text((130, 60), p.stamp, "black", 50, acc + (255,), anchor="mm"); st["stamp"] = S_.np()
    a = eo(seg(lt, 0, 0.5))
    belt = np.zeros((by1 - by0, W, 3), np.float32) + 26
    off = int(lt * speed) % 60
    for xs_ in range(-60 + off, W, 60):
        cv2.line(belt, (xs_, 0), (xs_ + 40, by1 - by0), (40, 40, 46), 10)
    belt[:8] = 70; belt[-8:] = 70
    F[by0:by1] = F[by0:by1] * (1 - a) + belt * a
    sx = W / 2
    count = 0
    for k in range(-1, 6):
        x = ((lt * speed) % spacing) + k * spacing - spacing
        idx = int(math.floor(lt * speed / spacing)) - k
        card = st["cards"][idx % len(st["cards"])]
        over(F, card, x - cw_ / 2, by0 - 40 - 20 + 30, a)
        if x > sx + 10:
            over_c(F, st["stamp"], x, by0 + 60, op=a, scale=0.7, angle=-12)
    phase = ((lt * speed) % spacing) / spacing
    press = math.sin(math.pi * clamp01((phase - 0.35) / 0.3)) if 0.35 < phase < 0.65 else 0
    P = Layer(140, 360); P.rect((60, 0, 80, 200), fill=(120, 120, 130, 255)); P.rrect((0, 200, 140, 260), 10, fill=(60, 60, 68, 255))
    over(F, P.np(), sx - 70, by0 - 470 + 230 * press, a)
    shipped = p.start_count + max(0, int((lt * speed - spacing * 0.5) // spacing))
    glass(F, W - 400, 160, 340, 170, r=36, op=a)
    L = Layer(340, 170); L.text((32, 30), p.counter_label, "semi", 26, wa(200)); L.text((30, 66), str(shipped), "black", 72, WHITE)
    over(F, L.np(), W - 400, 160, a)
    return F

# ================================================================== 27. comment-box ====
@effect("comment-box", "Reply-to-comment sticker pops in at the top (avatar, username, comment) with a little heart.",
        defaults=dict(user="creator_fan", comment="How do you edit your reels so fast?"), main="comment")
def fx_comment_box(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("cb")
    if "L" not in st:
        lines = wrap_text(p.comment, "bold", 44, 720); w = max(560, max(text_w(l, "bold", 44) for l in lines) + 90); h = 130 + 56 * len(lines) + 30
        L = Layer(w + 20, h + 30)
        L.rrect((0, 0, w, h), 34, fill=(255, 255, 255, 255)); L.poly([(70, h - 2), (110, h - 2), (66, h + 26)], fill=(255, 255, 255, 255))
        av = c.frames(1, 120, 120, key="cbav")[0]
        A = round_img(av, 64)
        L.text((116, 48), f"Reply to {p.user}'s comment", "medium", 28, (120, 120, 126, 255), anchor="lm")
        for j, ln in enumerate(lines): L.text((40, 108 + j * 56), ln, "bold", 44, (18, 18, 20, 255))
        arr = L.np(); over(arr, A, 36, 16)
        st["L"] = arr
    a = seg(lt, 0.15, 0.5)
    L = st["L"]
    over_c(F, L, W / 2, 200 + L.shape[0] / 2, op=eo(a * 1.5), scale=0.6 + 0.4 * eback(a), angle=-2.5 * eo(a))
    k = seg(lt, 0.9, 0.4)
    if k > 0:
        over_c(F, shape_layer("heart", 90, (255, 55, 95)), W / 2 + L.shape[1] / 2 - 40, 200 + L.shape[0] - 30, op=eo(k), scale=eback(k), angle=12)
    return F


# ================================================================== 28. warm-demo ====
@effect("warm-demo", "Warm film grade with drifting light leaks, a LIVE DEMO badge and feature pills that slide in beside the speaker.",
        defaults=dict(badge="LIVE DEMO", features=["Fast", "Simple", "Free to start"], accent="#FF9F0A"), main="features")
def fx_warm_demo(F, O, c):
    p, lt = c.cfg, c.lt; acc = rgb(p.accent); st = c.st("warm")
    if "yy" not in st:
        st["yy"], st["xx"] = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    g = (O @ LUMA)[..., None] / 255
    X = grade(O, sat=1.12, contrast=1.08) + (g - 0.5) * np.array([46, 10, -40], np.float32)
    yy, xx = st["yy"], st["xx"]; leak = np.zeros_like(yy)
    for cx0, cy0, sp, sz, ph in [(0.05, 0.1, 0.35, 0.35, 0), (0.95, 0.75, 0.28, 0.3, 2.1)]:
        lx = (cx0 + 0.12 * math.sin(lt * sp * 2 + ph)) * W / 4; ly = (cy0 + 0.08 * math.cos(lt * sp * 2 + ph)) * H / 4
        s_ = sz * W / 4; leak += np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * s_ * s_)) * (0.75 + 0.25 * math.sin(lt * 1.7 + ph))
    leak = cv2.resize(leak, (W, H))[..., None]
    F = np.clip(X + leak * np.array([255, 120, 40], np.float32) * 0.45, 0, 255)
    a = eo(seg(lt, 0, 0.5))
    bw = text_w(p.badge, "bold", 28) + 86
    glass(F, 50, 120, bw, 66, r=33, op=a)
    B = Layer(bw, 66); pu = 0.5 + 0.5 * math.sin(lt * 5)
    B.circle(36, 33, 9 + 4 * pu, fill=(255, 59, 48, int(120 * (1 - pu)))); B.circle(36, 33, 9, fill=(255, 59, 48, 255))
    B.text((58, 33), p.badge, "bold", 28, WHITE, anchor="lm", track=2); over(F, B.np(), 50, 120, a)
    for j, ft in enumerate(p.features[:4]):
        k = seg(lt, 0.6 + 0.35 * j, 0.5)
        if k <= 0: continue
        pw = text_w(ft, "semi", 34) + 120; ph = 92
        x = W - 50 - pw + (pw + 80) * (1 - eo(k)); y = 1180 + j * 124
        glass(F, x, y, pw, ph, r=46, op=eo(k), tone="light")
        L = Layer(pw, ph); L.circle(48, 46, 22, fill=acc + (255,)); check_path(L, 48, 46, 26, seg(lt, 0.8 + 0.35 * j, 0.25), WHITE, 4)
        L.text((86, 46), ft, "semi", 34, DARK, anchor="lm"); over(F, L.np(), x, y, eo(k))
    return F

# ================================================================== 29. caption-cutouts ====
CUT_STYLES = [("#FFFFFF", "#111111", "black"), ("#FFD60A", "#111111", "serifb"), ("#111111", "#FFFFFF", "mono"),
              ("#FF4D8D", "#FFFFFF", "round"), ("#7CF5C4", "#111111", "black"), ("#F2E8D5", "#C0392B", "serifb")]

def cutout_word(word, j):
    bg, fg, kind = CUT_STYLES[j % len(CUT_STYLES)]
    size = 78 if kind != "mono" else 70
    tw_ = text_w(word, kind, size); w, h = tw_ + 44, size * 1.35; pad = 14
    rng = np.random.default_rng(j * 7 + len(word))
    jit = lambda: float(rng.uniform(-6, 6))
    L = Layer(w + 2 * pad + 8, h + 2 * pad + 8)
    pts = [(pad + jit(), pad + jit()), (pad + w / 2, pad + jit() * 0.5), (pad + w + jit(), pad + jit()), (pad + w + jit(), pad + h + jit()),
           (pad + w / 2, pad + h + jit() * 0.5), (pad + jit(), pad + h + jit())]
    L.poly([(x + 7, y + 8) for x, y in pts], fill=(0, 0, 0, 110))
    L.poly(pts, fill=rgba(bg))
    L.text((pad + w / 2, pad + h / 2), word, kind, size, rgba(fg), anchor="mm")
    return L.np(), float(rng.uniform(-7, 7))

@effect("caption-cutouts", "Word-by-word captions made of ransom-note paper cut-outs (mixed colours, fonts and angles) that pop in as you speak.",
        defaults=dict(text="this is how I edit every reel in five minutes", words_per_line=3, y=1380, timings=[]), main="text")
def fx_caption_cutouts(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("cut")
    words = p.text.split(); n = len(words)
    if "w" not in st: st["w"] = [cutout_word(wd.upper() if j % 3 == 0 else wd, j) for j, wd in enumerate(words)]
    times = [float(t) for t in p.timings][:n] if p.timings else []
    if len(times) < n:
        per = (c.D - 0.5) / max(1, n); times = [0.2 + j * per for j in range(n)]
    cur = max([j for j in range(n) if lt >= times[j]], default=-1)
    if cur < 0: return F
    wpl = max(1, int(p.words_per_line)); c0 = (cur // wpl) * wpl
    chunk = list(range(c0, min(n, c0 + wpl)))
    ws = [st["w"][j][0].shape[1] * 0.92 for j in chunk]; total = sum(ws)
    sc = min(1.0, 1000 / total); x = W / 2 - total * sc / 2
    for j, wd in zip(chunk, ws):
        L, ang = st["w"][j]
        if lt >= times[j]:
            k = seg(lt, times[j], 0.28)
            over_c(F, L, x + wd * sc / 2, p.y + 6 * math.sin(lt * 3 + j), op=eo(k * 2), scale=sc * (0.4 + 0.6 * eback(k, 2.4)), angle=ang * (1 + (1 - eo(k)) * 2))
        x += wd * sc
    return F

# ================================================================== 30. split-browser ====
@effect("split-browser", "Split screen: a browser window on top scrolls your website while a cursor clicks; the speaker is below.", mask=True,
        defaults=dict(url="yourwebsite.com", page_title="Build in public", page_sub="Weekly notes on making things online.",
                      button="Subscribe", accent="#5B5BF6"), main="page_title")
def fx_split_browser(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("br"); TH = 900; acc = rgb(p.accent)
    th = int(TH * eo(seg(lt, 0, 0.55))); vh = H - th
    y0 = int(np.clip(c.subj.hy - 0.30 * vh, 0, H - vh)); F[th:] = O[y0:y0 + vh]
    if th <= 0: return F
    bx, by, bw, bh = 30, 40, 1020, 820
    if "page" not in st:
        PG = Layer(bw - 2, 1500)
        PG.rect((0, 0, bw, 1500), fill=(252, 252, 253, 255))
        PG.text((60, 70), "◆ " + p.url.split(".")[0].capitalize(), "bold", 28, (20, 20, 24, 255))
        for k_, t_ in enumerate(["Blog", "About", "Shop"]): PG.text((bw - 330 + k_ * 100, 74), t_, "medium", 24, (90, 90, 96, 255))
        PG.text((60, 190), p.page_title, "black", 76, (20, 20, 24, 255)); PG.text((60, 300), p.page_sub, "regular", 32, (110, 110, 118, 255))
        PG.rrect((60, 370, 60 + text_w(p.button, "bold", 28) + 80, 446), 38, fill=acc + (255,))
        PG.text((100, 408), p.button, "bold", 28, WHITE, anchor="lm")
        ims = c.frames(2, 440, 300, key="br")
        PG.paste(ims[0], 60, 510, mask_r=20); PG.paste(ims[1], 520, 510, mask_r=20)
        for j in range(9):
            yy_ = 860 + j * 48 + (60 if j >= 4 else 0)
            if j == 4: PG.text((60, 1060), "Latest posts", "bold", 40, (20, 20, 24, 255)); continue
            PG.rrect((60, yy_, 60 + [880, 820, 860, 540, 0, 880, 760, 840, 500][j], yy_ + 18), 9, fill=(225, 225, 232, 255))
        st["page"] = PG.np()
        Br = Layer(bw, 80)
        Br.rrect((0, 0, bw, 80), 0, fill=(236, 236, 241, 255))
        for j in range(3): Br.circle(36 + j * 28, 40, 9, fill=(190, 190, 198, 255))
        Br.rrect((150, 16, bw - 40, 64), 24, fill=(255, 255, 255, 255))
        Br.rrect((176, 32, 190, 48), 3, fill=(120, 120, 128, 255)); Br.arc((174, 22, 192, 40), 180, 360, (120, 120, 128, 255), 3)
        Br.text((206, 40), p.url, "medium", 26, (60, 60, 66, 255), anchor="lm"); st["bar"] = Br.np()
        st["bg"] = grad(W, TH, (30, 32, 44), (14, 15, 20))
    P = st["bg"].copy()
    shadow(P, bx, by, bw, bh, 26, 1, 0.5, 18)
    scroll = eio(seg(lt, 1.5, 2.0)) * 520
    page = st["page"][int(scroll):int(scroll) + bh - 80]
    win = np.zeros((bh, bw, 4), np.float32); win[80:80 + page.shape[0], 1:1 + page.shape[1]] = page
    win[:80] = st["bar"]; win[..., 3] *= rmask(bw, bh, 26)
    over(P, win, bx, by)
    k = eo(seg(lt, 0.3, 0.6)); clk = 0.95
    tx, ty = bx + 150, by + 80 + 408 - scroll
    cx_, cy_ = lerp(bx + 860, tx, k), lerp(by + 700, ty, k)
    if lt > clk and lt < clk + 0.4:
        R_ = Layer(160, 160); q = seg(lt, clk, 0.4); R_.circle(80, 80, 20 + 50 * q, outline=acc + (int(255 * (1 - q)),), width=5); over_c(P, R_.np(), tx, ty)
    over_c(P, shape_layer("cursor", 70, (0, 0, 0)), cx_ + 22, cy_ + 26, scale=1 - 0.15 * math.sin(math.pi * seg(lt, clk, 0.18)))
    F[:th] = P[TH - th:]
    F[th:th + 4] = F[th:th + 4] * 0.2 + np.array(acc, np.float32) * 0.8
    return F

# ================================================================== 31. led-sign ====
@effect("led-sign", "Glowing LED dot-matrix sign scrolls your message across the top of the frame.",
        defaults=dict(text="NEW VIDEO EVERY DAY", color="#FFB020", speed=32), main="text")
def fx_led_sign(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("led"); rows, cols, cell = 16, 96, 10
    col = np.array(rgb(p.color), np.float32)
    if "bm" not in st:
        fnt = font("mono", 15); txt = "   " + p.text.upper() + "   "
        tw_ = int(fnt.getlength(txt)) + 4
        im = Image.new("L", (tw_, rows), 0); ImageDraw.Draw(im).text((0, rows // 2), txt, font=fnt, fill=255, anchor="lm")
        bm = (np.asarray(im) > 110).astype(np.float32)
        st["bm"] = np.concatenate([np.zeros((rows, cols), np.float32), bm], 1)
        yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32)
        st["dot"] = np.clip(3.9 - np.sqrt((xx - 4.5) ** 2 + (yy - 4.5) ** 2), 0, 1)
        Pn = Layer(cols * cell + 60, rows * cell + 60)
        Pn.rrect((0, 0, cols * cell + 60, rows * cell + 60), 22, fill=(16, 13, 10, 245), outline=(80, 70, 60, 255), width=4)
        st["panel"] = Pn.np()
    bm = st["bm"]; Lb = bm.shape[1]; off = int(cols * 0.6 + lt * p.speed) % Lb
    idx = (np.arange(cols) + off) % Lb; B = bm[:, idx]
    big = np.kron(B, np.ones((cell, cell), np.float32)) * np.tile(st["dot"], (rows, cols))
    off_dots = np.tile(st["dot"], (rows, cols)) * 0.12
    a = eo(seg(lt, 0, 0.5)); x0, y0 = 60, 150
    over(F, st["panel"], x0 - 30, y0 - 30, a)
    reg = F[y0:y0 + rows * cell, x0:x0 + cols * cell]
    lit = (big[..., None] * col + off_dots[..., None] * col * 0.6) * a
    glow = cv2.GaussianBlur(big, (0, 0), 5)[..., None] * col * 0.7 * a
    reg[:] = np.clip(reg + glow + lit, 0, 255)
    return F

# ================================================================== 32. rabbit-hole ====
@effect("rabbit-hole", "Falling down the rabbit hole: the frame repeats inside itself, twisting and zooming endlessly toward your face.", mask=True,
        defaults=dict(levels=6, ratio=0.62, twist=14, speed=0.55))
def fx_rabbit_hole(F, O, c):
    p, lt = c.cfg, c.lt
    ph = (lt * p.speed) % 1; Z = p.ratio ** (-ph)
    cx, cy = c.subj.hx, c.subj.hy
    Ob = O.copy(); bd = 16
    Ob[:bd] = 255; Ob[-bd:] = 255; Ob[:, :bd] = 255; Ob[:, -bd:] = 255
    out = np.zeros_like(O)
    one = np.ones((H, W), np.float32)
    for k in range(p.levels):
        s = Z * p.ratio ** k
        if s < 0.04: break
        ang = (k - ph) * p.twist
        M = cv2.getRotationMatrix2D((cx, cy), ang, s)
        if k == 0 and s >= 1.0:
            out = cv2.warpAffine(O, M, (W, H), borderMode=cv2.BORDER_REFLECT) * (1 - 0.07 * k)
            continue
        img = cv2.warpAffine(Ob, M, (W, H)); m = cv2.warpAffine(one, M, (W, H))[..., None]
        sh = fast_blur(m[..., 0], 18)[..., None]
        out = out * (1 - sh * 0.45) * (1 - m) + img * (1 - 0.07 * k) * m
    return out

# ================================================================== 33. profile-wall ====
@effect("profile-wall", "Background becomes a scrolling wall of profile cards (avatars, names, follow buttons) behind the speaker.", mask=True,
        defaults=dict(names=["Aarav", "Riya", "Sam", "Kiran", "Maya", "Leo", "Zara", "Dev", "Nia", "Omar", "Ira", "Kabir"], speed=70,
                      accent="#0A84FF"))
def fx_profile_wall(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("wall"); acc = rgb(p.accent)
    cw_, chh, gx, gy = 300, 380, 45, 40; TH = 4 * (chh + gy)
    if "tile" not in st:
        T = np.zeros((TH, W, 3), np.float32) + 10
        k = 0
        for col in range(3):
            for row in range(4):
                nm = p.names[k % len(p.names)]; k += 1
                L = Layer(cw_, chh)
                L.rrect((0, 0, cw_, chh), 30, fill=(30, 30, 36, 255), outline=(255, 255, 255, 30), width=2)
                c1, c2 = PALETTE[k % len(PALETTE)], PALETTE[(k * 2 + 3) % len(PALETTE)]
                L.paste(grad(140, 140, c1, c2, angle=45), 80, 36, mask_r=70)
                L.text((150, 106), nm[0], "black", 64, WHITE, anchor="mm")
                L.text((150, 214), nm, "bold", 32, WHITE, anchor="mm")
                L.text((150, 254), "@" + nm.lower() + "_creates", "regular", 21, wa(150), anchor="mm")
                L.rrect((60, 296, 240, 348), 26, fill=acc + (255,)); L.text((150, 322), "Follow", "semi", 24, WHITE, anchor="mm")
                over(T, L.np(), gx + col * (cw_ + gx), gy / 2 + row * (chh + gy))
        st["tile"] = cv2.GaussianBlur(T, (0, 0), 1.0)
    a = eo(seg(lt, 0, 0.6))
    bg = np.empty_like(O)
    for col in range(3):
        x0 = int(gx / 2 + col * (cw_ + gx)); x1 = W if col == 2 else int(gx / 2 + (col + 1) * (cw_ + gx))
        if col == 0: x0 = 0
        off = int(lt * p.speed * (1 + 0.3 * col) + col * 230) % TH
        rows = (np.arange(H) + off) % TH
        bg[:, x0:x1] = st["tile"][rows][:, x0:x1]
    bg *= 0.8
    m = c.mask[..., None]
    out = bg * (1 - m) + O * m
    if c.ai:
        ms = cv2.resize(c.mask, (W // 4, H // 4)); ring = np.clip(cv2.GaussianBlur(ms, (0, 0), 6) - ms, 0, 1) * 2
        out += cv2.resize(ring, (W, H))[..., None] * np.array(acc, np.float32) * 0.6
    return O * (1 - a) + out * a

# ================================================================== 34. lineup ====
@effect("lineup", "Three of you in a lineup against a height-chart wall: delayed clones either side, each with a name plate.", mask=True,
        defaults=dict(labels=["Me at 9 AM", "Me at 2 PM", "Me at 11 PM"], delay=0.5), main="labels")
def fx_lineup(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("lineup")
    if "wall" not in st:
        Wl = grad(W, H, (88, 96, 112), (58, 64, 78)); L = Layer(W, H)
        for j, yy in enumerate(range(260, H, 110)):
            L.rect((0, yy, W, yy + 3), fill=wa(90 if j % 2 == 0 else 45))
            if j % 2 == 0:
                lab = f"{200 - j * 10}"
                L.text((26, yy - 34), lab, "bold", 26, wa(150)); L.text((W - 26, yy - 34), lab, "bold", 26, wa(150), anchor="rt")
        over(Wl, L.np(), 0, 0); st["wall"] = Wl
        st["plates"] = []
        for lab in p.labels[:3]:
            Pl = Layer(300, 74); Pl.rrect((0, 0, 300, 74), 10, fill=(14, 14, 16, 235)); Pl.text((150, 37), lab, "mono", 26, WHITE, anchor="mm")
            st["plates"].append(Pl.np())
    a = eo(seg(lt, 0, 0.6))
    m = c.mask[..., None]
    F = st["wall"] * a + O * (1 - a)
    sc = 1.36; cw_, ch_ = int(W / 2 * sc), int(H / 2 * sc)
    for k, xc in [(2, 225), (1, 855)]:
        idx = max(0, len(c.hist) - 1 - int(p.delay * c.fps * k))
        sm, smk = c.hist[idx]
        if smk is None: continue
        im = cv2.resize(sm.astype(np.float32), (cw_, ch_)); mk = cv2.resize(smk, (cw_, ch_))
        Lc = np.dstack([im / 255 * 0.82, mk * a])
        over(F, Lc, xc - c.subj.hx / 2 * sc, H - ch_)
    F = F * (1 - m) + O * m
    for j, xc in enumerate([225, 540, 855]):
        if j < len(st["plates"]):
            k_ = seg(lt, 0.5 + 0.2 * j, 0.4)
            over_c(F, st["plates"][j], xc, 1760 + 40 * (1 - eo(k_)), op=eo(k_))
    return F


def clean_plate(O, m):
    """Approximate background with the speaker removed (normalised blur fill)."""
    k = 4; sw, sh = W // k, H // k
    small = cv2.resize(O, (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.dilate(cv2.resize(m, (sw, sh)), np.ones((7, 7), np.uint8))
    wgt = np.clip(1 - ms * 1.2, 0, 1)
    fill = None; cov = np.zeros((sh, sw, 1), np.float32)
    for sig in (16, 48, 140):
        num = cv2.GaussianBlur(small * wgt[..., None], (0, 0), sig); den = cv2.GaussianBlur(wgt, (0, 0), sig)[..., None]
        f_ = num / np.maximum(den, 1e-4); w_ = np.clip(den * 8, 0, 1) * (1 - cov)
        fill = f_ * w_ if fill is None else fill + f_ * w_
        cov = cov + w_
    if wgt.sum() > 10:
        meanc = (small * wgt[..., None]).sum((0, 1)) / wgt.sum()
        fill = fill + meanc * np.clip(1 - cov, 0, 1)
    fill = cv2.resize(fill, (W, H))
    md = cv2.GaussianBlur(cv2.resize(ms, (W, H)), (0, 0), 4)[..., None]
    return O * (1 - md) + fill * md

def delayed(c, sec):
    idx = max(0, len(c.hist) - 1 - int(sec * c.fps))
    return c.hist[idx]

def cursor_layer(col, size=64):
    s = size; pts = [(s * .1, s * .05), (s * .1, s * .82), (s * .3, s * .64), (s * .45, s * .95), (s * .58, s * .89), (s * .44, s * .58), (s * .72, s * .58)]
    L = Layer(s, s); L.poly(pts, fill=WHITE)
    L.poly([(x + (s * .38 - x) * 0.2, y + (s * .5 - y) * 0.16) for x, y in pts], fill=rgba(col))
    return L.np()

def name_tag(text, col):
    w = text_w(text, "semi", 24) + 32; L = Layer(w, 44)
    L.rrect((0, 0, w, 44), 14, fill=rgba(col)); L.text((16, 22), text, "semi", 24, WHITE, anchor="lm"); return L.np()

# ================================================================== 35. live-cursor ====
@effect("live-cursor", "Multiplayer design-tool cursors with name tags glide around; one drags a selection box around your head.", mask=True,
        defaults=dict(names=["You", "Editor", "Client"], colors=["#FF4D8D", "#0A84FF", "#34C759"], label="speaker.mov"), main="names")
def fx_live_cursor(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("cur"); s = c.subj
    if "cur" not in st:
        st["cur"] = [cursor_layer(col) for col in p.colors]; st["tag"] = [name_tag(n, col) for n, col in zip(p.names, p.colors)]
    x0, y0, x1, y1 = s.hx - s.r * 1.35, s.hy - s.r * 1.45, s.hx + s.r * 1.35, s.hy + s.r * 1.55
    k = eio(seg(lt, 0.5, 0.9))
    sel_col = rgb(p.colors[1 % len(p.colors)])
    if k > 0:
        bx1, by1 = lerp(x0, x1, k), lerp(y0, y1, k)
        L = Layer(W, H)
        L.rect((x0, y0, bx1, by1), fill=sel_col + (28,))
        for (ax, ay, bx, by) in [(x0, y0, bx1, y0), (bx1, y0, bx1, by1), (x0, by1, bx1, by1), (x0, y0, x0, by1)]:
            L.line([(ax, ay), (bx, by)], sel_col + (255,), 3)
        if k >= 1:
            for hx_, hy_ in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]: L.rect((hx_ - 9, hy_ - 9, hx_ + 9, hy_ + 9), fill=WHITE); L.rect((hx_ - 7, hy_ - 7, hx_ + 7, hy_ + 7), fill=sel_col + (255,))
            tw_ = text_w(p.label, "semi", 22) + 24
            L.rrect((x0, y0 - 46, x0 + tw_, y0 - 10), 8, fill=sel_col + (255,)); L.text((x0 + 12, y0 - 28), p.label, "semi", 22, WHITE, anchor="lm")
        over(F, L.np(), 0, 0)
    paths = []
    for j in range(len(p.names)):
        if j == 1:
            if lt < 0.5: px, py = lerp(W + 60, x0, eo(seg(lt, 0, 0.5))), lerp(H * 0.7, y0, eo(seg(lt, 0, 0.5)))
            elif lt < 1.4: px, py = lerp(x0, x1, k), lerp(y0, y1, k)
            else: px, py = x1 + 40 * math.sin((lt - 1.4) * 1.3), y1 + 30 * math.sin((lt - 1.4) * 1.7)
        else:
            ph = j * 2.1
            px = W * (0.25 + 0.5 * (j % 2)) + 260 * math.sin(lt * 0.9 + ph); py = H * (0.55 + 0.18 * j) % (H * 0.9) + 220 * math.sin(lt * 0.7 + ph * 1.3)
        paths.append((px, py))
    for j, (px, py) in enumerate(paths):
        a = eo(seg(lt, 0.15 * j, 0.4))
        over(F, st["cur"][j], px - 6, py - 3, a); over(F, st["tag"][j], px + 34, py + 44, a)
    return F

# ================================================================== 36. clone-row ====
@effect("clone-row", "A row of time-delayed clones of you appears across the frame, each popping in one after another.", mask=True,
        defaults=dict(clones=4, delay=0.22, scale=0.55))
def fx_clone_row(F, O, c):
    p, lt = c.cfg, c.lt
    m = c.mask[..., None]; n = p.clones
    offs = [-400, 400, -215, 215, -560, 560][:n]
    xs = sorted([c.subj.hx + d for d in offs], key=lambda x: -abs(x - c.subj.hx))
    sc = p.scale * 2; cw_, ch_ = int(W / 2 * sc), int(H / 2 * sc)
    for j, xc in enumerate(xs):
        a = seg(lt, 0.3 + 0.18 * j, 0.35)
        if a <= 0: continue
        sm, smk = delayed(c, p.delay * (j + 1))
        if smk is None: continue
        im = cv2.resize(sm.astype(np.float32), (cw_, ch_)); mk = cv2.resize(smk, (cw_, ch_))
        Lc = np.dstack([im / 255 * (0.9 - 0.04 * j), mk])
        over_c(F, Lc, xc + (W / 4 - c.subj.hx / 2) * sc, H - ch_ / 2 + 30 * (1 - eback(a)), op=eo(a), scale=0.85 + 0.15 * eback(a))
    return F * (1 - m) + O * m

# ================================================================== 37. day-dial ====
@effect("day-dial", "A giant engraved dial with 'day' markings rotates behind you and clicks onto today's day number.", mask=True,
        defaults=dict(day=1, word="dial", label="day", ink="#1E1C1A", paper="#EFEAE0"), main="word")
def fx_day_dial(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("dial"); ink = rgb(p.ink)
    R = 540; D_ = 2 * R + 40
    if "dial" not in st:
        L = Layer(D_, D_); cx = cy = D_ / 2
        L.circle(cx, cy, R, fill=rgba(p.paper, 150)); L.circle(cx, cy, R, outline=ink + (255,), width=3)
        L.circle(cx, cy, R - 150, outline=ink + (200,), width=2); L.circle(cx, cy, R - 175, outline=ink + (120,), width=1)
        for t in range(120):
            a = math.radians(t * 3 - 90); ln = 34 if t % 10 == 0 else 18 if t % 5 == 0 else 10
            L.line([(cx + (R - 4) * math.cos(a), cy + (R - 4) * math.sin(a)), (cx + (R - 4 - ln) * math.cos(a), cy + (R - 4 - ln) * math.sin(a))], ink + (230,), 2)
        D = np.asarray(L.im).copy(); base = Image.fromarray(D)
        for k in range(12):
            n = (k % 12) + 1; ang = k * 30
            for txt, kind, size, rr, off in [(str(n), "serif", 84, R - 86, 0), (p.label, "serifi", 42, R - 86, 15)]:
                T = Image.new("RGBA", (300 * S, 160 * S), (0, 0, 0, 0)); ImageDraw.Draw(T).text((150 * S, 80 * S), txt, font=font(kind, size * S), fill=ink + (255,), anchor="mm")
                T = T.rotate(-(ang + off), resample=Image.BICUBIC, expand=True)
                a = math.radians(ang + off - 90)
                px, py = (cx + rr * math.cos(a)) * S, (cy + rr * math.sin(a)) * S
                base.alpha_composite(T, (int(px - T.size[0] / 2), int(py - T.size[1] / 2)))
        L.im = base; st["dial"] = L.np()
        V = Layer(200, 500); Tm = Image.new("RGBA", (500 * S, 200 * S), (0, 0, 0, 0))
        ImageDraw.Draw(Tm).text((250 * S, 100 * S), p.word, font=font("serifi", 150 * S), fill=ink + (255,), anchor="mm")
        V.im = Tm.rotate(90, expand=True); st["word"] = V.np()
    target = ((int(p.day) - 1) % 12) * 30
    q = seg(lt, 0.2, 1.6); rot = lerp(target + 160, target, eback(q, 1.3))
    tick = (math.floor(rot / 6) * 6) if q < 1 else rot
    cx, cy = c.subj.hx, c.subj.hy + c.subj.r * 0.3
    a = eo(seg(lt, 0, 0.6))
    over_c(F, st["dial"], cx, cy, op=a, scale=0.92 + 0.08 * a, angle=tick)
    Lp = Layer(30, max(10, cy - R)); Lp.rect((14, 0, 16, max(10, cy - R)), fill=ink + (255,)); over(F, Lp.np(), cx - 15, 0, a)
    over(F, st["word"], 24, cy - 250, a)
    m = c.mask[..., None]
    return F * (1 - m) + O * m

# ================================================================== 38. predict-keys ====
@effect("predict-keys", "Phone keyboard types your caption with key pops, while predictive-text suggestions complete each word.",
        defaults=dict(text="this tool edits my reels for me", decoys=["the", "and", "my", "is", "so", "it"]), main="text")
def fx_predict_keys(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("kb")
    words = p.text.split(); n = len(words)
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    kw, kh, gap, ky0 = 96, 120, 10, 1430
    def key_box(ch):
        for r_, row in enumerate(rows):
            if ch in row:
                i = row.index(ch); off = (W - len(row) * (kw + gap) + gap) / 2
                return off + i * (kw + gap), ky0 + r_ * (kh + 18)
        return None
    if "kb" not in st:
        L = Layer(W, H - 1240)
        L.rrect((0, 0, W, H - 1240), 0, fill=(209, 212, 219, 250))
        for r_, row in enumerate(rows):
            for ch in row:
                x, y = key_box(ch); y -= 1240
                L.rrect((x, y, x + kw, y + kh), 14, fill=(255, 255, 255, 255)); L.text((x + kw / 2, y + kh / 2), ch, "regular", 46, DARK, anchor="mm")
        L.rrect((240, ky0 + 3 * (kh + 18) - 1240, W - 240, ky0 + 3 * (kh + 18) + kh - 1240), 14, fill=(255, 255, 255, 255))
        L.text((W / 2, ky0 + 3 * (kh + 18) + kh / 2 - 1240), "space", "regular", 36, da(160), anchor="mm")
        st["kb"] = L.np()
    per = (c.D - 0.8) / max(1, n); typed = ""; active = None; sugg = None; tap = None
    for j, wd in enumerate(words):
        t0 = 0.5 + j * per; lt_ = lt - t0
        if lt_ < 0: break
        nl = min(len(wd), 2 if len(wd) > 3 else len(wd)); tk = per * 0.55 / max(1, nl)
        if lt_ < nl * tk:
            ci = int(lt_ / tk); typed_word = wd[:ci + 1]; active = wd[ci].lower(); sugg = (wd, j)
            typed += typed_word; break
        elif len(wd) > 3 and lt_ < nl * tk + 0.25:
            typed += wd[:nl]; sugg = (wd, j); tap = 1; break
        else:
            typed += wd + " "
    a = eo(seg(lt, 0, 0.5)); yo = 300 * (1 - a)
    over(F, st["kb"], 0, 1240 + yo)
    S_ = Layer(W, 110)
    if sugg:
        wd, j = sugg; d1, d2 = p.decoys[j % len(p.decoys)], p.decoys[(j + 3) % len(p.decoys)]
        for i_, s_ in enumerate([d1, wd, d2]):
            cx_ = W / 6 + i_ * W / 3
            if i_ == 1 and tap: S_.rrect((cx_ - 170, 12, cx_ + 170, 98), 18, fill=(255, 255, 255, 255))
            S_.text((cx_, 55), s_, "semi" if i_ == 1 else "regular", 38, DARK, anchor="mm")
            if i_ < 2: S_.rect((W / 3 * (i_ + 1) - 1, 30, W / 3 * (i_ + 1) + 1, 80), fill=da(60))
    over(F, S_.np(), 0, 1250 + yo)
    T = Layer(W, 150); T.rrect((40, 20, W - 40, 130), 55, fill=(255, 255, 255, 250))
    shown = typed[-34:]
    T.text((86, 75), shown, "regular", 40, DARK, anchor="lm")
    if int(lt * 2.2) % 2 == 0: cxp = 86 + text_w(shown, "regular", 40) + 4; T.rect((cxp, 50, cxp + 3, 100), fill=(10, 132, 255, 255))
    over(F, T.np(), 0, 1080 + yo, a)
    if active and key_box(active):
        x, y = key_box(active); y += yo
        Pv = Layer(kw + 40, kh + 110); Pv.rrect((0, 0, kw + 40, kh + 60), 22, fill=(255, 255, 255, 255)); Pv.rrect((20, kh, kw + 20, kh + 110), 12, fill=(255, 255, 255, 255))
        Pv.text(((kw + 40) / 2, (kh + 60) / 2), active, "regular", 64, DARK, anchor="mm")
        shadow(F, x - 20, y - 110, kw + 40, kh + 110, 22, 1, 0.25, 6); over(F, Pv.np(), x - 20, y - 110)
    return F

# ================================================================== 39. orbit-ring ====
@effect("orbit-ring", "A ring of glowing text orbits around your head in 3D, passing behind and in front of you.", mask=True,
        defaults=dict(text="NEW VIDEO • SUBSCRIBE • ", color="#7CF5C4", speed=0.9), main="text")
def fx_orbit_ring(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("ring"); col = rgb(p.color)
    txt = (p.text * 3)[:42] if len(p.text) < 42 else p.text[:60]
    if "g" not in st:
        st["g"] = {}
        for ch in set(txt):
            L = Layer(80, 90); L.text((40, 45), ch, "black", 60, WHITE, anchor="mm", stroke=2, stroke_fill=col + (255,)); st["g"][ch] = L.np()
    s = c.subj; cx, cy = s.hx, s.hy - s.r * 0.15; rx, ry = s.r * 2.3, s.r * 0.5
    a = eo(seg(lt, 0, 0.7)); n = len(txt); items = []
    for i, ch in enumerate(txt):
        th = -2 * math.pi * i / n + lt * p.speed; z = math.sin(th)
        items.append((z, cx + rx * math.cos(th) * (0.6 + 0.4 * a), cy + ry * z, ch, th))
    m = c.mask[..., None]
    for z, x, y, ch, th in sorted(items):
        if z >= 0 or ch == " ": continue
        over_c(F, st["g"][ch], x, y, op=a * (0.55 + 0.25 * (z + 1)), scale=0.75 + 0.2 * (z + 1), sx=0.35 + 0.65 * abs(math.sin(th)))
    F = F * (1 - m) + O * m
    for z, x, y, ch, th in sorted(items):
        if z < 0 or ch == " ": continue
        over_c(F, st["g"][ch], x, y, op=a, scale=0.95 + 0.25 * z, sx=0.35 + 0.65 * abs(math.sin(th)))
    return F

# ================================================================== 40. code-body ====
@effect("code-body", "Your body turns into scrolling code characters (matrix style) while the room goes dark.", mask=True,
        defaults=dict(color="#39FF88", chars="{}()<>=;:+-*/[]01abcdefxyz_#.", keep=0.22))
def fx_code_body(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("code"); cw_, ch_ = 16, 26
    cols, rows = W // cw_ + 1, H // ch_ + 1
    if "atlas" not in st:
        at = []
        for chh in p.chars:
            im = Image.new("L", (cw_ * S, ch_ * S), 0); ImageDraw.Draw(im).text((cw_ * S / 2, ch_ * S / 2), chh, font=font("mono", 22 * S), fill=255, anchor="mm")
            at.append(np.asarray(im.resize((cw_, ch_), Image.BOX)).astype(np.float32) / 255)
        st["atlas"] = np.stack(at); st["seed"] = np.random.default_rng(4).integers(0, len(p.chars), (rows * 3, cols))
    m = c.mask
    lum = cv2.resize((O @ LUMA) * m, (cols, rows), interpolation=cv2.INTER_AREA) / 255
    mk = cv2.resize(m, (cols, rows), interpolation=cv2.INTER_AREA)
    off = int(lt * 14)
    idx = st["seed"][(np.arange(rows)[:, None] - off) % (rows * 3), np.arange(cols)[None, :]]
    G = st["atlas"][idx]
    inten = np.clip(0.25 + lum * 1.3, 0, 1) * (mk > 0.35)
    img = (G * inten[:, :, None, None]).transpose(0, 2, 1, 3).reshape(rows * ch_, cols * cw_)[:H, :W]
    a = eo(seg(lt, 0, 0.8)); scan = H * eo(seg(lt, 0.1, 1.0))
    reveal = (np.arange(H, dtype=np.float32) < scan)[:, None, None].astype(np.float32)
    col = np.array(rgb(p.color), np.float32)
    code = img[..., None] * col * 1.25 + cv2.GaussianBlur(img, (0, 0), 3)[..., None] * col * 0.6
    mm = m[..., None]
    sub = O * p.keep + code
    out = O * (1 - mm) * (1 - 0.78 * a) + (sub * reveal + O * (1 - reveal)) * mm
    if 0 < scan < H:
        y = int(scan); out[max(0, y - 3):y + 3] = out[max(0, y - 3):y + 3] * 0.3 + col * 0.9
    return out

# ================================================================== 41. post-wall ====
@effect("post-wall", "A tilted 3D wall of social posts (thumbnails, likes) scrolls behind you.", mask=True,
        defaults=dict(speed=90))
def fx_post_wall(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("pw"); TW_, TH_ = 1500, 1840
    if "tile" not in st:
        T = np.zeros((TH_, TW_, 3), np.float32) + 14
        ims = c.frames(8, 330, 330, key="pw"); k = 0
        for col in range(4):
            for row in range(4):
                x = 30 + col * 370; y = 20 + row * 460 + (col % 2) * 120
                L = Layer(340, 430); L.rrect((0, 0, 340, 430), 22, fill=(250, 250, 250, 255))
                if k % 3 == 2: L.paste(grad(330, 330, PALETTE[k % 7], PALETTE[(k + 3) % 7], angle=40), 5, 5, mask_r=18)
                else: L.paste(ims[k % len(ims)], 5, 5, mask_r=18)
                L.circle(34, 376, 14, fill=(255, 55, 95, 255)); L.text((58, 376), fmt_num(1200 + (k * 7919) % 90000), "bold", 26, DARK, anchor="lm")
                L.rrect((200, 366, 320, 386), 10, fill=da(40)); k += 1
                over(T, L.np(), x, y % TH_)
                if y + 430 > TH_: over(T, L.np(), x, y - TH_)
        st["tile"] = T
        st["M"] = cv2.getPerspectiveTransform(np.float32([[0, 0], [TW_, 0], [TW_, TH_], [0, TH_]]),
                                              np.float32([[150, -60], [930, -60], [1420, H + 80], [-340, H + 80]]))
        yv = np.arange(H, dtype=np.float32); st["fog"] = (0.35 + 0.65 * np.clip(yv / (H * 0.7), 0, 1))[:, None, None]
    off = int(lt * p.speed) % TH_
    win = np.roll(st["tile"], -off, axis=0)
    bg = cv2.warpPerspective(win, st["M"], (W, H), borderValue=(8, 8, 10)) * st["fog"]
    a = eo(seg(lt, 0, 0.6)); m = c.mask[..., None]
    out = bg * (1 - m) + O * m
    return O * (1 - a) + out * a

# ================================================================== 42. letter-dive ====
@effect("letter-dive", "The frame becomes a giant word with you visible through the letters, then the camera dives into a letter.",
        defaults=dict(word="DIVE", bg="#F4F1EA", letter=-1), main="word")
def fx_letter_dive(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("dive")
    if "M" not in st:
        wd = p.word.upper(); size = fit_size([wd], "black", 1040, 900)
        L = Layer(W, H); L.text((W / 2, H / 2), wd, "black", size, WHITE, anchor="mm")
        M = L.np()[..., 3]; st["M"] = M
        li = p.letter if p.letter >= 0 else len(wd) // 2
        x_start = W / 2 - text_w(wd, "black", size) / 2 + text_w(wd[:li], "black", size)
        x_end = x_start + text_w(wd[li], "black", size)
        sub = (M > 0.5).astype(np.uint8); sub[:, :max(0, int(x_start))] = 0; sub[:, int(x_end):] = 0
        dt = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        ty, tx = np.unravel_index(np.argmax(dt), dt.shape); st["tgt"] = (float(tx), float(ty))
    q = ei(seg(lt, 0.9, 1.3)); z = math.exp(5.2 * q)
    tx, ty = st["tgt"]
    pop = eback(seg(lt, 0, 0.5))
    s_ = z * (0.85 + 0.15 * pop)
    M = cv2.getRotationMatrix2D((tx, ty), 0, s_)
    Mz = cv2.warpAffine(st["M"], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    if q > 0.85: Mz = np.maximum(Mz, seg(q, 0.85, 0.15))
    bgc = np.array(rgb(p.bg), np.float32)
    return bgc * (1 - Mz[..., None]) + O * Mz[..., None]

# ================================================================== 43. comment-drop ====
@effect("comment-drop", "Viewer comments fall from the top, bounce and pile up in front of you.",
        defaults=dict(comments=[["maya.edits", "how is this real"], ["dev_creates", "tutorial please"], ["riya", "saving this right now"],
                                ["k.frames", "okay this is genius"], ["sam_shoots", "who edits for you?"], ["nia.reels", "need this for my page"],
                                ["omar", "teach me"]]), main="comments")
def fx_comment_drop(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("cd")
    if "cards" not in st:
        st["cards"] = []; rng = np.random.default_rng(9)
        for j, (u, t) in enumerate(p.comments):
            w = max(text_w(t, "semi", 36), text_w(u, "bold", 26)) + 150; h = 120
            L = Layer(w + 10, h + 10); L.rrect((0, 0, w, h), 30, fill=(255, 255, 255, 255))
            L.circle(56, 60, 32, fill=PALETTE[j % 7] + (255,)); L.text((56, 60), u[0].upper(), "bold", 30, WHITE, anchor="mm")
            L.text((104, 28), u, "bold", 24, da(150)); L.text((104, 62), t, "semi", 36, DARK)
            st["cards"].append((L.np(), float(rng.uniform(-9, 9)), float(rng.uniform(-90, 90))))
    floor = H - 140; y_rest = floor
    for j, (L, ang, dx) in enumerate(st["cards"]):
        h = L.shape[0]; rest = y_rest - h / 2; y_rest -= h * 0.78
        t0 = 0.2 + j * 0.32; tt = lt - t0
        if tt < 0: continue
        g = 5200; y0 = -200; tfall = math.sqrt(2 * (rest - y0) / g)
        if tt < tfall: y = y0 + 0.5 * g * tt * tt; ang_ = ang * 2.2
        else:
            tb = tt - tfall; y = rest - 120 * abs(math.sin(tb * 9)) * math.exp(-tb * 7); ang_ = ang + 6 * math.sin(tb * 12) * math.exp(-tb * 5)
        shadow(F, W / 2 + dx - L.shape[1] / 2, y - h / 2, L.shape[1] - 10, h - 10, 30, 1, 0.3, 10)
        over_c(F, L, W / 2 + dx, y, angle=ang_)
    return F

# ================================================================== 44. layer-explode ====
def quad_warp(L, dst):
    h, w = L.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(dst))
    P = L.copy(); P[..., :3] *= P[..., 3:4]
    R = cv2.warpPerspective(P, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = R[..., 3:4]; R[..., :3] = np.where(a > 1e-4, R[..., :3] / np.maximum(a, 1e-4), 0)
    return R

@effect("layer-explode", "Exploded 3D view: the shot splits into background, you and a text layer that fan out with labels, then snap back.", mask=True,
        defaults=dict(title="THE BREAKDOWN", labels=["Background", "Subject", "Text"]), main="title")
def fx_layer_explode(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("lx")
    if "txt" not in st:
        size = fit_size([p.title], "black", 900, 120)
        T = Layer(W, H); T.text((W / 2, 300), p.title, "black", size, WHITE, anchor="mm", stroke=3, stroke_fill=(0, 0, 0, 120)); st["txt"] = T.np()
    q = eio(seg(lt, 0.3, 0.9)) * (1 - eio(seg(lt, c.D - 1.0, 0.7)))
    m = c.mask
    if q < 0.01:
        F = O.copy(); over(F, st["txt"], 0, 0, eo(seg(lt, 0, 0.4))); return F
    bgp = clean_plate(O, m)
    layers = [to_layer(bgp), np.dstack([O / 255, m]), st["txt"]]
    out = np.zeros_like(O) + np.array([14, 14, 18], np.float32)
    s = 1 - 0.42 * q; Lb = Layer(W, H)
    for k, L in enumerate(layers):
        w, h = W * s, H * s
        cx, cy = W / 2 + (k - 1) * 150 * q, H / 2 + (1 - k) * 210 * q
        x0, y0 = cx - w / 2, cy - h / 2; sk = 0.10 * q * h
        dst = [[x0, y0 + sk], [x0 + w * (1 - 0.18 * q), y0], [x0 + w * (1 - 0.18 * q), y0 + h], [x0, y0 + h + sk]]
        R = quad_warp(L, dst)
        if k == 0:
            al = R[..., 3:4]; out = out * (1 - al) + R[..., :3] * 255 * al
        else:
            over(out, R, 0, 0)
        Lb.line([tuple(dst[0]), tuple(dst[1]), tuple(dst[2]), tuple(dst[3]), tuple(dst[0])], wa(200 * q), 2)
        lx, ly = dst[0][0] + 24, dst[0][1] + 24
        Lb.rrect((lx, ly, lx + text_w(p.labels[k], "semi", 26) + 30, ly + 46), 14, fill=wa(235 * q))
        Lb.text((lx + 15, ly + 23), p.labels[k], "semi", 26, (16, 16, 20, int(255 * q)), anchor="lm")
    over(out, Lb.np(), 0, 0)
    return out

# ================================================================== 45. sketch-words ====
@effect("sketch-words", "Hand-written words, scribbled underlines and arrows draw themselves around you, pointing at you.", mask=True,
        defaults=dict(words=["this changed everything", "no editing skills", "just one prompt"], color="#FFFFFF"), main="words")
def fx_sketch_words(F, O, c):
    p, lt = c.cfg, c.lt; s = c.subj; col = rgb(p.color)
    spots = [((60, s.hy - s.r * 2.0), "l", (s.hx - s.r * 0.9, s.hy - s.r * 0.9)),
             ((W - 60, s.hy + s.r * 1.2), "r", (s.hx + s.r * 1.1, s.hy + s.r * 0.6)),
             ((60, s.hy + s.r * 3.4), "l", (s.hx - s.r * 0.9, s.hy + s.r * 2.6))]
    L = Layer(W, H)
    for j, wd in enumerate(p.words[:3]):
        (tx, ty), side, (ax, ay) = spots[j]; ty = float(np.clip(ty, 140, H - 200))
        k = seg(lt, 0.3 + 0.7 * j, 0.6)
        if k <= 0: continue
        size = 60; tw_ = text_w(wd, "serifi", size)
        x0 = tx if side == "l" else tx - tw_
        T = Layer(tw_ + 40, 100); T.text((10, 10), wd, "serifi", size, col + (255,), stroke=2, stroke_fill=(0, 0, 0, 120))
        Tn = T.np(); cut = int(Tn.shape[1] * eo(k)); Tn[:, cut:, 3] = 0
        over(F, Tn, x0 - 10, ty - 10)
        u = seg(lt, 0.3 + 0.7 * j + 0.45, 0.35)
        if u > 0:
            pts = [(x0 + i * tw_ / 24, ty + 84 + 7 * math.sin(i * 1.3 + j)) for i in range(int(25 * u))]
            if len(pts) > 1: L.line(pts, col + (255,), 5)
        v = seg(lt, 0.3 + 0.7 * j + 0.65, 0.4)
        if v > 0:
            sx_, sy_ = (x0 + tw_ * 0.5, ty + 100)
            ctrl = ((sx_ + ax) / 2 + (80 if side == "l" else -80), max(sy_, ay) + 60)
            pts = []
            for i in range(int(30 * v) + 1):
                t = i / 30; pts.append(((1 - t) ** 2 * sx_ + 2 * (1 - t) * t * ctrl[0] + t * t * ax, (1 - t) ** 2 * sy_ + 2 * (1 - t) * t * ctrl[1] + t * t * ay))
            if len(pts) > 1:
                L.line(pts, col + (255,), 5)
                if v >= 1:
                    (bx, by), (ex, ey) = pts[-3], pts[-1]; ang = math.atan2(ey - by, ex - bx)
                    for d in (2.5, -2.5):
                        L.line([(ex, ey), (ex - 34 * math.cos(ang + d / 6), ey - 34 * math.sin(ang + d / 6))], col + (255,), 5)
    over(F, L.np(), 0, 0)
    return F


def grid_xy():
    if "g" not in _GRID:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); _GRID["g"] = (xx, yy)
    return _GRID["g"]
_GRID = {}

# ================================================================== 46. blade-slice ====
@effect("blade-slice", "A glowing blade slashes across the frame; the two halves slide apart to reveal a title card, then slam back together.",
        defaults=dict(title="PLOT TWIST", sub="wait for it", accent="#FF3B30", angle=-24), main="title")
def fx_blade(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("blade"); acc = rgb(p.accent)
    xx, yy = grid_xy(); a_ = math.radians(p.angle); dx_, dy_ = math.cos(a_), math.sin(a_); nx, ny = -dy_, dx_
    cx, cy = W / 2, H / 2
    if "card" not in st:
        Cd = grad(W, H, (12, 12, 16), tuple(int(v * 0.55) for v in acc), angle=70)
        size = fit_size([p.title], "black", 980, 170)
        L = Layer(W + 200, 400); L.text((W / 2 + 100, 150), p.title, "black", size, WHITE, anchor="mm"); L.text((W / 2 + 100, 285), p.sub, "serifi", 60, wa(220), anchor="mm")
        over_c(Cd, L.np(), W / 2, H / 2 + 30, angle=-p.angle); st["card"] = Cd
        st["side"] = (xx - cx) * nx + (yy - cy) * ny; st["along"] = (xx - cx) * dx_ + (yy - cy) * dy_
    tc = 0.5; k = seg(lt, tc, 0.12)
    op_ = eo(seg(lt, tc + 0.12, 0.45)) * (1 - ei(seg(lt, c.D - 0.8, 0.25)))
    if op_ <= 0.001: F = O.copy()
    else:
        F = st["card"].copy(); sep = 250 * op_; slide = 70 * op_
        for sgn in (1, -1):
            sx, sy = nx * sep * sgn + dx_ * slide * sgn, ny * sep * sgn + dy_ * slide * sgn
            I = shift_img(O, sx, sy)
            sd = st["side"] - (nx * sx + ny * sy)
            mk = np.clip(sd * sgn + 0.5, 0, 1)[..., None]
            edge = np.clip(1 - np.abs(sd) / 40, 0, 1)[..., None] * mk
            F = F * (1 - mk) + (I * (1 - 0.35 * edge)) * mk
    glow = 0.0
    if lt >= tc: glow = 1 - seg(lt, tc + 0.12, 0.35)
    cl = seg(lt, c.D - 0.55, 0.1)
    if 0 < cl < 1: glow = max(glow, 1 - abs(cl - 0.5) * 2)
    if glow > 0:
        vis = (st["along"] < -1200 + 2400 * k)[..., None] if k < 1 else 1
        ln = np.exp(-(st["side"] ** 2) / (2 * 3.0 ** 2))[..., None] + 0.5 * np.exp(-(st["side"] ** 2) / (2 * 22 ** 2))[..., None]
        F = F + (np.array([255, 245, 235], np.float32) * ln * glow) * vis
    return F

# ================================================================== 47. head-stack ====
@effect("head-stack", "Copies of your head stack up into a wobbling totem above you, popping in one by one.", mask=True,
        defaults=dict(count=3, delay=0.15))
def fx_head_stack(F, O, c):
    p, lt = c.cfg, c.lt; s = c.subj
    rr = s.r * 1.3
    for j in reversed(range(p.count)):
        a = seg(lt, 0.3 + 0.28 * j, 0.45)
        if a <= 0: continue
        sm, smk = delayed(c, p.delay * (j + 1))
        if smk is None: continue
        hx2, hy2, r2 = s.hx / 2, s.hy / 2, rr / 2
        x0, y0, x1, y1 = int(max(0, hx2 - r2)), int(max(0, hy2 - r2)), int(min(W // 2, hx2 + r2)), int(min(H // 2, hy2 + r2))
        if x1 - x0 < 10 or y1 - y0 < 10: continue
        im = cv2.resize(sm[y0:y1, x0:x1].astype(np.float32), ((x1 - x0) * 2, (y1 - y0) * 2))
        mk = cv2.resize(smk[y0:y1, x0:x1], ((x1 - x0) * 2, (y1 - y0) * 2))
        yy_, xx_ = np.mgrid[0:mk.shape[0], 0:mk.shape[1]].astype(np.float32)
        circ = np.clip((rr - np.sqrt((xx_ - (hx2 - x0) * 2) ** 2 + (yy_ - (hy2 - y0) * 2) ** 2)) / 20, 0, 1)
        Lh = np.dstack([im / 255, mk * circ])
        sc = 0.84 ** (j + 1)
        ty = s.hy - sum(s.r * 1.85 * 0.84 ** (i + 1) for i in range(j + 1)) - 500 * (1 - eo(a))
        over_c(F, Lh, s.hx + 10 * math.sin(lt * 2.5 + j), ty, op=eo(a * 2), scale=sc * (0.7 + 0.3 * eback(a)), angle=7 * math.sin(lt * 3 + j * 1.7))
    return F

# ================================================================== 48. rack-focus ====
@effect("rack-focus", "Cinematic rack focus: a foreground title is sharp first, then focus pulls to you and an AF box locks on.", mask=True,
        defaults=dict(text="FOCUS", at=1.1), main="text")
def fx_rack_focus(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("rf")
    if "T" not in st:
        size = fit_size([p.text.upper()], "black", 980, 260)
        L = Layer(W, 420); L.text((W / 2, 210), p.text.upper(), "black", size, WHITE, anchor="mm", stroke=3, stroke_fill=(0, 0, 0, 90)); st["T"] = L.np()
    q = eio(seg(lt, p.at, 0.7)); m = c.mask[..., None]
    bg = fast_blur(O, 14 - 6 * q); sub = fast_blur(O, 10 * (1 - q)) if q < 0.98 else O
    F = bg * (1 - m) + sub * m
    T = st["T"]; sig = 14 * q
    if sig > 0.5: T = cv2.GaussianBlur(T, (0, 0), sig)
    a = eo(seg(lt, 0, 0.5))
    over_c(F, T, W / 2, 1450, op=a * (1 - 0.35 * q), scale=1 + 0.08 * q)
    af = seg(lt, p.at + 0.6, 0.25)
    if af > 0:
        s = c.subj; bw = s.r * 2.2 * (1.25 - 0.25 * eo(af)); x0, y0 = s.hx - bw / 2, s.hy - bw / 2; L = Layer(W, H); cl = 46
        col = (255, 214, 10, int(255 * (1 - seg(lt, p.at + 1.8, 0.4))))
        for (px, py, sx, sy) in [(x0, y0, 1, 1), (x0 + bw, y0, -1, 1), (x0, y0 + bw, 1, -1), (x0 + bw, y0 + bw, -1, -1)]:
            L.line([(px + sx * cl, py), (px, py), (px, py + sy * cl)], col, 5, joint=None)
        over(F, L.np(), 0, 0)
    return F

# ================================================================== 49. banknote ====
@effect("banknote", "You become the portrait on an engraved (fictional) banknote: guilloche patterns, serials, holographic strip.",
        mask=True, defaults=dict(bank="STUDIO RESERVE NOTE", value="100", serial="SR 260113 A", motto="IN CONTENT WE TRUST",
                                 ink="#1F5A45", paper="#E4E8D2"), main="bank")
def fx_banknote(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("note"); ink = np.array(rgb(p.ink), np.float32); paper = np.array(rgb(p.paper), np.float32)
    ox, oy, ax_, ay_ = W / 2, 860, 330, 430
    if "bg" not in st:
        xx, yy = grid_xy()
        g1 = np.abs(np.sin(xx / 9 + 5 * np.sin(yy / 120))) < 0.10
        g2 = np.abs(np.sin(yy / 9 + 5 * np.sin(xx / 140))) < 0.10
        pat = (g1 | g2).astype(np.float32) * 0.22
        bg = paper * (1 - pat[..., None]) + ink * pat[..., None]
        L = Layer(W, H); inkA = tuple(int(v) for v in ink) + (255,)
        for ins, wd in [(36, 6), (56, 2), (70, 1)]: L.rect((ins, ins, W - ins, ins + wd), inkA); L.rect((ins, H - ins - wd, W - ins, H - ins), inkA); L.rect((ins, ins, ins + wd, H - ins), inkA); L.rect((W - ins - wd, ins, W - ins, H - ins), inkA)
        for cx_, cy_ in [(140, 140), (W - 140, 140), (140, H - 140), (W - 140, H - 140)]:
            L.circle(cx_, cy_, 66, fill=tuple(int(v) for v in paper) + (255,), outline=inkA, width=3); L.text((cx_, cy_), p.value, "black", 52, inkA, anchor="mm")
        L.text((W / 2, 170), p.bank, "serifb", 46, inkA, anchor="mm", track=3)
        L.text((W / 2, 1420), p.value, "black", 210, inkA, anchor="mm")
        L.text((W / 2, 1580), p.motto, "serifi", 40, inkA, anchor="mm")
        L.text((W - 120, 250), p.serial, "mono", 34, (170, 40, 40, 255), anchor="rm"); L.text((120, H - 250), p.serial, "mono", 34, (170, 40, 40, 255), anchor="lm")
        L.ellipse((ox - ax_ - 22, oy - ay_ - 22, ox + ax_ + 22, oy + ay_ + 22), outline=inkA, width=6)
        L.ellipse((ox - ax_ - 36, oy - ay_ - 36, ox + ax_ + 36, oy + ay_ + 36), outline=inkA, width=2)
        over(bg, L.np(), 0, 0); st["bg"] = bg
        Mo = np.zeros((H, W), np.float32); cv2.ellipse(Mo, (int(ox), int(oy)), (ax_, ay_), 0, 0, 360, 1.0, -1); st["oval"] = cv2.GaussianBlur(Mo, (0, 0), 1.5)[..., None]
        st["line"] = 0.5 + 0.5 * np.cos(2 * np.pi * yy / 7 + 2.2 * np.sin(2 * np.pi * xx / 260))
    F = st["bg"].copy()
    vid = crop_cover(O, 2 * ax_ + 4, 2 * ay_ + 4, c.subj.hx, c.subj.hy + c.subj.r * 0.7)
    g = (vid @ LUMA) / 255; g = np.clip((g - 0.5) * 1.35 + 0.55, 0, 1)
    y0, x0 = int(oy - ay_ - 2), int(ox - ax_ - 2); h, w = g.shape
    ln = st["line"][y0:y0 + h, x0:x0 + w]
    e = np.clip(((1 - g) * 1.2 - ln) * 4 + 0.5, 0, 1)[..., None]
    eng = paper * (1 - e) + ink * e
    mo = st["oval"][y0:y0 + h, x0:x0 + w]
    reg = F[y0:y0 + h, x0:x0 + w]; reg[:] = reg * (1 - mo) + eng * mo
    xs = 180 + int(20 * math.sin(lt))
    hue = (np.arange(H, dtype=np.float32) / 260 + lt * 0.6) % 1
    holo = np.stack([0.5 + 0.5 * np.sin(2 * np.pi * (hue + k / 3)) for k in range(3)], 1) * 255
    F[:, xs:xs + 46] = F[:, xs:xs + 46] * 0.45 + holo[:, None, :] * 0.55
    xx, yy = grid_xy(); band = (lt * 900) % (W + 1400) - 700
    sh = np.clip(1 - np.abs(xx + yy * 0.5 - band - 500) / 90, 0, 1)[..., None] * 0.35
    return F + (255 - F) * sh

# ================================================================== 50. voice-note ====
@effect("voice-note", "Chat voice-message bubble plays your actual audio: live waveform fills as it plays, timer counts up.",
        defaults=dict(color="#0A84FF", label="Voice message"), main="label")
def fx_voice_note(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("vn"); col = rgb(p.color)
    w, h = 940, 190; x, y = (W - w) / 2, 1540
    if "av" not in st: st["av"] = round_img(c.frames(1, 160, 160, key="vn")[0], 112)
    a = seg(lt, 0, 0.45); op = eo(a * 1.5); sc = 0.7 + 0.3 * eback(a)
    prog = clamp01((lt - 0.45) / max(0.5, c.D - 0.8))
    L = Layer(w, h + 30); L.rrect((0, 0, w, h), 60, fill=col + (255,)); L.poly([(w - 70, h - 10), (w - 20, h + 26), (w - 120, h - 10)], fill=col + (255,))
    L.circle(95, 95, 52, fill=WHITE)
    if prog <= 0 or prog >= 1: L.poly([(80, 68), (80, 122), (124, 95)], fill=col + (255,))
    else: L.rect((76, 70, 90, 120), fill=col + (255,)); L.rect((100, 70, 114, 120), fill=col + (255,))
    i0 = int(c.seg_a * c.fps); ev = c.env[i0:i0 + max(2, int(c.D * c.fps))]
    nb = 44; x0, x1 = 175, w - 200; bw = (x1 - x0) / nb
    for j in range(nb):
        v = float(ev[min(len(ev) - 1, int(j * len(ev) / nb))]) if len(ev) else 0.4
        hh = 10 + 90 * min(1, v)
        L.rrect((x0 + j * bw + 2, 95 - hh / 2, x0 + (j + 1) * bw - 3, 95 + hh / 2), 4, fill=wa(255 if j / nb <= prog else 110))
    secs = lt - 0.45 if prog > 0 else c.D - 0.8
    L.text((x1 + 22, 95), f"0:{int(max(0, secs)):02d}", "semi", 30, WHITE, anchor="lm")
    Ln = L.np()
    over_c(F, Ln, x + w / 2, y + (h + 30) / 2, op=op, scale=sc)
    over_c(F, st["av"], x + w - 62, y + 95, op=op, scale=sc)
    Lb = Layer(300, 40); Lb.text((300, 0), p.label + " · " + ("Played" if prog >= 1 else "Playing" if prog > 0 else "Sent"), "medium", 24, wa(220), anchor="rt", stroke=1, stroke_fill=(0, 0, 0, 100))
    over(F, Lb.np(), x + w - 300, y + h + 40, op)
    return F

# ================================================================== 51. impact-frame ====
@effect("impact-frame", "Anime impact frames on the beat: inverted ink frame with speed lines, red two-tone, flash and screen shake.", mask=True,
        defaults=dict(beat=0.9, color="#FF2D2D"))
def fx_impact(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("imp"); col = np.array(rgb(p.color), np.float32)
    if "lines" not in st:
        st["lines"] = []
        for v in range(3):
            rng = np.random.default_rng(v + 1); m = np.zeros((H, W), np.float32)
            cx, cy = c.subj.hx, c.subj.hy
            for _ in range(70):
                a = rng.uniform(0, 2 * math.pi); w_ = rng.uniform(0.004, 0.02); r0 = rng.uniform(330, 520)
                pts = np.array([[cx + r0 * math.cos(a), cy + r0 * math.sin(a)], [cx + 2400 * math.cos(a - w_), cy + 2400 * math.sin(a - w_)],
                                [cx + 2400 * math.cos(a + w_), cy + 2400 * math.sin(a + w_)]], np.int32)
                cv2.fillConvexPoly(m, pts, 1.0, lineType=cv2.LINE_AA)
            st["lines"].append(m[..., None])
    if lt < 0.4: return O
    k = int((lt - 0.4) / p.beat); since = lt - 0.4 - k * p.beat
    g = ((O @ LUMA) / 255)[..., None]
    ink = np.clip((g - 0.45) * 9 + 0.5, 0, 1)
    if since < 0.05:
        Ls = st["lines"][k % 3]; return np.repeat((1 - ink) * 255 * (1 - Ls), 3, axis=2)
    if since < 0.10:
        return (1 - ink) * np.array([14, 6, 6], np.float32) + ink * col
    if since < 0.14:
        return O + (255 - O) * 0.55
    amp = 26 * math.exp(-(since - 0.14) * 9)
    S_ = shift_img(O, amp * math.sin(lt * 90), amp * math.cos(lt * 70))
    ca = int(10 * math.exp(-(since - 0.14) * 8))
    if ca > 0:
        S_ = S_.copy(); S_[..., 0] = np.roll(S_[..., 0], -ca, 1); S_[..., 2] = np.roll(S_[..., 2], ca, 1)
    return S_

# ================================================================== 52. dolly-zoom ====
@effect("dolly-zoom", "Vertigo dolly zoom: you stay the same size while the room stretches away behind you.", mask=True,
        defaults=dict(amount=0.45))
def fx_dolly(F, O, c):
    p, lt = c.cfg, c.lt; m = c.mask[..., None]
    bgp = clean_plate(O, c.mask)
    s = 1 + p.amount * eio(seg(lt, 0.2, max(0.5, c.D - 0.5)))
    M = cv2.getRotationMatrix2D((c.subj.hx, c.subj.hy), 0, s)
    bg = cv2.warpAffine(bgp, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    st = c.st("dz")
    if "v" not in st: st["v"] = vignette(0.6)
    vv = 1 - (1 - st["v"]) * (s - 1) / max(0.01, p.amount)
    return (bg * (1 - m) + O * m) * vv

# ================================================================== 53. press-font ====
@effect("press-font", "A hydraulic press slams down on your title, squashing it flat before it springs back with a wobble.",
        defaults=dict(text="PRESSED", color="#FFD60A"), main="text")
def fx_press_font(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("press")
    base_y = 470
    if "T" not in st:
        size = fit_size([p.text.upper()], "black", 900, 190)
        L = Layer(W, 260); L.text((W / 2, 130), p.text.upper(), "black", size, rgba(p.color), anchor="mm", stroke=6, stroke_fill=(18, 18, 22, 255))
        st["T"] = L.np(); st["gh"] = size * 0.74
        Pl = Layer(980, 520); Pl.rect((470, 0, 510, 420), fill=(150, 152, 160, 255)); Pl.rect((480, 0, 488, 420), fill=(210, 212, 220, 255))
        Pl.rrect((0, 400, 980, 500), 14, fill=(70, 72, 80, 255)); Pl.rrect((0, 400, 980, 424), 10, fill=(130, 132, 142, 255))
        st["P"] = Pl.np()
    T = st["T"]; th_ = T.shape[0]; gh = st["gh"]
    t_down, t_hit, t_up, t_rel = 0.6, 0.8, 1.25, 1.35
    sy, sx = 1.0, 1.0
    if t_hit <= lt < t_rel: sy, sx = 0.32, 1.28
    elif lt >= t_rel:
        q = lt - t_rel; el = 1 - math.exp(-q * 6) * math.cos(q * 22)
        sy = 0.32 + 0.68 * el; sx = 1.28 - 0.28 * el
    gtop = lambda s_: base_y + th_ / 2 - th_ * s_ + (th_ / 2 - gh / 2) * s_
    if lt < t_down: pb = -60
    elif lt < t_hit: pb = lerp(-60, gtop(1.0), ei(seg(lt, t_down, t_hit - t_down)))
    elif lt < t_up: pb = gtop(0.32)
    else: pb = lerp(gtop(0.32), -60, eo(seg(lt, t_up, 0.5)))
    a = seg(lt, 0.1, 0.4)
    if a > 0:
        R = cv2.resize(T, (max(1, int(W * sx)), max(1, int(th_ * sy))), interpolation=cv2.INTER_AREA)
        over(F, R, W / 2 - R.shape[1] / 2, base_y + th_ / 2 - R.shape[0], eo(a * 2))
    over(F, st["P"], W / 2 - 490, pb - 500)
    q = seg(lt, t_hit, 0.5)
    if 0 < q < 1:
        D = Layer(W, 300)
        for j in range(8):
            sd = -1 if j % 2 == 0 else 1; dx = (480 + 140 * q + j * 14) * sd; r_ = 18 + 40 * q
            D.circle(W / 2 + dx, 150 + (j % 4) * 12 - 30 * q, r_, fill=(235, 230, 220, int(150 * (1 - q))))
        over(F, D.np(), 0, base_y + th_ * 0.25 - 150)
    return F

# ================================================================== 54. paper-tear ====
@effect("paper-tear", "The frame rips open like paper along a jagged tear, revealing a title page underneath.",
        defaults=dict(title="PART TWO", sub="the real story"), main="title")
def fx_paper_tear(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("tear")
    xx, yy = grid_xy()
    if "tear" not in st:
        rng = np.random.default_rng(11); walk = np.cumsum(rng.normal(0, 3.2, W)); walk -= np.linspace(walk[0], walk[-1], W)
        tear = H * 0.47 + walk + rng.normal(0, 1.2, W); st["tear"] = tear.astype(np.float32)
        st["fib"] = (8 + 8 * cv2.resize(rng.random((1, W // 6)).astype(np.float32), (W, 1))[0]).astype(np.float32)
        bg = paper_bg(3); L = Layer(W, 500)
        size = fit_size([p.title], "serifb", 980, 150)
        L.text((W / 2, 180), p.title, "serifb", size, (26, 24, 22, 255), anchor="mm"); L.text((W / 2, 320), p.sub, "serifi", 62, (90, 70, 60, 255), anchor="mm")
        over(bg, L.np(), 0, H * 0.47 - 250); st["bg"] = bg
    q = eo(seg(lt, 0.5, 0.7))
    if q <= 0.001: return O
    tear = st["tear"][None, :]; fib = st["fib"][None, :]
    mt = np.clip(tear - yy + 0.5, 0, 1); et = np.clip(tear + fib - yy, 0, 1) * (1 - mt)
    mb = np.clip(yy - tear + 0.5, 0, 1); eb = np.clip(yy - (tear - fib), 0, 1) * (1 - mb)
    paperc = np.array([246, 242, 232], np.float32)
    top = np.dstack([(O * mt[..., None] + paperc * et[..., None]) / 255 / np.maximum(mt + et, 1e-4)[..., None], mt + et])
    bot = np.dstack([(O * mb[..., None] + paperc * eb[..., None]) / 255 / np.maximum(mb + eb, 1e-4)[..., None], mb + eb])
    F = st["bg"].copy()
    for Lp, dy, ang in [(top, -300 * q, -2.5 * q), (bot, 320 * q, 1.8 * q)]:
        M = cv2.getRotationMatrix2D((W / 2, H * 0.47), ang, 1.0); M[1, 2] += dy
        P = Lp.copy(); P[..., :3] *= P[..., 3:4]
        R = cv2.warpAffine(P, M, (W, H), borderValue=(0, 0, 0, 0)); al = R[..., 3:4]
        shadow_a = cv2.GaussianBlur(al, (0, 0), 14)
        if shadow_a.ndim == 2: shadow_a = shadow_a[..., None]
        F = F * (1 - shadow_a * 0.35)
        F = F * (1 - al) + R[..., :3] * 255
    return F

# ================================================================== 55. blow-away ====
@effect("blow-away", "You disintegrate into particles that blow away on the wind, leaving the empty room behind.", mask=True,
        defaults=dict(direction=1, start=0.5, sweep=1.3))
def fx_blow_away(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("blow")
    xx, yy = grid_xy()
    if "N" not in st:
        rng = np.random.default_rng(21)
        st["N"] = cv2.resize(rng.random((H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        gx = (xx / W) if p.direction > 0 else (1 - xx / W)
        st["T"] = p.start + p.sweep * (gx * 0.7 + (1 - yy / H) * 0.3) + 0.35 * st["N"]
        ys, xs = np.mgrid[0:H:5, 0:W:5]; st["pp"] = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        st["vel"] = np.stack([p.direction * (260 + rng.random(len(xs.ravel())) * 360), -(60 + rng.random(len(xs.ravel())) * 220)], 1).astype(np.float32)
        st["ph"] = rng.random(len(xs.ravel())).astype(np.float32) * 6.28
    m = c.mask
    bgp = clean_plate(O, m)
    vis = np.clip((st["T"] - lt) * 12, 0, 1) * m
    out = bgp * (1 - vis[..., None]) + O * vis[..., None]
    pp = st["pp"]; ix, iy = pp[:, 0].astype(int), pp[:, 1].astype(int)
    tp = st["T"][iy, ix]; age = lt - tp
    sel = (m[iy, ix] > 0.5) & (age > 0) & (age < 1.6)
    if sel.any():
        a_ = age[sel]; v = st["vel"][sel]; ph = st["ph"][sel]
        px = pp[sel, 0] + v[:, 0] * a_ + 60 * np.sin(a_ * 4 + ph)
        py = pp[sel, 1] + v[:, 1] * a_ - 120 * a_ * a_ + 40 * np.cos(a_ * 3 + ph)
        colr = O[iy[sel], ix[sel]]; alpha = np.clip(1 - a_ / 1.6, 0, 1)
        acc = np.zeros((H, W, 3), np.float32); wgt = np.zeros((H, W), np.float32)
        for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            qx = np.clip((px + ox).astype(int), 0, W - 1); qy = np.clip((py + oy).astype(int), 0, H - 1)
            ok = (px + ox >= 0) & (px + ox < W) & (py + oy >= 0) & (py + oy < H)
            np.add.at(acc, (qy[ok], qx[ok]), colr[ok] * alpha[ok, None]); np.add.at(wgt, (qy[ok], qx[ok]), alpha[ok])
        w_ = np.clip(wgt, 0, 1)[..., None]
        out = out * (1 - w_) + acc / np.maximum(wgt, 1e-4)[..., None] * w_
    return out

# ================================================================== 56. liquid-glass ====
@effect("liquid-glass", "Liquid-glass UI: a refractive glass lens glides over the shot magnifying it, above a glass tab bar with a sliding selector.",
        defaults=dict(tabs=["Home", "Search", "Create", "Profile"], radius=170), main="tabs")
def fx_liquid_glass(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("lq"); R = int(p.radius)
    if "uv" not in st:
        yy, xx = np.mgrid[-R - 4:R + 4, -R - 4:R + 4].astype(np.float32); d = np.sqrt(xx ** 2 + yy ** 2) / R
        f = 0.62 + 0.38 * np.clip(d, 0, 1) ** 2 + 0.22 * np.clip((d - 0.82) / 0.18, 0, 1) ** 2
        st["uv"] = (xx, yy, f, d)
        rim = Layer(2 * R + 8, 2 * R + 8); cc = R + 4
        rim.arc((4, 4, 2 * R + 4, 2 * R + 4), 180, 300, wa(230), 5); rim.arc((4, 4, 2 * R + 4, 2 * R + 4), 0, 120, wa(90), 3)
        rim.circle(cc, cc, R - 1, outline=wa(70), width=2); st["rim"] = rim.np()
    xx, yy, f, d = st["uv"]
    a = eo(seg(lt, 0, 0.6))
    cx = W / 2 + 300 * math.sin(lt * 0.9); cy = 760 + 330 * math.sin(lt * 0.6 + 0.8)
    chans = []
    for k, dk in enumerate((0.985, 1.0, 1.016)):
        mx = (cx + xx * f * dk).astype(np.float32); my = (cy + yy * f * dk).astype(np.float32)
        chans.append(cv2.remap(np.ascontiguousarray(O[..., k]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    lens = np.dstack(chans) * 1.04 + 8
    al = np.clip((1 - d) * R / 1.5, 0, 1) * a
    shadow(F, cx - R, cy - R, 2 * R, 2 * R, R, a, 0.35, 24)
    over(F, np.dstack([lens / 255, al]), cx - R - 4, cy - R - 4)
    over(F, st["rim"], cx - R - 4, cy - R - 4, a)
    bx, by, bw, bh = 110, 1690, 860, 140
    glass(F, bx, by, bw, bh, r=70, op=a, tone="clear", blur=18, border=0.55)
    n = len(p.tabs); tw_ = bw / n
    pos = (lt * 1.1) % n; i0 = int(pos); mv = eback(clamp01((pos - i0 - 0.6) / 0.4), 1.4); i1 = (i0 + 1) % n
    selx = bx + lerp(i0, i1, mv) * tw_
    glass(F, selx + 10, by + 12, tw_ - 20, bh - 24, r=58, op=a, tone="light", blur=10, border=0.6, shade=False)
    L = Layer(bw, bh)
    for j, t_ in enumerate(p.tabs):
        cxx = (j + 0.5) * tw_; on = abs(selx - bx - j * tw_) < tw_ / 2; colr = DARK if on else WHITE
        [lambda: L.rrect((cxx - 18, 30, cxx + 18, 62), 8, outline=colr, width=4),
         lambda: (L.circle(cxx - 4, 44, 15, outline=colr, width=4), L.line([(cxx + 7, 55), (cxx + 18, 66)], colr, 4)),
         lambda: (L.rect((cxx - 2, 28, cxx + 2, 64), colr), L.rect((cxx - 18, 44, cxx + 18, 48), colr)),
         lambda: (L.circle(cxx, 38, 10, outline=colr, width=4), L.arc((cxx - 20, 50, cxx + 20, 86), 180, 360, colr, 4))][j % 4]()
        L.text((cxx, 100), t_, "semi", 22, colr, anchor="mm")
    over(F, L.np(), bx, by, a)
    return F


# ================================================================== 57. glitch ====
@effect("glitch", "Digital glitch bursts: RGB split, sliced scanline displacement, corrupted blocks and flicker on a rhythm.",
        defaults=dict(intensity=1.0, every=0.8))
def fx_glitch(F, O, c):
    p, lt = c.cfg, c.lt
    rng = np.random.default_rng(c.i * 7919 + 13)
    burst = 1.0 if (lt % p.every) < 0.2 else 0.18
    k = burst * p.intensity
    X = O.copy(); dx = int(rng.integers(4, 30) * k)
    if dx: X[..., 0] = np.roll(O[..., 0], -dx, 1); X[..., 2] = np.roll(O[..., 2], dx, 1)
    for _ in range(int(7 * k) + 1):
        y0 = int(rng.integers(0, H - 20)); hh = int(rng.integers(8, 140)); sh = int(rng.integers(-110, 110) * k)
        X[y0:y0 + hh] = np.roll(X[y0:y0 + hh], sh, 1)
    if burst > 0.5:
        for _ in range(int(rng.integers(3, 8))):
            bw, bh = int(rng.integers(60, 380)), int(rng.integers(20, 120)); x0, y0 = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
            mode = int(rng.integers(0, 3)); blk = X[y0:y0 + bh, x0:x0 + bw]
            if mode == 0: X[y0:y0 + bh, x0:x0 + bw] = 255 - blk
            elif mode == 1: X[y0:y0 + bh, x0:x0 + bw] = blk * np.array([0.2, 1.0, 1.0], np.float32) + np.array([0, 40, 60], np.float32)
            else:
                sx, sy = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh)); X[y0:y0 + bh, x0:x0 + bw] = O[sy:sy + bh, sx:sx + bw]
        if rng.random() < 0.3: X = np.roll(X, int(rng.integers(-60, 60)), 0)
    X[::3] *= 0.86
    return X

# ================================================================== 58. vhs ====
@effect("vhs", "Old VHS tape look: chroma bleed, wobbling lines, tracking noise band, scanlines and a PLAY / date on-screen display.",
        defaults=dict(label="PLAY", date="", time="AM 10:42"), main="label")
def fx_vhs(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("vhs")
    if "xx" not in st:
        st["xx"], st["yy"] = grid_xy(); st["v"] = vignette(0.5)
        try: d = datetime.date.fromisoformat(p.date) if p.date else datetime.date.today()
        except Exception: d = datetime.date.today()
        L = Layer(W, H)
        L.text((70, 110), p.label + " ▶", "mono", 56, WHITE, stroke=2, stroke_fill=(0, 0, 0, 120))
        L.text((70, H - 210), p.time, "mono", 48, WHITE, stroke=2, stroke_fill=(0, 0, 0, 120))
        L.text((70, H - 150), d.strftime("%b. %d %Y").upper(), "mono", 48, WHITE, stroke=2, stroke_fill=(0, 0, 0, 120))
        st["osd"] = L.np()
    X = cv2.resize(cv2.resize(O, (W // 3, H), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    Y = cv2.cvtColor(np.clip(X, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y[..., 1] = np.roll(cv2.blur(Y[..., 1], (41, 1)), 8, 1); Y[..., 2] = np.roll(cv2.blur(Y[..., 2], (41, 1)), 8, 1)
    X = cv2.cvtColor(np.clip(Y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32)
    yv = np.arange(H, dtype=np.float32)
    band_y = (lt * 380) % (H + 400) - 200
    band = np.exp(-((yv - band_y) / 40) ** 2)
    off = 2.5 * np.sin(yv / 28 + lt * 9) + band * 38 * np.sin(yv * 0.9 + lt * 40)
    X = cv2.remap(X, (st["xx"] + off[:, None]).astype(np.float32), st["yy"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    X = grade(X, sat=0.78, contrast=0.92, tint=(1.05, 1.0, 0.92), lift=14)
    rng = np.random.default_rng(c.i)
    X += cv2.resize(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (W, H))[..., None] * 9
    X += band[:, None, None] * cv2.resize(rng.random((H // 8, W // 2)).astype(np.float32), (W, H))[..., None] * 110
    X[::2] *= 0.9
    X *= st["v"]
    over(X, st["osd"], 0, 0, 1.0 if int(lt * 1.5) % 4 else 0.85)
    return X

# ================================================================== 59. freeze-frame ====
@effect("freeze-frame", "'Yep, that's me' freeze frame: the shot freezes, you pop out as a sticker cut-out and a highlighted name tag slides in.",
        mask=True, defaults=dict(title="That's me.", sub="probably overthinking this", at=0.6, color="#FFD60A"), main="title")
def fx_freeze(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("frz")
    if lt < p.at: return O
    if "O" not in st:
        st["O"] = O.copy(); st["m"] = c.mask.copy(); st["h"] = (c.subj.hx, c.subj.hy, c.subj.r)
        ms = cv2.resize(st["m"], (W // 2, H // 2)); dil = cv2.dilate(ms, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        st["ol"] = cv2.resize(dil, (W, H))[..., None]
        st["sh"] = cv2.GaussianBlur(st["ol"], (0, 0), 16)
        if st["sh"].ndim == 2: st["sh"] = st["sh"][..., None]
        tw_ = text_w(p.title, "black", 78) + 60
        T = Layer(tw_ + 40, 220); T.rect((0, 10, tw_, 120), fill=rgba(p.color)); T.text((30, 65), p.title, "black", 78, (16, 16, 20, 255), anchor="lm")
        T.text((6, 150), p.sub, "semi", 36, WHITE, stroke=2, stroke_fill=(0, 0, 0, 160)); st["tag"] = T.np()
    Fz, m = st["O"], st["m"][..., None]
    t = lt - p.at; z = 1 + 0.05 * eo(seg(t, 0, 3))
    hx, hy, r = st["h"]
    bg = grade(Fz, sat=0.15, bright=0.6)
    pop = eback(seg(t, 0.05, 0.35))
    sub = Fz
    out = bg * (1 - st["sh"] * 0.5 * pop)
    ol = st["ol"] * pop
    out = out * (1 - ol) + 255 * ol
    out = out * (1 - m) + sub * m
    M = cv2.getRotationMatrix2D((hx, hy), 0, z * (1 + 0.03 * pop)); out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    fl = 1 - seg(t, 0, 0.18)
    if fl > 0: out = out + (255 - out) * 0.7 * fl
    k = eo(seg(t, 0.35, 0.45))
    if k > 0: over(out, st["tag"], 60 - 700 * (1 - k), min(H - 260, hy + r * 2.6))
    return out

# ================================================================== 60. film-burn ====
@effect("film-burn", "Analogue film look with an organic orange film burn that flares across the frame, gate weave, grain and dust.",
        defaults=dict(peak=0.5))
def fx_film_burn(F, O, c):
    p, lt = c.cfg, c.lt; st = c.st("burn")
    if "n" not in st:
        rng = np.random.default_rng(31); acc = np.zeros((H // 4, W // 4), np.float32)
        for o, amp in [(64, 1.0), (24, 0.5), (8, 0.25)]:
            acc += cv2.resize(rng.random((H // 4 // o + 2, W // 4 // o + 2)).astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_CUBIC) * amp
        st["n"] = acc / 1.75; st["gx"] = np.linspace(0, 1, W // 4, dtype=np.float32)[None, :]
        st["grain"] = [cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))[..., None] for _ in range(4)]
    rng = np.random.default_rng(c.i + 5)
    X = shift_img(O, float(rng.normal(0, 1.5)), float(rng.normal(0, 1.5)))
    X = grade(X, sat=0.9, contrast=1.05, tint=(1.06, 1.0, 0.9), lift=6)
    pk = clamp01(1 - abs(lt - c.D * p.peak) / max(0.4, c.D * 0.42))
    n = np.roll(st["n"], int(lt * 18), 0)
    B = np.clip((n * 0.9 + st["gx"] * 1.1 - (1.75 - 1.6 * pk)) * 3.2, 0, 1)
    B = cv2.resize(B, (W, H))[..., None]
    colr = np.where(B < 0.4, B / 0.4 * np.array([150, 20, 0], np.float32),
                    np.where(B < 0.75, np.array([150, 20, 0], np.float32) + (B - 0.4) / 0.35 * np.array([105, 120, 20], np.float32),
                             np.array([255, 140, 20], np.float32) + (B - 0.75) / 0.25 * np.array([0, 110, 200], np.float32)))
    X = 255 - (255 - X) * (255 - colr) / 255
    X += st["grain"][c.i % 4] * 10
    if rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 4))):
            cv2.circle(X, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(1, 4)), (20, 18, 16), -1)
    return X

# ================================================================== 61. zoom-blur ====
@effect("zoom-blur", "Punch-in zoom with radial motion blur on every beat, centred on your face.", mask=True,
        defaults=dict(beat=0.8, amount=0.22))
def fx_zoom_blur(F, O, c):
    p, lt = c.cfg, c.lt
    if lt < 0.3: return O
    k = int((lt - 0.3) / p.beat); since = lt - 0.3 - k * p.beat
    punch = math.exp(-since * 6)
    z = 1 + p.amount * punch; cx, cy = c.subj.hx, c.subj.hy
    base = cv2.warpAffine(O, cv2.getRotationMatrix2D((cx, cy), 0, z), (W, H), borderMode=cv2.BORDER_REFLECT)
    if punch < 0.08: return base
    acc = base.copy(); n = 5
    for i in range(1, n):
        acc += cv2.warpAffine(O, cv2.getRotationMatrix2D((cx, cy), 0, z * (1 + 0.035 * i * punch)), (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n

# ================================================================== 62. mirror-world ====
@effect("mirror-world", "Reality folds: a mirror seam sweeps in to make you symmetrical, then the ceiling folds down into an upside-down mirror world.", mask=True,
        defaults=dict(fold_at=0.45))
def fx_mirror(F, O, c):
    p, lt = c.cfg, c.lt
    xx, yy = grid_xy(); cx = c.subj.hx
    seam = lerp(W + 20, cx, eo(seg(lt, 0.1, 0.7)))
    mx = np.where(xx > seam, 2 * seam - xx, xx).astype(np.float32)
    t2 = c.D * p.fold_at
    fold = lerp(-40, max(60, c.subj.top - 50), eo(seg(lt, t2, 0.8))) if lt >= t2 else -40
    my = np.where(yy < fold, 2 * fold - yy, yy).astype(np.float32)
    out = cv2.remap(O, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if fold > 0:
        top = (yy < fold)[..., None]
        out = np.where(top, out * np.array([0.82, 0.86, 0.98], np.float32), out)
        out = out + (255 - out) * np.exp(-((yy[:, :1] - fold) / 5) ** 2)[..., None] * 0.6
    g = np.exp(-((xx[:1] - seam) / 6) ** 2)[..., None]
    return out + (255 - out) * g * 0.5

# ------------------------------------------------------------- render ----
def build_cfg(name, user):
    d = json.loads(json.dumps(EFFECTS[name].defaults))
    d.update(user.get("*", {}))
    d.update(user.get(name, {}))
    return types.SimpleNamespace(**d)

def render(src, out, names, user_cfg, start=0.0, duration=None, fps=30, mask_mode="auto", crf=20, stills=(), seg_len=None, label=False):
    dur_in, has_audio, _ = probe(src)
    if dur_in <= 0: sys.exit(f"Cannot read video: {src}")
    dur = min(duration or dur_in - start, dur_in - start)
    n = int(round(dur * fps))
    if n < 2: sys.exit("Clip too short")
    k = len(names)
    L = seg_len if seg_len else dur / k
    plan = [(nm, i * L, (i + 1) * L if i < k - 1 else dur) for i, nm in enumerate(names)]
    plan = [p for p in plan if p[1] < dur]
    cfgs = {nm: build_cfg(nm, user_cfg) for nm in names}
    need_mask = any(EFFECTS[nm].mask for nm in names)
    segm = Segmenter(mask_mode) if need_mask else None
    subj = Subject()
    env = audio_env(src, start, dur, fps, n)
    ctx = Ctx(src=src, start=start, fps=fps, env=env, cache={}, state={}, hist=[], subj=subj,
              ai=bool(segm and segm.ai), dur=dur, n=n)
    vf = f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", src, "-vf", vf,
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    if has_audio:
        cmd += ["-ss", f"{start}", "-t", f"{dur}", "-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-bsf:v", "filter_units=remove_types=6",                       # drop encoder-info SEI
            "-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-flags:a", "+bitexact", "-movflags", "+faststart", "-shortest", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    still_idx = {int(round(s * fps)): s for s in stills}
    fsz = W * H * 3
    for i in range(n):
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz: break
        O8 = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        t = i / fps
        cur = plan[-1]
        for p in plan:
            if p[1] <= t < p[2]: cur = p; break
        name, a, b = cur
        ctx.i, ctx.t, ctx.lt, ctx.D, ctx.seg_a, ctx.cfg = i, t, t - a, b - a, a, cfgs[name]
        ctx.e = float(env[min(i, n - 1)])
        ctx.O8 = O8
        ctx.mask = None
        if segm is not None:
            ctx.mask = segm(O8); subj.update(ctx.mask)
        O = O8.astype(np.float32)
        small = cv2.resize(O8, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        smask = cv2.resize(ctx.mask, (W // 2, H // 2)) if ctx.mask is not None else None
        ctx.hist.append((small, smask))
        if len(ctx.hist) > 48: ctx.hist.pop(0)
        R = EFFECTS[name].fn(O.copy(), O, ctx)
        kf = eio(min(seg(ctx.lt, 0, 0.22), seg(ctx.D - ctx.lt, 0, 0.22)))
        if a == 0: kf = eio(seg(ctx.D - ctx.lt, 0, 0.22)) if b < dur else 1.0
        elif b >= dur - 1e-6: kf = eio(seg(ctx.lt, 0, 0.22))
        if kf < 1: R = O + (R - O) * kf
        if label:
            key = ("label", name)
            if key not in ctx.cache:
                txt = f"{names.index(name) + int(label):02d}  {name}"; tw_ = text_w(txt, "semi", 30) + 48
                Lb = Layer(tw_, 60); Lb.rrect((0, 0, tw_, 60), 30, fill=(0, 0, 0, 170)); Lb.text((24, 30), txt, "semi", 30, (255, 255, 255, 255), anchor="lm")
                ctx.cache[key] = Lb.np()
            over(R, ctx.cache[key], W - ctx.cache[key].shape[1] - 30, 40)
        R8 = np.clip(R, 0, 255).astype(np.uint8)
        enc.stdin.write(R8.tobytes())
        if i in still_idx:
            Image.fromarray(R8).save(f"{os.path.splitext(out)[0]}_t{still_idx[i]:05.2f}.png")
        if i % max(1, int(fps * 2)) == 0:
            print(f"[effects] {name:18s} {t:6.1f}s / {dur:.1f}s", file=sys.stderr, flush=True)
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()
    if enc.returncode != 0: sys.exit("ffmpeg encode failed")
    return out

def main():
    ap = argparse.ArgumentParser(description="Motion-graphics effects for vertical talking-head clips.")
    ap.add_argument("input", nargs="?"); ap.add_argument("output", nargs="?")
    ap.add_argument("-e", "--effect", help="effect name, comma list, or 'all'")
    ap.add_argument("--list", action="store_true", help="list effects and their options")
    ap.add_argument("--config", help="JSON file: {\"effect-name\": {option: value}, \"*\": {...}}")
    ap.add_argument("--text", help="shortcut for the effect's main text (use | for line breaks / list items)")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--duration", type=float)
    ap.add_argument("--seg", type=float, help="seconds per effect when several effects are chained")
    ap.add_argument("--fps", type=float, default=30); ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--mask", default="auto", choices=["auto", "ai", "silhouette"])
    ap.add_argument("--stills", default="", help="comma list of times (s) to also save as PNG")
    ap.add_argument("--label", type=int, nargs="?", const=1, default=0, help="burn effect number+name into the corner (showcases); optional start number")
    a = ap.parse_args()
    if a.list or not a.effect:
        for k, v in EFFECTS.items():
            print(f"{k:20s} {v.desc}")
            if v.defaults: print(f"{'':20s} options: {', '.join(v.defaults)}" + (f"  (--text -> {v.main})" if v.main else ""))
        return
    names = list(EFFECTS) if a.effect == "all" else [x.strip() for x in a.effect.split(",") if x.strip()]
    bad = [x for x in names if x not in EFFECTS]
    if bad: sys.exit(f"Unknown effect(s): {', '.join(bad)}. Run --list.")
    user = json.load(open(a.config)) if a.config else {}
    if a.text:
        for nm in names:
            mk = EFFECTS[nm].main
            if mk:
                cur = EFFECTS[nm].defaults.get(mk)
                val = [s.strip() for s in a.text.split("|")] if isinstance(cur, list) else a.text.replace("|", "\n")
                user.setdefault(nm, {})[mk] = val
    stills = [float(s) for s in a.stills.split(",") if s.strip()]
    render(a.input, a.output, names, user, a.start, a.duration, a.fps, a.mask, a.crf, stills, a.seg, a.label)
    print(a.output)

if __name__ == "__main__":
    main()
