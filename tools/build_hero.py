"""Gera os mapas do hero (diffuse/alpha/depth/normal + overlay) a partir da foto e da arte.

Uso: python tools/build_hero.py <foto> <arte> <saida-dir>
Precisa de tools/vision.swift (Apple Vision) para rosto e recorte.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 2048, 2336
SMALL = (512, 584)
# Olhos do template de referência (person-diffuse original), em px do canvas.
REF_LEFT, REF_RIGHT = np.array([890.5, 910.7]), np.array([1141.3, 906.8])
REF_MID = (REF_LEFT + REF_RIGHT) / 2
# A foto é cortada no peito; a escala garante que ela chegue até a base do canvas.
EXTRA_SCALE = 1.22

TOOLS = Path(__file__).parent


def vision(img, mask_out):
    out = subprocess.run(["swift", str(TOOLS / "vision.swift"), str(img), str(mask_out)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def similarity(src_l, src_r, dst_l, dst_r):
    """Matriz 2x3 que leva os dois olhos de origem nos dois olhos de destino."""
    s = complex(*(dst_r - dst_l)) / complex(*(src_r - src_l))
    a, b = s.real, s.imag
    m = np.array([[a, -b, 0.0], [b, a, 0.0]])
    m[:, 2] = dst_l - m[:, :2] @ src_l
    return m


def warp(img, m, border):
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def edge_fade(alpha, valid):
    """Suaviza as bordas onde a imagem de origem acaba (laterais e base cortadas)."""
    dist = cv2.distanceTransform((valid > 127).astype(np.uint8), cv2.DIST_L2, 5)
    return alpha * np.clip(dist / 90.0, 0, 1)


def main(photo, art, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)

    # --- foto base ---
    pv = vision(photo, work / "photo-seg.png")
    p_img = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    p_seg = cv2.imread(str(work / "photo-seg.png"), cv2.IMREAD_GRAYSCALE)
    p_seg = cv2.resize(p_seg, (p_img.shape[1], p_img.shape[0]))
    half = (REF_RIGHT - REF_LEFT) / 2 * EXTRA_SCALE
    dst_l, dst_r = REF_MID - half, REF_MID + half
    mp = similarity(np.array(pv["leftEye"]), np.array(pv["rightEye"]), dst_l, dst_r)

    diffuse = warp(p_img, mp, (255, 255, 255))
    valid = warp(np.full(p_seg.shape, 255, np.uint8), mp, 0)
    alpha = warp(p_seg, mp, 0).astype(np.float32) / 255
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    alpha = edge_fade(alpha, valid)

    # Fundo branco liso fora da silhueta, como no template.
    a3 = alpha[..., None]
    diffuse = (diffuse * a3 + 255 * (1 - a3)).astype(np.uint8)

    # --- profundidade: corpo achatado + cúpula na cabeça ---
    face = pv["faceBox"]
    fx, fy, fw, fh = face
    corners = np.array([[fx, fy], [fx + fw, fy + fh]], np.float64)
    corners = corners @ mp[:, :2].T + mp[:, 2]
    (x0, y0), (x1, y1) = corners
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.12
    rx, ry = (x1 - x0) * 0.5, (y1 - y0) * 0.72
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dome = np.clip(1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0, 1) ** 0.5
    body = 0.6 * (alpha > 0.5)
    depth = np.maximum(body, (0.6 + 0.4 * dome) * (dome > 0)) * alpha
    depth = cv2.GaussianBlur(depth, (0, 0), 18)
    depth_small = cv2.resize(depth.astype(np.float32), SMALL, interpolation=cv2.INTER_AREA)

    # --- normal derivada da profundidade (RGB: plano = 128,128,255) ---
    gx = cv2.Sobel(depth_small, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(depth_small, cv2.CV_32F, 0, 1, ksize=5)
    k = 6.0
    n = np.dstack([-gx * k, gy * k, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    normal = ((n * 0.5 + 0.5) * 255).astype(np.uint8)

    # --- arte do reveal, alinhada pelos olhos ---
    av = vision(art, work / "art-seg.png")
    a_img = cv2.imread(str(art), cv2.IMREAD_COLOR)
    a_seg = cv2.imread(str(work / "art-seg.png"), cv2.IMREAD_GRAYSCALE)
    a_seg = cv2.resize(a_seg, (a_img.shape[1], a_img.shape[0]))
    # Pinceladas e orelhas da máscara ficam fora do recorte de pessoa: soma a tinta que
    # encosta na silhueta (o fundo da arte é off-white, ~240, então o limiar fica abaixo).
    ink = np.clip((228 - a_img.min(axis=2).astype(np.float32)) / 25, 0, 1)
    person = (a_seg > 127).astype(np.uint8)
    n_lbl, lbl = cv2.connectedComponents(((ink > 0.3) | (person > 0)).astype(np.uint8))
    keep = np.isin(lbl, np.unique(lbl[person > 0]))
    ink = ink * keep
    a_alpha = np.maximum(a_seg.astype(np.float32) / 255, ink)
    ma = similarity(np.array(av["leftEye"]), np.array(av["rightEye"]), dst_l, dst_r)
    overlay = warp(a_img, ma, (255, 255, 255))
    o_valid = warp(np.full(a_seg.shape, 255, np.uint8), ma, 0)
    o_alpha = warp((a_alpha * 255).astype(np.uint8), ma, 0).astype(np.float32) / 255
    o_alpha = cv2.GaussianBlur(o_alpha, (0, 0), 1.5)
    o_alpha = edge_fade(o_alpha, o_valid)
    overlay_rgba = np.dstack([overlay, (o_alpha * 255).astype(np.uint8)])

    def save(name, img, q=92):
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_WEBP_QUALITY, q])

    save("person-diffuse.webp", diffuse)
    save("person-alpha.webp", (alpha * 255).astype(np.uint8))
    save("person-depth.webp", (np.clip(depth_small, 0, 1) * 255).astype(np.uint8))
    save("person-normal.webp", cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    save("overlay.webp", overlay_rgba)

    # Prévia lado a lado: foto | arte | meio a meio, para conferir o encaixe.
    comp = diffuse.copy()
    comp[:, W // 2:] = (overlay[:, W // 2:] * o_alpha[:, W // 2:, None]
                        + diffuse[:, W // 2:] * (1 - o_alpha[:, W // 2:, None])).astype(np.uint8)
    for p in (dst_l, dst_r):
        cv2.circle(comp, tuple(int(v) for v in p), 10, (0, 0, 255), -1)
    preview = np.hstack([diffuse, overlay, comp])
    cv2.imwrite(str(out.parent / "preview-encaixe.jpg"),
                cv2.resize(preview, (preview.shape[1] // 3, preview.shape[0] // 3)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
