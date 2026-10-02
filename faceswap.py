#!/usr/bin/env python3
"""
Face swap en tiempo real: webcam -> cambio de rostro (InSwapper) -> cámara virtual.

A diferencia de LivePortrait (../avatar), aquí NO se anima una foto: se sustituye solo tu
cara dentro de tu propio vídeo, así que el cuerpo, las manos y el fondo son los reales y
se mueven libremente.

Etapas por frame:
  1. Detección de rostro (SCRFD det_10g) + suavizado temporal de los 5 puntos clave.
  2. Recorte alineado 256x256 del rostro (plantilla ArcFace).
  3. InSwapper 128 genera tu cara con la identidad de la foto fuente. Con pixel boost el recorte
     de 256 se parte en 4 rejillas intercaladas de 128 y se generan las 4 (salida real a 256 px).
  4. (--occlusion) XSeg detecta lo que tapa la cara (manos, micro, pelo) y lo respeta.
  5. (--enhance) GFPGAN 1.4 restaura detalle (InSwapper trabaja a solo 128 px).
  6. Pegado con máscara difuminada, solo sobre la región del rostro.
  7. (--background) Robust Video Matting te recorta y se sustituye el fondo.
"""

import os
import sys
import time
import glob
import argparse
import threading

if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
FACES_DIR = os.path.join(APP_DIR, "caras")
VIDEOS_DIR = os.path.join(APP_DIR, "videos")
BACKGROUNDS_DIR = os.path.join(APP_DIR, "fondos")

# Plantillas de 5 puntos (ojo izq, ojo der, nariz, comisura izq, comisura der)
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
# InSwapper usa la plantilla ArcFace desplazada 8 px en x a 128 px (insightface estimate_norm)
INSWAPPER_128 = ARCFACE_112 + np.array([8.0, 0.0], dtype=np.float32)
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32) * 512.0

SWAP_CROP = 256  # el recorte de trabajo es 2x el de InSwapper: mejor máscara y pegado más nítido


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------

class OneEuroFilter:
    """Filtro 1€: quita el temblor en reposo sin añadir retraso al moverse."""
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, dt):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx_prev), dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev

    def reset(self):
        self.x_prev = None


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transformación de similitud (Umeyama) que lleva src -> dst. Devuelve matriz 2x3."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s_c ** 2).sum() * n
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def make_box_mask(size: int, blur: float = 0.3) -> np.ndarray:
    """Máscara rectangular con bordes difuminados (misma receta que FaceFusion)."""
    blur_amount = int(size * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    mask = np.ones((size, size), np.float32)
    mask[:blur_area, :] = 0
    mask[-blur_area:, :] = 0
    mask[:, :blur_area] = 0
    mask[:, -blur_area:] = 0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return mask


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, M: np.ndarray) -> None:
    """
    Pega `crop` (alineado con M) sobre `frame` in-place, usando `mask` (HxW, 0..1).
    Solo trabaja sobre el rectángulo que ocupa el rostro, no el frame completo.
    """
    h, w = frame.shape[:2]
    size = crop.shape[0]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float32) @ IM.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    IM_roi = IM.copy()
    IM_roi[:, 2] -= (x0, y0)
    rw, rh = x1 - x0, y1 - y0
    warped = cv2.warpAffine(crop, IM_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, IM_roi, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi + m * (warped.astype(np.float32) - roi)).clip(0, 255).astype(np.uint8)


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Escala `img` hasta cubrir w x h sin deformarla y recorta el centro."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * scale)), max(h, round(ih * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])


def preload_tensorrt() -> bool:
    """Carga las librerías de TensorRT 10 del paquete pip tensorrt-cu13-libs (no están en el sistema)."""
    import ctypes
    try:
        import tensorrt_libs
    except ImportError:
        print("[!] TensorRT no instalado (uv pip install 'tensorrt-cu13-libs<11'): se usará CUDA.")
        return False
    lib_dir = os.path.dirname(tensorrt_libs.__file__)
    for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return True


def pick_providers(backend: str):
    ort.preload_dlls()  # carga CUDA/cuDNN de los paquetes pip nvidia-*
    available = ort.get_available_providers()
    if backend == "cpu" or "CUDAExecutionProvider" not in available:
        if backend != "cpu":
            print("[!] CUDA no disponible en onnxruntime: se usará CPU (muy lento).")
        return ["CPUExecutionProvider"]
    if backend == "trt" and "TensorrtExecutionProvider" in available and preload_tensorrt():
        cache = os.path.join(MODELS_DIR, "trt_cache")
        os.makedirs(cache, exist_ok=True)
        if not any(f.endswith(".engine") for f in os.listdir(cache)):
            print("[*] Primera ejecución con TensorRT: compilando motores (~7 min, solo esta vez)...")
        return [("TensorrtExecutionProvider", {"trt_fp16_enable": "True", "trt_engine_cache_enable": "True",
                                               "trt_engine_cache_path": cache}),
                "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_session(name: str, providers, trt_ok: bool = True):
    """
    trt_ok=False: no usar TensorRT para este modelo. InSwapper y GFPGAN dan caras rotas con
    TensorRT FP16 y en FP32 son más lentos que con CUDA; XSeg en FP16 devuelve una máscara casi
    invertida (la cara queda "tapada" y no se cambia). Solo el detector y ArcFace usan TRT.
    """
    path = os.path.join(MODELS_DIR, name)
    if not trt_ok and isinstance(providers[0], tuple):
        providers = providers[1:]
    if not os.path.exists(path):
        sys.exit(f"[!] Falta el modelo {path}. Ejecuta ./download_models.sh")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # oculta avisos de nodos que caen a CPU
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# --------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------

class SCRFD:
    """Detector de rostros SCRFD (det_10g de InsightFace): cajas + 5 puntos clave."""
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, session, det_size=640, thresh=0.5, nms=0.4):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.size, self.thresh, self.nms = det_size, thresh, nms
        self.centers = {}
        for stride in self.STRIDES:
            n = det_size // stride
            yy, xx = np.mgrid[:n, :n]
            c = (np.stack([xx, yy], axis=-1).reshape(-1, 2) * stride).astype(np.float32)
            self.centers[stride] = np.repeat(c, self.NUM_ANCHORS, axis=0)

    def detect(self, img):
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.STRIDES):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= self.thresh)[0]
            if keep.size == 0:
                continue
            c = self.centers[stride][keep]
            d = outs[i + 3].reshape(-1, 4)[keep] * stride
            k = outs[i + 6].reshape(-1, 10)[keep] * stride
            boxes.append(np.hstack([c - d[:, :2], c + d[:, 2:]]))
            kpss.append((k.reshape(-1, 5, 2) + c[:, None, :]))
            scores.append(sc[keep])
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.vstack(boxes) / scale
        kpss = np.vstack(kpss) / scale
        scores = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes([[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in boxes],
                               scores.tolist(), self.thresh, self.nms)
        idx = np.array(idx).reshape(-1)
        return np.hstack([boxes[idx], scores[idx, None]]).astype(np.float32), kpss[idx].astype(np.float32)


class FaceSwapper:
    def __init__(self, providers, det_size=640, enhance=False, occlusion=False, enhance_blend=0.8,
                 pixel_boost=False):
        self.detector = SCRFD(load_session("det_10g.onnx", providers), det_size=det_size)

        self.arcface = load_session("w600k_r50.onnx", providers)
        self.swapper = load_session("inswapper_128_fp16.onnx", providers, trt_ok=False)
        # Matriz que proyecta el embedding ArcFace al espacio latente de InSwapper
        self.emap = numpy_helper.to_array(onnx.load(os.path.join(MODELS_DIR, "inswapper_128_fp16.onnx"))
                                          .graph.initializer[-1]).astype(np.float32)
        self.enhance = enhance
        self.enhance_blend = float(enhance_blend)
        self.occlusion = occlusion
        self.pixel_boost = pixel_boost
        self.gfpgan = load_session("gfpgan_1.4.onnx", providers, trt_ok=False) if enhance else None
        self.xseg = load_session("xseg_1.onnx", providers, trt_ok=False) if occlusion else None

        self.box_mask = make_box_mask(SWAP_CROP)
        self.enhance_mask = make_box_mask(512)
        self.latent = None
        self.kps_filter = OneEuroFilter(min_cutoff=1.2, beta=0.08)
        self.last_center = None
        self.t_prev = None

    # --- identidad fuente ----------------------------------------------------------
    def detect(self, img):
        return self.detector.detect(img)

    def embedding(self, img, kps):
        M = similarity_transform(kps, ARCFACE_112)
        crop = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.arcface.run(None, {self.arcface.get_inputs()[0].name: blob})[0][0]
        return emb / np.linalg.norm(emb)

    def set_source(self, paths):
        embs = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[!] No se pudo leer {p}")
                continue
            bboxes, kpss = self.detect(img)
            if len(bboxes) == 0:
                print(f"[!] Sin rostro en {p}")
                continue
            i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
            embs.append(self.embedding(img, kpss[i]))
        if not embs:
            sys.exit("[!] Ninguna imagen fuente tiene un rostro detectable.")
        emb = np.mean(embs, axis=0)
        emb /= np.linalg.norm(emb)
        latent = emb[None] @ self.emap
        self.latent = (latent / np.linalg.norm(latent)).astype(np.float32)
        print(f"[+] Identidad fuente calculada a partir de {len(embs)} imagen(es).")

    # --- por frame ---------------------------------------------------------------
    def _swap_face(self, frame, kps):
        M = similarity_transform(kps, INSWAPPER_128 * (SWAP_CROP / 128.0))
        crop = cv2.warpAffine(frame, M, (SWAP_CROP, SWAP_CROP), borderMode=cv2.BORDER_REPLICATE)
        if self.pixel_boost:
            # Pixel boost (FaceFusion): 4 sub-imágenes de 128 tomando 1 de cada 2 píxeles en x/y
            n = SWAP_CROP // 128
            tiles = crop.reshape(128, n, 128, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, 128, 128, 3)
            outs = [self._run_swapper(t) for t in tiles]
            fake = np.stack(outs).reshape(n, n, 128, 128, 3).transpose(2, 0, 3, 1, 4).reshape(SWAP_CROP, SWAP_CROP, 3)
        else:
            small = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            fake = cv2.resize(self._run_swapper(small), (SWAP_CROP, SWAP_CROP), interpolation=cv2.INTER_CUBIC)

        mask = self.box_mask
        if self.occlusion:
            mask = np.minimum(mask, self._occlusion_mask(crop))
        paste_back(frame, fake, mask, M)

    def _run_swapper(self, img128):
        blob = cv2.dnn.blobFromImage(img128, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        out = self.swapper.run(None, {"target": blob, "source": self.latent})[0][0]
        return (out.transpose(1, 2, 0)[..., ::-1].clip(0, 1) * 255).astype(np.uint8)

    def _occlusion_mask(self, crop):
        """XSeg: 1 = piel del rostro visible, 0 = algo delante (mano, micro, pelo)."""
        x = cv2.resize(crop, (256, 256)).astype(np.float32)[None] / 255.0
        m = self.xseg.run(None, {"input": x})[0][0, :, :, 0].clip(0, 1)
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]))
        return (cv2.GaussianBlur(m, (0, 0), 5).clip(0.5, 1) - 0.5) * 2

    def _enhance_face(self, frame, kps):
        M = similarity_transform(kps, FFHQ_512)
        crop = cv2.warpAffine(frame, M, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        x = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (512, 512), (127.5, 127.5, 127.5), swapRB=True)
        out = self.gfpgan.run(None, {"input": x})[0][0]
        restored = ((out.transpose(1, 2, 0)[..., ::-1].clip(-1, 1) + 1) * 127.5).astype(np.uint8)
        paste_back(frame, restored, self.enhance_mask * self.enhance_blend, M)

    def process(self, frame, many_faces=False):
        """Cambia el/los rostro(s) in-place. Devuelve el número de rostros cambiados."""
        now = time.perf_counter()
        dt = 1.0 / 30.0 if self.t_prev is None else min(max(now - self.t_prev, 1 / 120), 1 / 10)
        self.t_prev = now

        bboxes, kpss = self.detect(frame)
        if len(bboxes) == 0:
            self.kps_filter.reset()
            self.last_center = None
            return 0

        if many_faces:
            targets = list(kpss)
        else:
            i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
            kps = kpss[i].astype(np.float32)
            # Si el rostro "salta" (otra persona, re-detección lejos) no se suaviza desde el anterior
            center = kps.mean(0)
            face_w = bboxes[i, 2] - bboxes[i, 0]
            if self.last_center is not None and np.linalg.norm(center - self.last_center) > 0.5 * face_w:
                self.kps_filter.reset()
            self.last_center = center
            targets = [self.kps_filter.update(kps, dt)]

        for kps in targets:
            self._swap_face(frame, kps)
            if self.enhance:
                self._enhance_face(frame, kps)
        return len(targets)


class BackgroundReplacer:
    """
    Robust Video Matting: separa a la persona del fondo con un alfa suave (pelo incluido) y
    memoria entre frames, así el borde no parpadea. El fondo nuevo puede ser una imagen, un
    vídeo en bucle o tu propio fondo difuminado ("blur").
    """
    def __init__(self, providers, spec):
        # En CUDA: el modelo es recurrente y de tamaño dinámico, TensorRT no aporta y es arriesgado
        self.session = load_session("rvm_mobilenetv3_fp32.onnx", providers, trt_ok=False)
        self.spec = spec
        self.cap, self.image, self.cached = None, None, None
        if spec != "blur":
            path = find_in(spec, BACKGROUNDS_DIR)
            self.image = cv2.imread(path)
            if self.image is None:
                self.cap = cv2.VideoCapture(path)
                ok, self.image = self.cap.read()
                if not ok:
                    sys.exit(f"[!] No se pudo leer el fondo {path} (ni como imagen ni como vídeo).")
        self.reset()

    def reset(self):
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4

    def matte(self, frame):
        """Alfa de la persona (HxW, 0..1). Llamar con el frame original, antes del cambio de cara."""
        h, w = frame.shape[:2]
        src = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, swapRB=True)
        ratio = np.array([min(1.0, 512.0 / max(h, w))], np.float32)  # RVM rinde mejor a ~512 px internos
        out = self.session.run(None, {"src": src, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2],
                                      "r4i": self.rec[3], "downsample_ratio": ratio})
        self.rec = out[2:]
        return np.ascontiguousarray(out[1][0, 0])

    def _background(self, frame):
        h, w = frame.shape[:2]
        if self.spec == "blur":
            small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
            return cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (w, h), interpolation=cv2.INTER_LINEAR)
        if self.cap is not None:  # vídeo: un frame del fondo por frame de salida, en bucle
            ok, img = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self.cap.read()
            if ok:
                self.image = img
        if self.cached is None or self.cached[0] is not self.image or self.cached[1].shape[:2] != (h, w):
            self.cached = (self.image, cover_resize(self.image, w, h))
        return self.cached[1]

    def apply(self, frame, pha):
        """Compone `frame` (ya con la cara cambiada) sobre el fondo nuevo, in-place."""
        frame[:] = cv2.blendLinear(frame, self._background(frame), pha, 1.0 - pha)


# --------------------------------------------------------------------------------------
# Entrada / salida
# --------------------------------------------------------------------------------------

class LatestFrameSource:
    """Lee la fuente en un hilo aparte y expone siempre el frame más reciente."""
    def __init__(self, cap, pace_fps=None):
        self.cap = cap
        self.period = 1.0 / pace_fps if pace_fps else None
        self.frame, self.seq, self.consumed = None, 0, 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        t_next = time.perf_counter()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.period:  # vídeo: reproducir en bucle
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()
            if self.period:
                t_next += self.period
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()

    def read(self, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != self.consumed or not self.running, timeout)
            if self.seq == self.consumed:
                return None
            self.consumed = self.seq
            return self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class VirtualCamSender:
    """Envía a la cámara virtual a ritmo constante desde un hilo propio."""
    def __init__(self, cam):
        self.cam, self.frame = cam, None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def _loop(self):
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is None:
                time.sleep(0.005)
                continue
            self.cam.send(frame)
            self.cam.sleep_until_next_frame()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cam.close()


def open_webcam(webcam_id, width, height, fps):
    dev = f"/dev/video{webcam_id}" if str(webcam_id).isdigit() else str(webcam_id)
    for attempt in range(4):
        for src in (dev, int(webcam_id) if str(webcam_id).isdigit() else None):
            if src is None:
                continue
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        time.sleep(0.4)
    sys.exit(f"[!] No se pudo abrir la webcam {dev} (¿la está usando otra app, p. ej. `avatar`?).")


def find_default_virtual_cam():
    for node in ["/dev/video2", "/dev/video10", "/dev/video3", "/dev/video4"]:
        if os.path.exists(node):
            return node
    return "/dev/video2"


def find_in(path, folder):
    """La ruta tal cual si existe; si no, relativa a la carpeta del proyecto (caras/ o videos/)."""
    if os.path.exists(path):
        return path
    alt = os.path.join(folder, path)
    return alt if os.path.exists(alt) else path


def resolve_sources(items):
    paths = []
    for item in items:
        item = find_in(item, FACES_DIR)
        if os.path.isdir(item):
            for ext in ("png", "jpg", "jpeg", "webp"):
                paths += sorted(glob.glob(os.path.join(item, f"*.{ext}")))
        else:
            paths.append(item)
    return paths


def parse_args():
    ap = argparse.ArgumentParser(description="Face swap en tiempo real hacia cámara virtual.")
    ap.add_argument("--source", "-s", nargs="+", required=True,
                    help="Foto(s) o carpeta con la cara a usar. Se busca también dentro de caras/ (p. ej. -s miranda). Con varias se promedia la identidad.")
    ap.add_argument("--webcam-id", "-w", default="0", help="Índice o ruta de la webcam (default: 0).")
    ap.add_argument("--input-video", type=str, default=None,
                    help="Usar un vídeo en bucle en lugar de la webcam (pruebas). Se busca también dentro de videos/.")
    ap.add_argument("--output-device", "-o", default=find_default_virtual_cam(),
                    help="Dispositivo de cámara virtual v4l2loopback.")
    ap.add_argument("--width", type=int, default=1280, help="Ancho de captura/salida (default: 1280).")
    ap.add_argument("--height", type=int, default=720, help="Alto de captura/salida (default: 720).")
    ap.add_argument("--fps", type=int, default=30, help="FPS de la cámara virtual (default: 30).")
    ap.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=False,
                    help="Restaurar el rostro con GFPGAN (más nítido, más coste de GPU).")
    ap.add_argument("--enhance-blend", type=float, default=0.8,
                    help="Mezcla de GFPGAN con el resultado original, 0-1 (default: 0.8).")
    ap.add_argument("--occlusion", action=argparse.BooleanOptionalAction, default=True,
                    help="Respetar lo que tapa la cara, como tus manos (XSeg). Default: activado.")
    ap.add_argument("--background", "-b", default=None,
                    help="Sustituir el fondo: imagen, vídeo (en bucle) o 'blur' para difuminar el tuyo. "
                         "Se busca también dentro de fondos/.")
    ap.add_argument("--pixel-boost", action=argparse.BooleanOptionalAction, default=False,
                    help="Generar la cara a 256 px en vez de 128 (4 pasadas de InSwapper: más nítido, ~2x más lento).")
    ap.add_argument("--many-faces", action="store_true", help="Cambiar todos los rostros, no solo el más grande.")
    ap.add_argument("--det-size", type=int, default=640, choices=[320, 480, 640],
                    help="Resolución del detector (default: 640; 320 es más rápido si estás cerca de la cámara).")
    try:
        import tensorrt_libs  # noqa: F401
        default_backend = "trt"
    except ImportError:
        default_backend = "cuda"
    ap.add_argument("--backend", choices=["cuda", "trt", "cpu"], default=default_backend,
                    help=f"trt: detector y ArcFace con TensorRT (compila motores la 1.ª vez, ~7 min); "
                         f"cuda: todo con CUDA. Default: {default_backend}.")
    ap.add_argument("--no-preview", action="store_true", help="Sin ventana de previsualización.")
    ap.add_argument("--no-virtualcam", action="store_true", help="Sin salida a cámara virtual.")
    ap.add_argument("--mirror", action="store_true", help="Espejar la imagen (como un espejo).")
    ap.add_argument("--dry-run", type=float, default=0.0,
                    help="Medir rendimiento durante N segundos y salir.")
    return ap.parse_args()


def main():
    args = parse_args()
    sources = resolve_sources(args.source)
    if not sources:
        sys.exit("[!] No se encontraron imágenes fuente.")

    providers = pick_providers(args.backend)
    print("=" * 60)
    print(" FACE SWAP EN TIEMPO REAL")
    print(f" Fuente:      {len(sources)} imagen(es) ({sources[0]}{' ...' if len(sources) > 1 else ''})")
    print(f" Entrada:     {args.input_video or f'webcam {args.webcam_id} ({args.width}x{args.height})'}")
    print(f" Oclusión:    {args.occlusion}   Mejora GFPGAN: {args.enhance}   Pixel boost: {args.pixel_boost}   "
          f"Varios rostros: {args.many_faces}")
    print(f" Fondo:       {args.background or 'el real'}")
    print(f" Backend:     {providers[0] if isinstance(providers[0], str) else providers[0][0]}")
    print("=" * 60)

    swapper = FaceSwapper(providers, det_size=args.det_size, enhance=args.enhance,
                          occlusion=args.occlusion, enhance_blend=args.enhance_blend,
                          pixel_boost=args.pixel_boost)
    swapper.set_source(sources)
    bg = BackgroundReplacer(providers, args.background) if args.background else None

    if args.input_video:
        args.input_video = find_in(args.input_video, VIDEOS_DIR)
        cap = cv2.VideoCapture(args.input_video)
        if not cap.isOpened():
            sys.exit(f"[!] No se pudo abrir {args.input_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src = LatestFrameSource(cap, pace_fps=fps)
    else:
        src = LatestFrameSource(open_webcam(args.webcam_id, args.width, args.height, args.fps))

    first = None
    for _ in range(50):
        first = src.read()
        if first is not None:
            break
    if first is None:
        sys.exit("[!] La fuente de vídeo no entrega frames.")
    out_h, out_w = first.shape[:2]

    vcam = None
    if not args.no_virtualcam:
        try:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=args.fps, device=args.output_device,
                                      fmt=pyvirtualcam.PixelFormat.BGR)
            vcam = VirtualCamSender(cam)
            print(f"[+] Cámara virtual: {cam.device} ({out_w}x{out_h} @ {args.fps} FPS)")
        except Exception as e:
            print(f"[!] No se pudo abrir la cámara virtual {args.output_device}: {e}")
            print("    Carga el módulo: sudo modprobe v4l2loopback exclusive_caps=1 card_label=\"AI Virtual Cam\"")

    preview = not args.no_preview and args.dry_run <= 0
    win = "Face Swap (q: salir | espacio: on/off | e: GFPGAN | p: boost | o: oclusión | b: fondo | h: HUD)"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, out_w, out_h)
    print("[+] En marcha. " + ("Teclas: q salir, espacio on/off, e GFPGAN, p pixel boost, o oclusión, b fondo, h HUD." if preview else ""))

    swap_on, hud, bg_on = True, True, bg is not None
    t_start = time.perf_counter()
    lat, n_faces, frames = [], 0, 0
    try:
        frame = first
        while True:
            if frame is None:
                frame = src.read()
                if frame is None:
                    continue
            t0 = time.perf_counter()
            if args.mirror:
                frame = cv2.flip(frame, 1)
            pha = bg.matte(frame) if bg_on else None
            if swap_on:
                n_faces = swapper.process(frame, many_faces=args.many_faces)
            if pha is not None:
                bg.apply(frame, pha)
            lat.append((time.perf_counter() - t0) * 1000)
            frames += 1

            if vcam is not None:
                vcam.submit(frame)

            if preview:
                show = frame
                if hud:
                    show = frame.copy()
                    recent = lat[-30:]
                    fps_est = 1000.0 / max(np.mean(recent), 1e-3)
                    onoff = lambda v: "ON" if v else "OFF"
                    txt = (f"{np.mean(recent):.1f} ms (~{fps_est:.0f} FPS max) | rostros: {n_faces} | "
                           f"swap {onoff(swap_on)} | GFPGAN {onoff(swapper.enhance)} | "
                           f"boost {onoff(swapper.pixel_boost)} | oclusion {onoff(swapper.occlusion)} | fondo {onoff(bg_on)}")
                    # Franja oscura + un solo trazo: un contorno grueso tiene otro ancho de letra y no casa
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    band = show[:th + 20, :tw + 24]
                    band[:] = (band * 0.35).astype(np.uint8)
                    cv2.putText(show, txt, (12, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
                cv2.imshow(win, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    swap_on = not swap_on
                elif key == ord("p"):
                    swapper.pixel_boost = not swapper.pixel_boost
                elif key == ord("h"):
                    hud = not hud
                elif key == ord("e"):
                    if swapper.gfpgan is None:
                        print("[*] Cargando GFPGAN...")
                        swapper.gfpgan = load_session("gfpgan_1.4.onnx", providers, trt_ok=False)
                    swapper.enhance = not swapper.enhance
                elif key == ord("o"):
                    if swapper.xseg is None:
                        print("[*] Cargando XSeg...")
                        swapper.xseg = load_session("xseg_1.onnx", providers, trt_ok=False)
                    swapper.occlusion = not swapper.occlusion
                elif key == ord("b"):
                    if bg is None:
                        print("[*] Cargando Robust Video Matting (fondo difuminado; usa -b para otro fondo)...")
                        bg = BackgroundReplacer(providers, "blur")
                    bg_on = not bg_on
                    bg.reset()  # la memoria de RVM no vale tras una pausa
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.dry_run > 0 and time.perf_counter() - t_start >= args.dry_run:
                break
            frame = None
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        if vcam is not None:
            vcam.close()
        if preview:
            cv2.destroyAllWindows()

    if lat:
        warm = lat[10:] or lat
        elapsed = time.perf_counter() - t_start
        print(f"[=] {frames} frames en {elapsed:.1f} s -> {frames / elapsed:.1f} FPS de salida | "
              f"proceso medio {np.mean(warm):.1f} ms, p95 {np.percentile(warm, 95):.1f} ms")


if __name__ == "__main__":
    main()
