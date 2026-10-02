#!/usr/bin/env bash
# Descarga los modelos ONNX que usa faceswap.py en ./models
set -euo pipefail
cd "$(dirname "$0")/models"

FF=https://github.com/facefusion/facefusion-assets/releases/download
get() {  # get <url> <archivo>
    if [ -s "$2" ]; then echo "[=] $2 ya existe"; return; fi
    echo "[↓] $2"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

get "$FF/models-3.0.0/inswapper_128_fp16.onnx" inswapper_128_fp16.onnx   # intercambio de rostro
get "$FF/models-3.0.0/gfpgan_1.4.onnx"         gfpgan_1.4.onnx           # --enhance
get "$FF/models-3.1.0/xseg_1.onnx"             xseg_1.onnx               # --occlusion (manos delante de la cara)
get https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx \
    rvm_mobilenetv3_fp32.onnx                                                # --background (recortarte del fondo)

# Detector (SCRFD) + reconocimiento (ArcFace) del pack buffalo_l de InsightFace
if [ ! -s det_10g.onnx ] || [ ! -s w600k_r50.onnx ]; then
    get https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip buffalo_l.zip
    unzip -o -j buffalo_l.zip det_10g.onnx w600k_r50.onnx
    rm -f buffalo_l.zip
fi
echo "[+] Modelos listos en $(pwd)"
