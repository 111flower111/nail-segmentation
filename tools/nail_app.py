#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nail_app.py —— 指甲病害智能检测 · 手机端 Web 应用（单文件，无外部模板依赖）
============================================================================

一个跑在 AidLux 里的 Flask 应用：手机浏览器打开 → 点按钮调起相机拍照/选图
→ 后端跑 ONNX 分割 → 页面显示原图、分割结果、各类占比与结论。

为什么用 Web 而不是 AidLux 的 cvs GUI：
    1) HTML 的 <input capture> 能直接调起安卓相机, 绕开 AidLux 摄像头 API;
    2) 手机/电脑浏览器都能访问, 方便截图写简历;
    3) 单文件, 传输和部署都省事。

依赖:  pip install flask onnxruntime==1.19.2

用法:
    python nail_app.py
    # 然后在手机浏览器打开 http://127.0.0.1:5000  (或 http://<手机IP>:5000)
"""
from __future__ import annotations

import base64
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request

# ------------------------------------------------------------------ 配置
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'nail_3class_512.onnx')
SIZE = 512
NAMES = ['background', 'healthy', 'diseased']
NAMES_CN = {'background': '背景', 'healthy': '健康指甲', 'diseased': '病甲'}
PALETTE_RGB = np.array([[0, 0, 0], [46, 204, 113], [231, 76, 60]], np.uint8)
# 低于这个占比就认为"没找到指甲"
MIN_NAIL = 0.05

app = Flask(__name__)
_lock = threading.Lock()          # onnxruntime session 非线程安全, 串行化
_sess = None


def get_session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(MODEL, sess_options=so,
                                     providers=['CPUExecutionProvider'])
    return _sess


def infer(bgr: np.ndarray):
    """BGR 图 -> (类别图 uint8[原尺寸], logits, 耗时ms)"""
    h0, w0 = bgr.shape[:2]
    x = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = x[:, :, ::-1].astype(np.float32)          # BGR -> RGB (不可省)
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    sess = get_session()
    with _lock:
        t0 = time.time()
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
        ms = (time.time() - t0) * 1000
    logits = outs[0]
    pred = outs[1][0] if len(outs) > 1 else logits.argmax(1)[0].astype(np.uint8)
    if (h0, w0) != (SIZE, SIZE):
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return pred, logits, ms


def b64_jpg(bgr: np.ndarray, quality: int = 88, max_side: int = 900) -> str:
    """BGR -> data URI（顺便限一下尺寸，页面加载快）"""
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>指甲病害智能检测</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
         background:#0f172a; color:#e2e8f0; padding:16px; }
  .wrap { max-width:560px; margin:0 auto; }
  h1 { font-size:19px; margin:4px 0 2px; letter-spacing:.5px; }
  .sub { font-size:12px; color:#94a3b8; margin-bottom:16px; }
  .card { background:#1e293b; border-radius:14px; padding:16px; margin-bottom:14px; }
  .btn { display:block; width:100%; padding:16px; border:0; border-radius:12px;
         font-size:16px; font-weight:600; color:#04240f; cursor:pointer;
         background:linear-gradient(135deg,#4ade80,#22c55e); }
  .btn:active { transform:scale(.985); }
  input[type=file] { display:none; }
  .tip { font-size:12px; color:#94a3b8; margin-top:10px; line-height:1.7; }
  .tip b { color:#fbbf24; }
  .imgs { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .imgs figure { margin:0; }
  .imgs img { width:100%; border-radius:10px; display:block; background:#000; }
  .imgs figcaption { font-size:12px; color:#94a3b8; text-align:center; margin-top:6px; }
  .verdict { font-size:22px; font-weight:700; text-align:center; padding:14px;
             border-radius:12px; margin-bottom:14px; }
  .ok   { background:rgba(34,197,94,.15);  color:#4ade80; }
  .bad  { background:rgba(239,68,68,.15);  color:#f87171; }
  .warn { background:rgba(251,191,36,.15); color:#fbbf24; }
  .bar { margin:10px 0; }
  .bar .lab { display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px; }
  .bar .track { height:9px; background:#0f172a; border-radius:5px; overflow:hidden; }
  .bar .fill { height:100%; border-radius:5px; transition:width .4s; }
  .meta { font-size:12px; color:#94a3b8; text-align:center; }
  #load { text-align:center; padding:26px; color:#94a3b8; font-size:14px; display:none; }
  .spin { width:26px; height:26px; margin:0 auto 10px; border:3px solid #334155;
          border-top-color:#22c55e; border-radius:50%; animation:r .8s linear infinite; }
  @keyframes r { to { transform:rotate(360deg); } }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>指甲病害智能检测</h1>
  <div class="sub">SegFormer-B0 · MMSegmentation · ONNX Runtime · 端侧实时分割</div>

  <div class="card">
    <input type="file" id="f" accept="image/*" capture="environment">
    <button class="btn" onclick="document.getElementById('f').click()">📷 拍照 / 选择图片</button>
    <div class="tip">
      拍摄建议：<b>靠近拍摄，让指甲占画面 1/2 以上</b>，背景尽量简单、光线均匀。
    </div>
  </div>

  <div id="load"><div class="spin"></div>正在分析…</div>
  <div id="out" class="hidden"></div>
</div>

<script>
const NAMES = {0:'背景', 1:'健康指甲', 2:'病甲'};
const COLORS = {0:'#64748b', 1:'#22c55e', 2:'#ef4444'};

document.getElementById('f').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('load').style.display = 'block';
  document.getElementById('out').classList.add('hidden');

  const fd = new FormData();
  fd.append('image', file);
  try {
    const r = await fetch('/predict', { method:'POST', body: fd });
    const d = await r.json();
    if (d.error) { alert('出错：' + d.error); return; }
    render(d);
  } catch (err) {
    alert('请求失败：' + err);
  } finally {
    document.getElementById('load').style.display = 'none';
  }
};

function render(d) {
  const nail = d.stats.healthy + d.stats.diseased;
  let cls, txt;
  if (nail < 5)        { cls = 'warn'; txt = '未检测到指甲，请靠近重拍'; }
  else if (d.stats.diseased >= 5) { cls = 'bad';  txt = '检测到病甲区域'; }
  else                 { cls = 'ok';   txt = '指甲健康'; }

  let bars = '';
  for (const i of [1, 2, 0]) {
    const p = d.stats[['background','healthy','diseased'][i]];
    bars += `<div class="bar">
      <div class="lab"><span>${NAMES[i]}</span><span>${p.toFixed(1)}%</span></div>
      <div class="track"><div class="fill" style="width:${p}%;background:${COLORS[i]}"></div></div>
    </div>`;
  }

  document.getElementById('out').innerHTML = `
    <div class="card">
      <div class="imgs">
        <figure><img src="${d.original}"><figcaption>原图</figcaption></figure>
        <figure><img src="${d.overlay}"><figcaption>检测结果</figcaption></figure>
      </div>
    </div>
    <div class="card">
      <div class="verdict ${cls}">${txt}</div>
      ${bars}
      <div class="meta">推理耗时 ${d.ms.toFixed(0)} ms · 输入 512×512 · 3 类分割</div>
    </div>`;
  document.getElementById('out').classList.remove('hidden');
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


@app.route('/health')
def health():
    return jsonify(ok=True, model=os.path.basename(MODEL), exists=os.path.isfile(MODEL))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error='没有收到图片'), 400
    raw = request.files['image'].read()
    if not raw:
        return jsonify(error='图片为空'), 400
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error='图片解码失败'), 400

    try:
        pred, _logits, ms = infer(bgr)
    except Exception as e:                       # noqa: BLE001
        return jsonify(error=f'推理失败: {e}'), 500

    color = PALETTE_RGB[pred][:, :, ::-1]        # RGB -> BGR
    overlay = cv2.addWeighted(bgr, 0.55, color, 0.45, 0)
    total = pred.size
    stats = {n: float((pred == i).sum()) / total * 100 for i, n in enumerate(NAMES)}

    return jsonify(ms=ms, stats=stats,
                   original=b64_jpg(bgr), overlay=b64_jpg(overlay))


if __name__ == '__main__':
    if not os.path.isfile(MODEL):
        raise SystemExit(f'[x] 找不到模型: {MODEL}\n    把 nail_3class_512.onnx 放到本文件同目录')
    print(f'[ok] 模型: {MODEL}')
    print('[ok] 服务启动中… 手机浏览器打开  http://127.0.0.1:5000')
    print('     若打不开, 试 http://<手机IP>:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)
