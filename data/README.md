# 資料

影像序列放在 `data/frames/`，檔名為 `00.jpg … 31.jpg`。

**不需要手動準備**：執行 `main.py`（或 `python -m src.data_download`）時，若影像不存在，
程式會自動下載 [LearnOpenGL skybox](https://learnopengl.com/Advanced-OpenGL/Cubemaps) 到 `data/skybox/`，
並在 CPU 上渲染出整個序列。

渲染設定：虛擬相機（800 × 600 px，垂直視角 45°）繞垂直軸旋轉 360°，共 32 步、每步 11.25°，
加上暗角（vignette）與小幅隨機旋轉（σ = 2°），隨機種子固定為 327，因此每次產生的影像都相同。
焦距 f = 300 / tan(22.5°) ≈ 724 px。

