# 影像拼接：SIFT 與 RANSAC

這個專案實作 SIFT 特徵點提取與匹配，並使用 RANSAC 演算法估計幾何變換矩陣，完成多張影像的全景拼接：

1. **SIFT** 特徵點與描述子
2. **候選匹配**：使用 Lowe 的最近鄰／次近鄰**比值檢定**
3. **轉換對應**：translation、affine（最小平方法）、homography matrix（正規化 DLT + SVD）
4. **RANSAC**：以內點數量為主、誤差為次的評比方式，最後用所有內點做一次最小平方法重新擬合
5. **影像變形**到共同畫布上，實作 **average blend** 與 **feather blend** 兩種融合方式
6. **多張影像拼接**：相鄰影像兩兩估計 homography，串接成對應到中間那張的轉換後一次融合
7. **圓柱全景**：圓柱投影 + 相鄰影像對齊 + 串接轉換矩陣 + 一次融合所有影像

未來規劃：嘗試用 CNN 取代步驟 2–4 的對齊。

## 專案結構

```
Image_Stitching/
├── main.py                 # 執行兩張拼接、多張拼接、圓柱全景
├── src/
│   ├── data_download.py    # 下載測試資料 skybox、以 CPU 渲染 32 張影像
│   ├── features.py         # SIFT、比值檢定
│   ├── transforms.py       # fit_translation / fit_affine / fit_homography、apply_transform、compute_error
│   ├── ransac.py           # 估計變換矩陣
│   └── stitcher.py         # 影像合成：畫布與變形、圓柱投影、average／feather blend、影像拼接、圓柱全景
├── data/frames/            # 00.jpg … 31.jpg 測試資料
├── data/skybox/            # 下載的 skybox 圖片
└── results/                # 輸出結果
```

## 使用說明

```bash
pip install -r requirements.txt

python main.py
python main.py --pano-ids 0 2 4 6 8 10 12 14 --out results
```

輸出（`results/`）：

| 檔案 | 內容 |
|------|------|
| `pair_01_03_average.png`、`pair_01_03_feather.png` | 兩張影像以 homography 拼接，比較    average／feather blend |
| `multiple.png` | 多張影像以 homography 拼接到中間那張的平面上 |
| `panorama.png`、`panorama_cropped.png` | 圓柱全景，以及裁掉黑邊的版本 |

常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--pair-ids` | `1 3` | 兩張拼接用的影像編號 |
| `--multi-ids` | `0 1 2 3 4` | 多張拼接用的影像編號 |
| `--pano-ids` | `0 2 4 6 8 10` | 圓柱全景用的影像編號 |
| `--input` | 無 | 自己的影像（資料夾或檔案），見下方說明 |
| `--ratio` | `0.75` | 比值檢定門檻 |
| `--threshold` | `3.0` | RANSAC 內點誤差門檻（px） |
| `--iters` | `1000` | RANSAC 迭代次數 |
| `--focal` | 約 724（自己的影像：影像寬度） | 圓柱投影的焦距（px） |

## 使用自己的影像進行拼接

把照片放進資料夾（例如 `data/my_images/`），用 `--input` 指定：

```bash
python main.py --input data/my_images           # 整個資料夾，依由左到右排序檔名
python main.py --input left.jpg right.jpg       # 直接列出拼接目標檔案
```

- **2 張**：執行兩張拼接 + 圓柱全景；**3 張以上**：執行多張拼接 + 圓柱全景。
- 照片要**由左到右依序**拍攝並照順序命名（例如 `01.jpg`、`02.jpg` …），相鄰兩張建議重疊 30–50%。
- 拍攝時盡量**原地轉動相機**、不要移動位置，曝光固定效果更好；天空、白牆等沒有紋理的區域很難找到配對點。
- 影像長邊預設縮到 1000 px（`--max-size`，0 表示不縮）以加快計算。
- `--focal` 預設為影像寬度（縮放後），全景出現明顯彎曲時可以調整。
- 總視角太大（接近 180°）時平面多張拼接會自動跳過，請看圓柱全景的結果。

## 測試資料

未指定 `--input`時，如果 `data/frames/` 裡缺少 `00.jpg … 31.jpg`，程式會自動：

1. 從 [LearnOpenGL](https://learnopengl.com/img/textures/skybox.zip) 下載 `skybox.zip`，並解壓到 `data/skybox/`
2. 在 CPU 上渲染 32 張影像到 `data/frames/`
