"""測試資料：下載 LearnOpenGL 的 skybox，並渲染 32 張的全景影像序列。

可直接執行（`python -m src.data_download`），或由 main.py 呼叫 `ensure_frames()`。

區塊：
  1. PanoramaRenderer - 純 CPU 的渲染器（NumPy + OpenCV，不需要 OpenGL）。用針孔相機繞垂直軸旋轉，
     從 cubemap 取出透視視角的影像，並可加入各種不完美因素
     （相機隨機旋轉、暗角、曝光閃爍、感測器雜訊）。
     使用水平視角：f = W / (2 tan(fov/2))。
  2. 下載 + ensure_frames - 缺少 skybox 時下載，缺少 00.jpg ... 31.jpg 時渲染。
"""
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# 1. PanoramaRenderer（全景渲染器）
# ----------------------------------------------------------------------
# LearnOpenGL skybox 的 cube-map 各面順序
_FACES = ["right", "left", "top", "bottom", "front", "back"]


class PanoramaRenderer:
    def __init__(self, skybox_dir="skybox", width=800, height=600, fov=62.0,
                 vignette=0.35, exposure_jitter=0.12, noise_sigma=2.0,
                 face_size=1536, seed=327):
        self.W, self.H, self.fov = width, height, fov
        self.vignette = vignette
        self.exposure_jitter = exposure_jitter
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        self.f = width / (2.0 * np.tan(np.radians(fov) / 2.0))

        self.faces = []
        for name in _FACES:
            img = cv2.imread(os.path.join(skybox_dir, name + ".jpg"), cv2.IMREAD_COLOR_RGB)
            if img is None:
                raise FileNotFoundError(f"{skybox_dir}/{name}.jpg not found")
            self.faces.append(cv2.resize(img, (face_size, face_size), interpolation=cv2.INTER_AREA))
        self.face_size = face_size

        # 每個像素在相機座標系中的光線方向（x 向右、y 向下、z 向前）
        xs = (np.arange(width) - (width - 1) / 2.0) / self.f
        ys = (np.arange(height) - (height - 1) / 2.0) / self.f
        gx, gy = np.meshgrid(xs, ys)
        self.cam_rays = np.stack([gx, gy, np.ones_like(gx)], axis=-1).astype(np.float64)

        # 暗角遮罩
        r = np.hypot(*np.meshgrid((np.arange(width) - width / 2) / (width / 2),
                                  (np.arange(height) - height / 2) / (height / 2)))
        self.vig = (1.0 - self.vignette * np.clip(r / np.sqrt(2), 0, 1) ** 2)[..., None]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotation(yaw, pitch, roll):
        """回傳矩陣的三個 column 依序為相機的右 / 下 / 前方向，以世界座標表示（y 軸朝上）。"""
        yaw, pitch, roll = np.radians([yaw, pitch, roll])
        fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)])
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # 繞光軸旋轉（roll）
        c, s = np.cos(roll), np.sin(roll)
        right, up = c * right + s * up, -s * right + c * up
        return np.stack([right, -up, fwd], axis=1)

    def _sample_cubemap(self, d):
        """d：(H,W,3) 世界座標方向 -> (H,W,3) float32 顏色（依 OpenGL cube-map 慣例）。"""
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
        out = np.zeros(d.shape, dtype=np.float32)
        n = self.face_size
        # (遮罩, 面索引, sc, tc, ma)，依照 OpenGL 規格書中的對照表
        cases = [
            ((ax >= ay) & (ax >= az) & (x > 0), 0, -z, -y, ax),   # +X 右
            ((ax >= ay) & (ax >= az) & (x <= 0), 1, z, -y, ax),    # -X 左
            ((ay > ax) & (ay >= az) & (y > 0), 2, x, z, ay),       # +Y 上
            ((ay > ax) & (ay >= az) & (y <= 0), 3, x, -z, ay),     # -Y 下
            ((az > ax) & (az > ay) & (z > 0), 4, x, -y, az),       # +Z 前
            ((az > ax) & (az > ay) & (z <= 0), 5, -x, -y, az),     # -Z 後
        ]
        for mask, fi, sc, tc, ma in cases:
            if not mask.any():
                continue
            u = (sc / np.maximum(ma, 1e-9) + 1.0) * 0.5 * n - 0.5
            v = (tc / np.maximum(ma, 1e-9) + 1.0) * 0.5 * n - 0.5
            mapped = cv2.remap(self.faces[fi], u.astype(np.float32), v.astype(np.float32),
                               interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[mask] = mapped[mask]
        return out

    def render_view(self, yaw, pitch=0.0, roll=0.0, rand_rot=0.0):
        """渲染一張影像。`rand_rot` 是相機隨機擾動的標準差（度）。"""
        if rand_rot > 0:
            yaw += self.rng.normal(0, rand_rot)
            pitch += self.rng.normal(0, rand_rot)
            roll += self.rng.normal(0, rand_rot)
        R = self._rotation(yaw, pitch, roll)
        d = self.cam_rays @ R.T
        img = self._sample_cubemap(d)

        gain = 1.0 + self.rng.uniform(-self.exposure_jitter, self.exposure_jitter)
        img = img * self.vig * gain
        img = img + self.rng.normal(0, self.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render_sequence(self, output_folder="output", num_images=24, pitch=6.0, rand_rot=1.0,
                        start_yaw=-180.0, verbose=True):
        """水平掃過完整 360 度，存成 00.jpg、01.jpg、..."""
        os.makedirs(output_folder, exist_ok=True)
        yaws = start_yaw + np.arange(num_images) * 360.0 / num_images
        for i, yaw in enumerate(yaws):
            img = self.render_view(yaw, pitch, 0.0, rand_rot)
            path = os.path.join(output_folder, f"{i:02d}.jpg")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if verbose:
                print(f"[{i + 1:02d}/{num_images}] yaw={yaw:7.1f} deg -> {path}")
        return yaws


# ----------------------------------------------------------------------
# 2. 下載 + ensure_frames
# ----------------------------------------------------------------------
SKYBOX_URL = "https://learnopengl.com/img/textures/skybox.zip"
# learnopengl.com 會對 Python 預設（或太短）的 User-Agent 回 HTTP 406
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 序列的虛擬相機：800 x 600 px，垂直視角 45 度
IMG_W, IMG_H = 800, 600
CAMERA_FOV_V = 45.0
FOCAL = (IMG_H / 2) / np.tan(np.radians(CAMERA_FOV_V) / 2)       # 約 724 px
NUM_FRAMES = 32


def _has_frames(frames_dir: Path, n: int) -> bool:
    return all((frames_dir / f"{i:02d}.jpg").is_file() for i in range(n))


def _has_skybox(skybox_dir: Path) -> bool:
    return all((skybox_dir / f"{f}.jpg").is_file() for f in _FACES)


def download_skybox(skybox_dir: PathLike) -> Path:
    """下載 LearnOpenGL 的 skybox.zip 並解壓到 `skybox_dir`（已存在則略過）。"""
    skybox_dir = Path(skybox_dir)
    if _has_skybox(skybox_dir):
        return skybox_dir
    skybox_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = skybox_dir.parent / "skybox.zip"
    print(f"downloading skybox: {SKYBOX_URL}")
    try:
        req = urllib.request.Request(SKYBOX_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(skybox_dir.parent)          # 壓縮檔內含 skybox/ 資料夾
    except Exception as e:
        raise RuntimeError(
            f"Automatic download failed. Download {SKYBOX_URL} manually and unzip it so that "
            f"{skybox_dir}/ contains right/left/top/bottom/front/back.jpg") from e
    finally:
        zip_path.unlink(missing_ok=True)
    if not _has_skybox(skybox_dir):
        raise RuntimeError(f"{skybox_dir}/*.jpg not found after unzipping")
    return skybox_dir


def ensure_frames(frames_dir: PathLike = "data/frames", skybox_dir: PathLike = "data/skybox",
                  num_frames: int = NUM_FRAMES) -> Path:
    """若 `frames_dir` 缺少 00.jpg ... 31.jpg，就下載 skybox 並渲染整個序列。"""
    frames_dir = Path(frames_dir)
    if _has_frames(frames_dir, num_frames):
        return frames_dir

    download_skybox(skybox_dir)
    print(f"rendering {num_frames} frames into {frames_dir}/ ...")
    fov_h = np.degrees(2 * np.arctan(IMG_W / (2 * FOCAL)))
    PanoramaRenderer(str(skybox_dir), IMG_W, IMG_H, fov=fov_h, vignette=0.35,
                     exposure_jitter=0.0, noise_sigma=0.0, seed=327).render_sequence(
        str(frames_dir), num_images=num_frames, pitch=0.0, rand_rot=2.0, verbose=False)
    return frames_dir


if __name__ == "__main__":
    ensure_frames()
