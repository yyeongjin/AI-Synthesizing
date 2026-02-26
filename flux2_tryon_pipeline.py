# %% [markdown]
# # FLUX.2 [klein] 4B — 사이즈별 Virtual Try-On
#
# 1. GrabCut 누끼 (하드 마스크, 반투명 없음)
# 2. 사이즈별 스케일링 (본체 너비 기준)
# 3. seamlessClone으로 자연스러운 합성
# 4. FLUX.2로 최종 보정 [원본사람 + 합성결과]

# %% [markdown]
# ## 1. 설치

# %%
import os
!pip install -q mediapipe bitsandbytes

# %%
import torch, numpy as np, cv2, time
import mediapipe as mp
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %% [markdown]
# ## 2. 설정

# %%
PERSON_PATH  = "/content/person.jpg"
GARMENT_PATH = "/content/garment.jpg"
OUTPUT_DIR   = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42

# %% [markdown]
# ## 3. FLUX.2 로드

# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
).to(DEVICE)
print("✅ FLUX.2 로드 완료!")

# %% [markdown]
# ## 4. 사이즈 차트

# %%
#@title 👕 사이즈 차트
S_어깨 = 46.4 #@param {type:"number"}
M_어깨 = 47.6 #@param {type:"number"}
L_어깨 = 49.5 #@param {type:"number"}
XL_어깨 = 51.4 #@param {type:"number"}
XXL_어깨 = 53.3 #@param {type:"number"}

SIZE_CHART = {
    "S": S_어깨, "M": M_어깨, "L": L_어깨, "XL": XL_어깨, "2XL": XXL_어깨,
}
for sz, v in SIZE_CHART.items():
    print(f"  {sz:>3}: 어깨 {v}cm")

# %% [markdown]
# ## 5. 사람 측정

# %%
MODEL_PATH = "/content/pose_landmarker_heavy.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        MODEL_PATH,
    )
    print("✅ PoseLandmarker 다운로드")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# %%
#@title 📏 사람 측정
height_cm = 175 #@param {type:"integer"}

_img = cv2.imread(PERSON_PATH)
_rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
H_PX, W_PX = _img.shape[:2]

with PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
)) as lmk:
    _det = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=_rgb))

assert _det.pose_landmarks, "포즈 감지 실패!"
_lm = _det.pose_landmarks[0]
LS = np.array([_lm[11].x * W_PX, _lm[11].y * H_PX])
RS = np.array([_lm[12].x * W_PX, _lm[12].y * H_PX])
SHOULDER_PX = np.linalg.norm(LS - RS)
SHOULDER_CTR = ((LS + RS) / 2).astype(int)
print(f"📏 어깨 {SHOULDER_PX:.0f}px, 중심 ({SHOULDER_CTR[0]}, {SHOULDER_CTR[1]})")

# %% [markdown]
# ## 6. 옷 누끼 (하드 마스크)

# %%
def extract_garment_hard(path):
    """GrabCut → 하드 바이너리 마스크 (반투명 없음)"""
    img = cv2.imread(path)
    h, w = img.shape[:2]

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mx, my = int(w * 0.15), int(h * 0.05)
    mask[my:h-my, mx:w-mx] = cv2.GC_PR_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)

    # 하드 마스크: 전경=255, 배경=0 (반투명 없음)
    hard = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 노이즈 제거 + 구멍 채우기
    k = np.ones((5, 5), np.uint8)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, k, iterations=5)  # 구멍 채우기
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, k, iterations=2)   # 노이즈 제거

    # 본체 너비: 상단 20~60% 범위 (어깨 부근)
    top, bot = int(h * 0.2), int(h * 0.6)
    region = hard[top:bot, :]
    cols = np.where(region.max(axis=0) > 0)[0]
    body_w = cols[-1] - cols[0] if len(cols) > 1 else w

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ 누끼: 본체 {body_w}px / 전체 {w}px ({body_w/w:.0%})")
    return rgb, hard, body_w


GARMENT_RGB, GARMENT_MASK, GARMENT_BODY_W = extract_garment_hard(GARMENT_PATH)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(Image.open(GARMENT_PATH)); ax[0].set_title("원본"); ax[0].axis("off")
ax[1].imshow(GARMENT_MASK, cmap="gray"); ax[1].set_title("마스크"); ax[1].axis("off")
masked = GARMENT_RGB.copy(); masked[GARMENT_MASK == 0] = 200
ax[2].imshow(masked); ax[2].set_title("누끼 결과"); ax[2].axis("off")
plt.show()

# %% [markdown]
# ## 7. 합성 함수

# %%
def composite_size(person_bgr, garment_rgb, garment_mask, body_w,
                   target_cm, ref_cm):
    """
    seamlessClone으로 자연스럽게 합성

    1. 비율 = target_cm / ref_cm
    2. 옷 본체가 사람_어깨 × 비율 되도록 스케일
    3. cv2.seamlessClone으로 합성 (조명/색감 자동 매칭)
    """
    ratio = target_cm / ref_cm
    target_body_px = int(SHOULDER_PX * ratio)
    scale = target_body_px / body_w

    gh, gw = garment_rgb.shape[:2]
    nw, nh = int(gw * scale), int(gh * scale)

    g_resized = cv2.resize(garment_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    m_resized = cv2.resize(garment_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # 붙일 위치 (어깨 중심)
    cx, cy = SHOULDER_CTR[0], SHOULDER_CTR[1]
    # 옷 상단이 어깨 약간 위에 오도록
    center_y = cy + nh // 2 - int(nh * 0.2)
    center_x = cx

    # seamlessClone은 중심점 기준
    # 이미지 범위 체크
    ph, pw = person_bgr.shape[:2]

    # 옷이 이미지 밖으로 나가면 자르기
    x1 = center_x - nw // 2
    y1 = center_y - nh // 2
    x2, y2 = x1 + nw, y1 + nh

    # 클리핑
    cx1 = max(0, -x1)
    cy1 = max(0, -y1)
    cx2 = nw - max(0, x2 - pw)
    cy2 = nh - max(0, y2 - ph)

    g_crop = g_resized[cy1:cy2, cx1:cx2]
    m_crop = m_resized[cy1:cy2, cx1:cx2]

    # center for seamlessClone
    paste_cx = max(0, x1) + g_crop.shape[1] // 2
    paste_cy = max(0, y1) + g_crop.shape[0] // 2

    # seamlessClone (RGB → BGR 변환)
    g_bgr = cv2.cvtColor(g_crop, cv2.COLOR_RGB2BGR)
    try:
        result = cv2.seamlessClone(g_bgr, person_bgr, m_crop,
                                    (paste_cx, paste_cy), cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone 실패시 직접 합성
        result = person_bgr.copy()
        ry1 = max(0, y1)
        rx1 = max(0, x1)
        m3 = np.stack([m_crop/255.0]*3, axis=-1)
        region = result[ry1:ry1+g_crop.shape[0], rx1:rx1+g_crop.shape[1]]
        region[:] = (g_bgr * m3 + region * (1 - m3)).astype(np.uint8)

    print(f"  {target_cm}cm: ratio={ratio:.3f}, 본체 {body_w}→{target_body_px}px, 옷 {nw}x{nh}")
    return result


def resize_for_flux(img, target=1024):
    w, h = img.size
    s = target / max(w, h)
    nw = int(w * s) // 16 * 16
    nh = int(h * s) // 16 * 16
    return img.resize((nw, nh), Image.LANCZOS), (nw, nh)


def flux_refine(person_pil, garment_pil, composite_pil, size_label, seed=42):
    """
    옷 원본 + 합성 이미지 → FLUX.2 → 자연스러운 try-on
    image[0] = 원본 옷 (색상/패턴/질감)
    image[1] = 합성 이미지 (사이즈/위치/사람)
    """
    _, msize = resize_for_flux(composite_pil, 1024)
    garment_r = garment_pil.resize(msize, Image.LANCZOS)
    comp_r = composite_pil.resize(msize, Image.LANCZOS)

    prompt = (
        "The person in image 2 is wearing a garment. "
        "Make the garment look exactly like image 1 in color, pattern, and texture. "
        "Keep the garment at the same size and position as image 2. "
        "Keep the person's face, body, tattoos, pose, and background from image 2. "
        "Photorealistic photo with natural fabric draping and shadows."
    )

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    try:
        out = pipe(
            prompt=prompt,
            image=[garment_r, comp_r],
            height=msize[1], width=msize[0],
            guidance_scale=5,
            num_inference_steps=28,
            caption_upsample_temperature=0.15,
            generator=gen,
        ).images[0]
        return out.resize(person_pil.size, Image.LANCZOS)
    except Exception as e:
        print(f"    ⚠️ FLUX.2 실패: {e}")
        return composite_pil

# %% [markdown]
# ## 8. 실행

# %%
#@title 🚀 실행
ref_size = "M" #@param ["S", "M", "L", "XL", "2XL"] {type:"string"}
#@markdown ↑ 옷 이미지가 어떤 사이즈인지
use_ai = True #@param {type:"boolean"}

person_pil = Image.open(PERSON_PATH).convert("RGB")
garment_pil = Image.open(GARMENT_PATH).convert("RGB")
person_bgr = cv2.imread(PERSON_PATH)
ref_cm = SIZE_CHART[ref_size]

print(f"기준: {ref_size} ({ref_cm}cm), 본체: {GARMENT_BODY_W}px, 어깨: {SHOULDER_PX:.0f}px\n")

composites = {}
results = {}

for sz, target_cm in SIZE_CHART.items():
    t0 = time.time()

    # 1단계: seamlessClone 합성
    comp_bgr = composite_size(
        person_bgr, GARMENT_RGB, GARMENT_MASK, GARMENT_BODY_W,
        target_cm, ref_cm,
    )
    comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
    comp_pil = Image.fromarray(comp_rgb)
    composites[sz] = comp_pil

    # 2단계: FLUX.2 보정
    if use_ai:
        refined = flux_refine(person_pil, garment_pil, comp_pil, sz, seed=SEED)
        results[sz] = refined
        print(f"    → AI ✅ ({time.time()-t0:.1f}s)")
    else:
        results[sz] = comp_pil
        print(f"    ✅")

    results[sz].save(os.path.join(OUTPUT_DIR, f"tryon_{sz}.jpg"), quality=95)

# %% [markdown]
# ## 9. 결과

# %%
sizes = list(SIZE_CHART.keys())
n = len(sizes)

fig, axes = plt.subplots(2, n+1, figsize=(4*(n+1), 10))

axes[0][0].imshow(Image.open(GARMENT_PATH))
axes[0][0].set_title("원본 옷", fontsize=11)
axes[0][0].axis("off")
for i, sz in enumerate(sizes):
    r = SIZE_CHART[sz] / ref_cm
    axes[0][i+1].imshow(composites[sz])
    axes[0][i+1].set_title(f"{sz} 합성\n{SIZE_CHART[sz]}cm ({r:.2f}x)", fontsize=9)
    axes[0][i+1].axis("off")

axes[1][0].imshow(person_pil)
axes[1][0].set_title("원본 사람", fontsize=11)
axes[1][0].axis("off")
for i, sz in enumerate(sizes):
    axes[1][i+1].imshow(results[sz])
    axes[1][i+1].set_title(f"{sz} {'AI' if use_ai else ''}", fontsize=10)
    axes[1][i+1].axis("off")

plt.suptitle(f"사이즈별 Try-On (기준: {ref_size})", fontsize=14)
plt.tight_layout()
plt.show()

# %%
try:
    from google.colab import files
    uploaded = files.upload()
    for fn, data in uploaded.items():
        with open(f"/content/{fn}", "wb") as f:
            f.write(data)
        print(f"저장: /content/{fn}")
except ImportError:
    pass
