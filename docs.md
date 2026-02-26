# AI-Synthesizing 기술 문서

## 목차

1. [시스템 개요](#시스템-개요)
2. [파이프라인 상세](#파이프라인-상세)
3. [API 및 함수 레퍼런스](#api-및-함수-레퍼런스)
4. [사용 예시](#사용-예시)
5. [현재 상태 및 한계](#현재-상태-및-한계)
6. [향후 개선 방향](#향후-개선-방향)

---

## 시스템 개요

### 목적
사용자의 사진과 의류 상품 이미지를 입력으로, **사이즈 차트(cm) 기반**으로 S/M/L/XL/2XL 각 사이즈의 핏 차이를 시각적으로 보여주는 가상 피팅 시스템.

### 아키텍처

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│  사람 사진   │   │  옷 이미지   │   │  사이즈 차트  │
└──────┬──────┘   └──────┬──────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  │
  MediaPipe          GrabCut               │
  (포즈 감지)        (누끼)                 │
       │                  │                  │
       ▼                  ▼                  ▼
  어깨 위치/너비     옷 본체 마스크      비율 계산
  (px)              (하드 바이너리)     (target/ref)
       │                  │                  │
       └────────┬─────────┘──────────────────┘
                │
                ▼
         사이즈별 스케일링
         (옷 본체를 어깨×ratio px로)
                │
                ▼
         seamlessClone 합성
         (조명/색감 자동 매칭)
                │
                ▼
         FLUX.2 보정
         (옷 원본 + 합성 → 자연스러운 결과)
                │
                ▼
         사이즈별 결과 (S~2XL)
```

### 실행 환경

| 항목 | 요구사항 |
|------|---------|
| 런타임 | Google Colab |
| GPU | A100 (40/80GB VRAM) |
| Python | 3.12 |
| diffusers | 0.36.0+ |
| FLUX.2 모델 | black-forest-labs/FLUX.2-dev |
| HuggingFace | 로그인 필요 (gated model) |

---

## 파이프라인 상세

### Step 1: MediaPipe 포즈 감지

```python
PoseLandmarker Heavy → 랜드마크 11(왼쪽어깨), 12(오른쪽어깨)

SHOULDER_PX  = ||L_shoulder - R_shoulder||   # 픽셀 어깨 너비
SHOULDER_CTR = (L_shoulder + R_shoulder) / 2  # 배치 기준점
```

- 모델: `pose_landmarker_heavy.task` (자동 다운로드)
- 입력: person.jpg (RGB)
- 출력: 어깨 픽셀 너비, 어깨 중심점 좌표

### Step 2: GrabCut 누끼 (배경 제거)

```python
GrabCut(8회 반복) → 하드 바이너리 마스크
  ├── MORPH_CLOSE(5회) → 내부 구멍 채우기
  └── MORPH_OPEN(2회)  → 외곽 노이즈 제거

본체 너비 측정: 이미지 상단 20~60% 영역에서 최대 가로폭
```

- 하드 마스크: 0 또는 255 (반투명 없음)
- 흰배경 의류 이미지에 최적화

### Step 3: 사이즈별 스케일링

```
ratio = target_cm / ref_cm
target_body_px = SHOULDER_PX × ratio
scale = target_body_px / garment_body_w
```

| 사이즈 | 어깨(cm) | ratio | 옷 본체(px) 예시 |
|--------|----------|-------|-----------------|
| S | 46.4 | 0.975 | 132 |
| M | 47.6 | 1.000 | 136 |
| L | 49.5 | 1.040 | 141 |
| XL | 51.4 | 1.080 | 147 |
| 2XL | 53.3 | 1.120 | 152 |

### Step 4: seamlessClone 합성

```python
cv2.seamlessClone(garment_BGR, person_BGR, mask, center, NORMAL_CLONE)
```

- 포아송 블렌딩으로 조명/색감 자동 매칭
- 이미지 범위 초과 시 자동 클리핑
- seamlessClone 실패 시 알파 블렌딩 fallback

### Step 5: FLUX.2 보정

```python
pipe(
    prompt="...",
    image=[garment_r, comp_r],        # 옷 원본 + 합성
    guidance_scale=5,
    num_inference_steps=28,
    caption_upsample_temperature=0.15  # 이미지 분석 기반 프롬프트 보강
)
```

- 입력: 옷 원본 이미지 (색상/패턴/질감 기준) + 합성 이미지 (사이즈/위치 기준)
- 모델: `black-forest-labs/FLUX.2-dev` (bf16)
- 실패 시 seamlessClone 결과 그대로 사용

---

## API 및 함수 레퍼런스

### `extract_garment_hard(path)`
GrabCut으로 의류 배경 제거.

**입력**: 이미지 파일 경로  
**출력**: `(rgb, mask, body_w)` — RGB 배열, 바이너리 마스크, 옷 본체 너비(px)

### `composite_size(person_bgr, garment_rgb, garment_mask, body_w, target_cm, ref_cm)`
사이즈 비율로 옷 스케일링 후 seamlessClone 합성.

**입력**:
- `person_bgr`: 사람 이미지 (BGR)
- `garment_rgb`, `garment_mask`: 누끼 결과
- `body_w`: 옷 본체 너비
- `target_cm`, `ref_cm`: 목표/기준 어깨 너비

**출력**: 합성 결과 (BGR)

### `flux_refine(person_pil, garment_pil, composite_pil, size_label, seed)`
FLUX.2로 합성 결과를 자연스럽게 보정.

**입력**: 원본 사람, 원본 옷, 합성 결과, 사이즈 라벨  
**출력**: 보정된 이미지 (PIL)

---

## 사용 예시

```python
# 사이즈 차트 설정
SIZE_CHART = {
    "S": 46.4, "M": 47.6, "L": 49.5, "XL": 51.4, "2XL": 53.3
}

# 기준 사이즈 (옷 이미지의 실제 사이즈)
ref_size = "M"

# 실행
for sz, target_cm in SIZE_CHART.items():
    comp = composite_size(person, garment, mask, body_w, target_cm, ref_cm)
    result = flux_refine(person_pil, garment_pil, comp_pil, sz)
    result.save(f"tryon_{sz}.jpg")
```

---

## 현재 상태 및 한계

### ✅ 동작 확인
- 사이즈별 스케일링: 픽셀 단위 차이 확인 (S: 132px → 2XL: 152px)
- GrabCut 누끼: 흰배경 의류 이미지에서 배경 제거 성공
- seamlessClone 합성: 조명/색감 자연스러운 합성
- FLUX.2 파이프라인: 모델 로드 및 추론 동작

### ⚠️ 알려진 이슈
- **seamlessClone 색감 동화**: 흰옷이 피부색에 동화되는 경우  
- **FLUX.2 사이즈 유지**: 보정 후 사이즈가 달라질 수 있음
- **어깨 기준 차이 체감**: S~2XL = 15% 차이로 시각적 차이 미묘

### 🔮 향후 개선 방향
1. **DensePose 워핑**: 단순 스케일링 → 3D 체형에 맞는 입체 변형
2. **Inpainting**: 옷 영역만 마스킹하여 자연스럽게 재생성
3. **SAM 누끼**: GrabCut 대신 Segment Anything Model
4. **가슴단면 반영**: 어깨 외 가슴 너비도 스케일링에 반영
5. **전용 Try-On 모델**: IDM-VTON 등 목적 특화 모델 연동
