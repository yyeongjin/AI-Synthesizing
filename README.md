# AI-Synthesizing: 사이즈별 Virtual Try-On

FLUX.2 기반 가상 피팅 시스템. 사이즈 차트(cm)로 의류의 실제 핏 차이를 시각적으로 보여줍니다.

## 핵심 아이디어

> 옷 이미지를 사이즈 비율대로 직접 스케일링 → 사람 위에 합성 → FLUX.2로 자연스럽게 보정

## 파이프라인

```
사람 사진 + 옷 사진 + 사이즈 차트(cm)
       ↓
① MediaPipe → 사람 어깨 위치/너비 감지
② GrabCut → 옷 누끼 (배경 제거, 하드 마스크)
③ 사이즈별 비율로 옷 스케일링 (S/M/L/XL/2XL)
④ seamlessClone → 사람 위에 합성
⑤ FLUX.2 → 옷 원본 + 합성 이미지로 자연스럽게 보정
       ↓
사이즈별 Try-On 결과 (5장)
```

## 사용법

### 1. Google Colab에서 실행

`flux2_tryon_pipeline.ipynb`을 Colab에 업로드 후 순서대로 실행.

### 2. 필요 사항

| 항목 | 설명 |
|------|------|
| **GPU** | A100 권장 (FLUX.2 bf16 ~24GB VRAM) |
| **HuggingFace** | 로그인 필요 (gated model) |
| **person.jpg** | 사람 사진 (상반신/전신) |
| **garment.jpg** | 의류 상품 이미지 (흰배경 권장) |

### 3. 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `ref_size` | 옷 이미지의 실제 사이즈 | M |
| `use_ai` | FLUX.2 보정 사용 여부 | True |
| `height_cm` | 사람 키 (cm) | 175 |
| 사이즈 차트 | S~2XL 어깨너비 (cm) | 쇼핑몰에서 입력 |

## 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| 포즈 감지 | MediaPipe PoseLandmarker Heavy |
| 옷 누끼 | OpenCV GrabCut + Morphology |
| 합성 | OpenCV seamlessClone |
| AI 보정 | FLUX.2-dev (bf16) |

## 파일 구조

```
AI-Synthesizing/
├── README.md                       # 이 문서
├── docs.md                         # 상세 기술 문서
├── flux2_tryon_pipeline.ipynb      # Colab 노트북
└── flux2_tryon_pipeline.py         # 소스 코드
```

## 사이즈 스케일링 원리

```
기준: 옷 이미지 = M (47.6cm)
사람 어깨 = 136px

S:   136 × (46.4/47.6) = 132px
M:   136 × (47.6/47.6) = 136px  (기준)
L:   136 × (49.5/47.6) = 141px
XL:  136 × (51.4/47.6) = 147px
2XL: 136 × (53.3/47.6) = 152px
```
