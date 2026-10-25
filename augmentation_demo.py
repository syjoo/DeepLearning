"""데이터 증강(Data Augmentation) 실험 예제 - PyTorch 없이 Pillow + NumPy만 사용.

실행:
    python augmentation_demo.py                 # 내장 샘플 이미지 사용
    python augmentation_demo.py my_photo.jpg    # 내 이미지 사용
    python augmentation_demo.py my_photo.jpg --seed 7 --out results

결과(out 폴더에 PNG 4장):
    1_each_transform.png    변환 하나씩 적용한 모습
    2_online_epochs.png     온라인 증강: 같은 이미지가 epoch마다 다르게 보임
    3_label_consistency.png 레이블 일관성 (6을 180° 회전하면 9처럼 보임)
    4_pipeline_grid.png     여러 변환을 조합한 무작위 파이프라인 결과
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

SIZE = 224  # 모든 이미지를 SIZE x SIZE 로 맞춤


# ---------------------------------------------------------------- 샘플 이미지
def make_sample_image(size=SIZE):
    """좌우가 비대칭인 간단한 장면(집 + 해 + 나무)을 그린다.
    비대칭이어야 뒤집기/회전 효과가 눈에 잘 보인다."""
    img = Image.new("RGB", (size, size), (135, 200, 235))           # 하늘
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(size * 0.72), size, size], fill=(90, 170, 90))  # 잔디
    d.ellipse([int(size * 0.68), 12, int(size * 0.68) + 44, 56], fill=(255, 210, 50))  # 해(오른쪽 위)
    d.rectangle([40, 110, 120, 170], fill=(200, 90, 70))              # 집 몸체(왼쪽)
    d.polygon([(30, 110), (80, 70), (130, 110)], fill=(120, 50, 40))  # 지붕
    d.rectangle([68, 135, 92, 170], fill=(250, 235, 200))             # 문
    d.rectangle([165, 120, 177, 172], fill=(110, 70, 40))             # 나무 기둥(오른쪽)
    d.ellipse([145, 80, 197, 132], fill=(30, 120, 50))                # 나무 잎
    return img


def make_digit_image(char="6", size=SIZE):
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    font = None
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "malgunbd.ttf"):
        try:
            font = ImageFont.truetype(name, int(size * 0.8))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size=int(size * 0.8))  # Pillow >= 10.1
    d.text((size / 2, size / 2), char, fill=(20, 40, 90), font=font, anchor="mm")
    return img


# ---------------------------------------------------------------- 변환 함수들
# 모두 (PIL.Image, rng) -> PIL.Image. rng는 np.random.Generator (재현 가능한 무작위).
def hflip(img, rng):
    return ImageOps.mirror(img)


def rotate(img, rng, max_deg=15):
    ang = rng.uniform(-max_deg, max_deg)
    return img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def random_resized_crop(img, rng, scale=(0.5, 1.0)):
    w, h = img.size
    s = rng.uniform(*scale)                       # 원본 면적 대비 잘라낼 비율
    cw, ch = int(w * np.sqrt(s)), int(h * np.sqrt(s))
    x0 = int(rng.integers(0, w - cw + 1))
    y0 = int(rng.integers(0, h - ch + 1))
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)


def translate_zoom(img, rng, max_shift=0.1, zoom=(0.9, 1.2)):
    w, h = img.size
    z = rng.uniform(*zoom)
    dx, dy = rng.uniform(-max_shift, max_shift, 2) * np.array([w, h])
    # affine: 출력 좌표 -> 입력 좌표 (중심 기준 줌 + 이동)
    a = 1 / z
    cx, cy = w / 2, h / 2
    return img.transform(
        (w, h), Image.AFFINE,
        (a, 0, cx - a * cx - dx * a, 0, a, cy - a * cy - dy * a),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def brightness_contrast(img, rng):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))


def color_jitter(img, rng):
    # 채도(saturation) 조절 + 색조(hue) 이동
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.6))
    hsv = np.array(img.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-12, 13))) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def gaussian_noise(img, rng, sigma=15):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape)   # 평균 0, 표준편차 sigma 의 잡음
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


TRANSFORMS = {
    "Horizontal Flip": hflip,
    "Rotation (+-15)": rotate,
    "Crop & Resize": random_resized_crop,
    "Translate & Zoom": translate_zoom,
    "Brightness/Contrast": brightness_contrast,
    "Color Jitter": color_jitter,
    "Gaussian Noise": gaussian_noise,
}


def augment(img, rng, p=0.5):
    """온라인 증강 파이프라인: 작은 변환들을 각각 확률 p 로 적용해 '조합'한다."""
    for fn in TRANSFORMS.values():
        if rng.random() < p:
            img = fn(img, rng)
    return img


# ---------------------------------------------------------------- 그리드 저장
def save_grid(images, titles, path, cols, cell=SIZE):
    pad, label_h = 8, 22
    rows = (len(images) + cols - 1) // cols
    W = cols * (cell + pad) + pad
    H = rows * (cell + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (im, t) in enumerate(zip(images, titles)):
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + label_h + pad)
        d.text((x, y + 4), t, fill=(30, 30, 30))
        sheet.paste(im.resize((cell, cell)), (x, y + label_h))
    sheet.save(path)
    print("saved:", path)


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="입력 이미지 경로 (생략하면 내장 샘플)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="aug_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)  # 같은 seed -> 같은 결과 (재현성)

    if args.image:
        img = ImageOps.fit(Image.open(args.image).convert("RGB"), (SIZE, SIZE))
    else:
        img = make_sample_image()

    # 1) 변환 하나씩
    imgs = [img] + [fn(img, rng) for fn in TRANSFORMS.values()]
    titles = ["Original"] + list(TRANSFORMS)
    save_grid(imgs, titles, os.path.join(args.out, "1_each_transform.png"), cols=4)

    # 2) 온라인 증강: epoch 마다 같은 원본에서 매번 다른 결과
    imgs = [img] + [augment(img, rng) for _ in range(7)]
    titles = ["Original"] + [f"epoch {i + 1}" for i in range(7)]
    save_grid(imgs, titles, os.path.join(args.out, "2_online_epochs.png"), cols=4)

    # 3) 레이블 일관성: 6 -> 180도 회전 -> 9처럼 보임
    six = make_digit_image("6")
    flipped = six.rotate(180)
    save_grid([six, flipped, make_digit_image("9")],
              ["label=6 (original)", "6 rotated 180 -> looks like 9 (label still 6?!)", "real 9 for comparison"],
              os.path.join(args.out, "3_label_consistency.png"), cols=3)

    # 4) 조합 파이프라인 결과 12장
    imgs = [augment(img, rng) for _ in range(12)]
    save_grid(imgs, [f"sample {i + 1}" for i in range(12)],
              os.path.join(args.out, "4_pipeline_grid.png"), cols=4)

    # 간단한 수치 확인: 증강 후에도 픽셀 평균/표준편차가 어떻게 변하는지
    base = np.asarray(img, dtype=np.float32)
    print(f"\noriginal mean={base.mean():.1f} std={base.std():.1f}")
    for name, fn in TRANSFORMS.items():
        a = np.asarray(fn(img, rng), dtype=np.float32)
        print(f"{name:22s} mean={a.mean():6.1f} std={a.std():6.1f}")


if __name__ == "__main__":
    main()
