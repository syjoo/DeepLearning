# Lesson 13. Data Augmentation (데이터 증강)

| 파일 | 설명 |
|---|---|
| `Lesson13_lecture.pptx` / `.pdf` | 강의 슬라이드 56장 (발표자 노트 포함) |
| `Lesson13_해설집.md` | 초등학생도 이해할 수 있는 해설집 + 3시간 시간표 + 실습 풀이 + 원본 검토 |
| `code/step1_transforms.py` | 변환(뒤집기·이동·회전·확대·밝기·노이즈) NumPy 구현 |
| `code/step2_label.py` | 레이블 일관성 실험 |
| `code/step3_pipeline.py` | 온라인 증강 파이프라인 (PyTorch) |
| `code/step4_compare.py` | 증강 유무 비교 실험 (결과: `_res.npy`) |
| `code/step5_advanced.py` | Cutout / Mixup / CutMix / TTA |
| `code/practice_solutions.py` | 실습 문제 솔루션 |
| `code/make_images.py` | 슬라이드 그림 생성 (`step4_compare.py`를 먼저 실행) |
| `deck/` | 슬라이드 생성 소스. `npm install` 후 `node build.js` |

실행: `cd code && python step4_compare.py` (환경: numpy, scikit-learn, torch, matplotlib)
