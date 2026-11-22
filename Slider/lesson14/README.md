# Lesson 14. Advanced Optimization (개선된 최적화)

| 파일 | 설명 |
|---|---|
| `Lesson14_lecture.pptx` / `.pdf` | 강의 슬라이드 64장 (발표자 노트 포함) |
| `Lesson14_해설집.md` | 초등학생도 이해할 수 있는 해설집 + 3시간 시간표 + 실습 풀이 + 원본 검토 |
| `code/optim_np.py` | NumPy 옵티마이저 모음: SGD(모멘텀) / Adam(AdamW) / Lion / Adafactor + 학습률 스케줄 |
| `code/step1_gd_ema.py` | 경사 하강 손계산, 학습률, EMA, 협곡에서 SGD vs 모멘텀 |
| `code/step2_adam.py` | Adam 손계산, 편향 보정, AdamW vs Adam+L2, **NumPy vs torch.optim 검증(assert)** |
| `code/step3_lion_adafactor.py` | Lion(부호), Adafactor(행·열 분해), 메모리 계산 |
| `code/step4_schedule_sharp.py` | 코사인·웜업·재시작 스케줄, sharp vs flat 장난감 실험 |
| `code/step5_numpy_mlp.py` | NumPy 2층 신경망(역전파 직접 구현) + 6가지 옵티마이저 비교 |
| `code/step6_torch_compare.py` | PyTorch 버전: torch.optim 비교, 직접 만든 Lion, 스케줄러, 파라미터 그룹 (결과: `_res6.npy`) |
| `code/practice_solutions.py` | 실습 문제 5개 솔루션 (assert) |
| `code/make_images.py` | 슬라이드 그림 생성 (`step6_torch_compare.py`를 먼저 실행) |
| `deck/` | 슬라이드 생성 소스. `npm install` 후 `node build.js` |

실행: `cd code && python step2_adam.py` (환경: numpy, scikit-learn, torch>=2.5, matplotlib). Windows 에서 한글이 깨지면 `set PYTHONUTF8=1`.
PDF 는 PowerPoint 에서 PPTX 를 열어 PDF 로 내보낸 것이다.
