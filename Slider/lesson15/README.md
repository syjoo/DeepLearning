# Lesson 15. Explainability (설명 가능성: 모델 결정의 이해)

| 파일 | 설명 |
|---|---|
| `Lesson15_lecture.pptx` / `.pdf` | 강의 슬라이드 69장 (발표자 노트 포함) |
| `Lesson15_해설집.md` | 초등학생도 이해할 수 있는 해설집 + 3시간 시간표 + 실습 풀이 + 원본 검토 |
| `code/step1_permutation.py` | 순열 중요도 (NumPy, 식 15.3) |
| `code/step2_shap.py` | 정확한 샤플리 값/SHAP (NumPy, 식 15.4) + 근사 + PyTorch 대조 |
| `code/step3_gradcam_numpy.py` | 손으로 만든 CNN 위의 Grad-CAM + 그래디언트 검사 (NumPy) |
| `code/step4_gradcam_torch.py` | PyTorch hook Grad-CAM, '도장' 지름길 실험, Integrated Gradients |
| `code/step5_attention.py` | 어텐션 해부(NumPy) + `nn.MultiheadAttention` 대조 |
| `code/step6_fraud_case.py` | 사기 탐지 종합 사례(가상 데이터, NumPy 신경망) |
| `code/practice_solutions.py` | 실습 문제 5개 솔루션 (assert 로 검증) |
| `code/make_images.py` | 슬라이드 그림 생성 (step1~6 을 먼저 실행) |
| `deck/` | 슬라이드 생성 소스. `npm install` 후 `node build.js` |

실행: `cd code && python step1_permutation.py` … `python step6_fraud_case.py` (환경: numpy, scikit-learn, torch, matplotlib).
Windows 콘솔 한글 깨짐 시 `set PYTHONIOENCODING=utf-8`.
PDF 는 PowerPoint(COM)로 `.pptx` 에서 변환했다.
