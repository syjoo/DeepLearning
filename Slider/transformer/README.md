# Transformer 강의 자료

- `Transformer_lecture.pptx` : 강의 슬라이드 (80장, 6교시 = 3시간 x 2회, 마지막에 실습 문제 + 솔루션)
- `numpy_version/step1~6` : PyTorch/TensorFlow 없이 NumPy만으로 구현 (step5 = 미니 자동미분, step6 = 학습 + 글 생성)
- `pytorch_version/step1~6` : PyTorch 실전 버전 (step6 = nn.MultiheadAttention 과 결과 비교)
- `practice_solutions.py` : 실습 문제 정답
- `tiny_corpus.txt` : 학습용 짧은 글

실행: `cd numpy_version && python step6_transformer.py` / `cd pytorch_version && python step5_gpt_train.py`
슬라이드 재생성: `deck/build.js` (pptxgenjs 필요, 환경변수 PPTX_NODE_MODULES = node_modules 경로)
