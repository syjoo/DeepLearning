// 2교시: 변환 하나하나를 NumPy로 구현
const { C, FONT } = require("./lib");
const { img, F, table } = require("./helpers");

module.exports = function part2(d, pres) {
  d.section("2교시", "변환을 직접 만들어 보자", "뒤집기 · 이동 · 회전 · 확대 · 밝기 · 대비 · 노이즈  (Step 1: NumPy만 사용)", "2교시 시작. 라이브러리를 쓰기 전에 원리를 직접 구현해 본다. 모든 변환은 '위치를 옮기기'와 '값을 바꾸기' 두 종류.");

  {
    const s = d.slide("변환의 두 종류", { tag: "2교시 · 개요" });
    d.card(s, { x: 0.6, y: 1.55, w: 5.9, h: 3.0, fill: C.tealL, title: "① 기하 변환 (위치를 바꾼다)", tColor: C.teal, size: 16, body: "뒤집기, 이동, 회전, 확대/축소, 크롭\n\n· 픽셀의 '값'은 그대로, '자리'만 바뀜\n· 정답이 바뀔 위험: 방향이 의미인 데이터" });
    d.card(s, { x: 6.8, y: 1.55, w: 5.9, h: 3.0, fill: C.orangeL, title: "② 광도 변환 (값을 바꾼다)", tColor: C.orange, size: 16, body: "밝기, 대비, 색상 변화, 노이즈\n\n· '자리'는 그대로, 값만 바뀜\n· 정답이 바뀔 위험: 색이 의미인 데이터(신호등 등)" });
    d.bullets(s, ["두 종류를 섞어서 쓰는 것이 일반적이다 (예: 약간 돌리고 + 약간 어둡게 + 노이즈)", "각각을 '작게' 쓰고 '여러 개' 조합하는 것이 핵심 (Best Practice 3)"], { x: 0.6, y: 4.9, w: 12.1, h: 1.8, size: 18 });
  }

  // 뒤집기
  {
    const s = d.slide("수평 뒤집기 (Horizontal Flip)", { tag: "2교시 · 변환 1" });
    F(d, s, "x′[i, j] = x[i, W − 1 − j]", { x: 0.6, y: 1.5, w: 6.0, h: 0.8, size: 24, fill: C.tealL });
    d.code(s, ["def flip_h(img):", "    return img[:, ::-1]    # 열 순서만 거꾸로", "", "a = np.array([[1, 2, 3],", "              [4, 5, 6]])", "flip_h(a)   # [[3 2 1]", "            #  [6 5 4]]"], { x: 0.6, y: 2.6, w: 6.0, h: 3.4, size: 14, title: "step1_transforms.py" });
    d.bullets(s, [
      { t: "객체의 대칭성이 허용될 때만 사용", b: true },
      "○ 고양이, 자동차, 나뭇잎",
      "✕ 숫자, 문자, 교통 표지판",
      "코드는 한 줄: 배열 슬라이싱 ::-1",
      "상하 뒤집기는 img[::-1] (일반 사진에선 드묾)",
    ], { x: 7.0, y: 1.5, w: 5.7, h: 4.5, size: 18 });
    s.addNotes("NumPy 슬라이싱 [::-1]은 '끝에서 처음까지 거꾸로'. 작은 배열로 손으로 확인하게 한다.");
  }

  // 아핀 변환
  {
    const s = d.slide("회전·이동·확대는 '좌표 변환' 하나로 통일", { tag: "2교시 · 변환 2~4" });
    F(d, s, "입력 위치 q  =  A · (p − c) + c + t", { x: 0.6, y: 1.5, w: 12.1, h: 0.8, size: 26, fill: C.tealL });
    table(s, ["변환", "A (2×2 행렬)", "t (이동)", "쉬운 설명"], [
      ["회전 θ", "[[cos θ, sin θ], [−sin θ, cos θ]]", "0", "그림의 중심 c를 축으로 돌린다"],
      ["이동", "단위행렬 I", "(−dy, −dx)", "그림을 반대로 읽으면 그림이 밀려 보인다"],
      ["확대 s배", "I / s", "0", "더 작은 영역을 읽어 와 크게 보이게 한다"],
    ], { x: 0.6, y: 2.6, w: 12.1, colW: [1.8, 4.8, 2.0, 3.5], fontSize: 13, rowH: 0.5 });
    d.card(s, { x: 0.6, y: 5.25, w: 12.1, h: 1.55, fill: C.orangeL, title: "왜 '거꾸로' 계산할까? (역방향 매핑)", tColor: C.orange, size: 15, body: "앞으로 옮기면(입력 → 출력) 출력 칸 중 값이 안 채워지는 '구멍'이 생긴다.\n그래서 출력의 모든 칸에 대해 '이 칸은 입력의 어디에서 왔을까?'를 묻고, 그 위치의 값을 읽어 온다. 구멍이 생기지 않는다." });
    s.addNotes("아핀(affine) 변환 = 선형 변환 + 이동. 행렬 A와 벡터 t만 바꾸면 회전/이동/확대/기울이기가 모두 된다. 이 방식은 PyTorch의 affine_grid / grid_sample, OpenCV의 warpAffine과 같은 원리.");
  }

  {
    const s = d.slide("칸 사이의 값은? 쌍선형 보간", { tag: "2교시 · 변환 2~4" });
    d.text(s, "회전하면 읽어 올 위치가 정수 칸이 아니라 '칸 사이'(예: 2.3, 4.6)에 떨어진다. 이웃한 4칸을 거리에 비례해 섞는다.", { x: 0.6, y: 1.4, w: 12.1, h: 0.6, size: 17 });
    d.grid(s, [[0.2, 0.8], [0.4, 1.0]], { x: 1.2, y: 2.4, cw: 1.3, ch: 0.9, size: 20 });
    d.text(s, "위쪽 줄이 y=0, 왼쪽 열이 x=0 인 칸. 읽을 위치 (y=0.25, x=0.5)", { x: 1.2, y: 4.35, w: 5.5, h: 0.6, size: 14, color: C.muted });
    F(d, s, "값 = (1−fy)(1−fx)·a + (1−fy)·fx·b + fy·(1−fx)·c + fy·fx·d", { x: 6.2, y: 2.4, w: 6.5, h: 0.9, size: 17, fill: C.tealL });
    d.code(s, ["fy, fx = 0.25, 0.5", "a, b, c, d = 0.2, 0.8, 0.4, 1.0", "val = (1-fy)*(1-fx)*a + (1-fy)*fx*b + fy*(1-fx)*c + fy*fx*d", "print(val)    # 0.55"], { x: 6.2, y: 3.5, w: 6.5, h: 1.6, size: 13 });
    d.bullets(s, ["손계산: 0.75·0.5·0.2 + 0.75·0.5·0.8 + 0.25·0.5·0.4 + 0.25·0.5·1.0 = 0.075 + 0.3 + 0.05 + 0.125 = 0.55", "(0.2와 0.8의 중간은 0.5, 아래쪽 0.4와 1.0의 중간은 0.7 → 위쪽에 가까운 만큼 0.55로 결정)"], { x: 0.6, y: 5.3, w: 12.1, h: 1.4, size: 15 });
    s.addNotes("학생이 직접 손으로 계산해 보게 한다. 결과 0.55를 코드로 확인.");
  }

  // warp 코드
  {
    const s = d.slide("Step 1 코드 ①: warp (회전·이동·확대의 핵심)", { tag: "2교시 · 실습 Step 1" });
    d.code(s, [
      "def warp(img, A, t):",
      "    h, w = img.shape",
      "    c = np.array([(h-1)/2, (w-1)/2])           # 이미지 중심",
      "    ys, xs = np.mgrid[0:h, 0:w]",
      "    p = np.stack([ys, xs], -1).reshape(-1, 2) - c   # 출력 좌표",
      "    q = p @ A.T + c + t                        # 입력에서 읽을 위치",
      "    y0, x0 = floor(q[:,0]), floor(q[:,1])      # 왼쪽 위 칸",
      "    fy, fx = q[:,0]-y0, q[:,1]-x0              # 칸 안에서의 비율",
      "    # 4개 이웃의 가중 평균 (범위 밖은 0 = 흰 배경)",
      "    out = px(y0,x0)*(1-fy)*(1-fx) + px(y0,x0+1)*(1-fy)*fx \\",
      "        + px(y0+1,x0)*fy*(1-fx) + px(y0+1,x0+1)*fy*fx",
      "    return out.reshape(h, w)",
    ], { x: 0.4, y: 1.45, w: 7.9, h: 5.3, size: 12, title: "step1_transforms.py (px는 범위 밖이면 0을 돌려주는 도우미)" });
    d.explain(s, ["출력의 모든 칸 p를 중심 기준 좌표로 바꾼다", "p @ A.T + c + t 로 '입력의 어디서 읽을지' 계산", "읽을 위치 주변 4칸의 값을 비율로 섞는다", "입력 밖이면 0(흰색): 회전하면 모서리가 비는 이유"], { x: 8.5, y: 1.45, w: 4.3, h: 5.3 });
  }

  {
    const s = d.slide("Step 1 코드 ②: 회전·이동·확대는 warp에 행렬만 바꿔 끼운다", { tag: "2교시 · 실습 Step 1" });
    d.code(s, [
      "def rotate(img, deg):",
      "    th = np.deg2rad(deg)",
      "    A = np.array([[cos(th), sin(th)], [-sin(th), cos(th)]])",
      "    return warp(img, A, np.zeros(2))",
      "",
      "def shift(img, dy, dx):",
      "    return warp(img, np.eye(2), np.array([-dy, -dx]))",
      "",
      "def zoom(img, s):                    # s > 1 이면 확대",
      "    return warp(img, np.eye(2) / s, np.zeros(2))",
      "",
      "# 자체 검사 (코드의 마지막 부분)",
      "assert np.allclose(rotate(img, 0), img)      # 0도 회전 = 원본",
      "assert np.allclose(shift(img, 0, 0), img)    # 0칸 이동 = 원본",
      "# 오른쪽으로 1칸 이동해도 픽셀 값의 합은 같다 (18.375 → 18.375)",
    ], { x: 0.4, y: 1.45, w: 7.9, h: 5.3, size: 12, title: "step1_transforms.py" });
    d.explain(s, ["회전은 A(2×2 행렬)만 바뀐다", "이동은 t만 바뀐다. 부호가 '−'인 이유: 읽는 위치를 반대로 옮기면 그림이 밀린다", "확대는 A를 1/s로: 같은 칸 수로 더 작은 영역을 읽는다", "0도 회전과 0칸 이동이 원본과 같은지 assert로 확인한다"], { x: 8.5, y: 1.45, w: 4.3, h: 5.3 });
    s.addNotes("실제 실행 출력: 원본 합 18.375, 오른쪽 1칸 이동 후 합 18.375 (모서리 잘림이 없는 이미지라서 합이 같다). 밝기 +0.2 평균 0.479 vs 원본 0.287. 노이즈 0.1 후 평균 절대 차이 0.059.");
  }

  // 광도
  {
    const s = d.slide("밝기와 대비", { tag: "2교시 · 변환 5~6" });
    F(d, s, "x′ = clip( (x − 0.5) · c + 0.5 + b,  0, 1 )", { x: 0.6, y: 1.5, w: 12.1, h: 0.85, size: 26, fill: C.tealL });
    table(s, ["파라미터", "뜻", "예", "효과"], [
      ["b (밝기)", "모든 픽셀에 더하는 값", "+0.2", "전체가 밝아진다 (배경이 회색으로)"],
      ["c (대비)", "중간 회색 0.5를 기준으로 늘이고 줄이는 배율", "×1.5", "어두운 곳은 더 어둡게, 밝은 곳은 더 밝게"],
      ["clip", "0~1 범위를 넘지 않게 자른다", "필수", "범위를 벗어나면 이후 계산이 깨진다"],
    ], { x: 0.6, y: 2.65, w: 12.1, colW: [1.9, 4.4, 1.3, 4.5], fontSize: 13, rowH: 0.5 });
    d.code(s, ["def brightness_contrast(img, b=0.0, c=1.0):", "    return np.clip((img - 0.5) * c + 0.5 + b, 0, 1)"], { x: 0.6, y: 5.3, w: 7.2, h: 1.4, size: 14, title: "step1_transforms.py" });
    d.text(s, "실행: 원본 평균 0.287 → 밝기 +0.2 후 0.479 (+0.192: 일부 칸이 clip으로 1을 넘지 못해 0.2보다 약간 작다)", { x: 8.0, y: 5.3, w: 4.7, h: 1.4, size: 13, color: C.muted });
  }

  {
    const s = d.slide("가우시안 노이즈와 색상 변화", { tag: "2교시 · 변환 7" });
    F(d, s, "x′ = clip( x + ε,  0, 1 ),     ε ~ N(0, σ²)", { x: 0.6, y: 1.5, w: 12.1, h: 0.85, size: 26, fill: C.tealL });
    d.code(s, ["def add_noise(img, sigma, rng):", "    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)"], { x: 0.6, y: 2.65, w: 7.2, h: 1.4, size: 13, title: "step1_transforms.py" });
    d.bullets(s, ["σ는 노이즈의 크기: 0.05는 거의 안 보이고, 0.4는 숫자가 사라진다", "센서의 변동(저조도 촬영, 오래된 카메라)을 흉내 낸다", "칸마다 다른 난수를 더한다 (밝기는 모든 칸에 같은 값)"], { x: 7.9, y: 2.65, w: 4.8, h: 2.7, size: 15 });
    d.card(s, { x: 0.6, y: 4.3, w: 7.0, h: 2.4, fill: C.purpleL, title: "색상 변화 (Color Jitter)", tColor: C.purple, size: 15, body: "컬러 사진에서 색조·채도·밝기·대비를 조금씩 흔든다.\n(8×8 흑백 숫자에는 해당 없음 → torchvision의 ColorJitter 참고)\n조심: 바나나는 노랗고, 신호등은 색이 정답이다." });
    d.card(s, { x: 7.9, y: 5.45, w: 4.8, h: 1.25, fill: C.orangeL, title: "크롭 & 리사이즈", tColor: C.orange, size: 14, body: "무작위 영역을 자르고 원래 크기로 확대 = zoom + shift 와 비슷" });
  }

  // 약함 vs 과함
  {
    const s = d.slide("실용적 주의: 과하면 독이 된다", { tag: "2교시 · 주의사항" });
    img(s, "mild_vs_strong", 0.6, 1.4, 5.8);
    d.bullets(s, [
      { t: "약한 증강(초록): 회전 ±10°, 이동 ±1칸, 노이즈 0.05 → 아직 숫자로 읽힌다", b: true },
      { t: "너무 센 증강(빨강): 회전 ±60°, 이동 ±3칸, 노이즈 0.4 → 숫자 구조가 사라진다", b: true },
      "구조가 무너지면 모델은 '진짜 숫자'를 배울 수 없다",
      "그래서 시각적 검토는 필수: 증강된 배치를 항상 눈으로 확인한다",
      "3교시 실험에서 센 증강의 정확도가 37%까지 떨어지는 것을 본다",
    ], { x: 6.7, y: 1.5, w: 6.0, h: 5.2, size: 16 });
    s.addNotes("검증 방법: 증강된 이미지를 16~32장 격자로 그려 보고, 사람이 보고도 같은 클래스라고 말할 수 있는지 확인한다.");
  }

  {
    const s = d.slide("체크리스트: 새 변환을 쓰기 전에", { tag: "2교시 · 정리" });
    const items = [["1", "사람이 봐도 같은 정답인가?", "6↔9, 2↔3 같은 반례를 찾아본다"], ["2", "현장에서 실제로 일어나는 변화인가?", "도로 영상에 90° 회전은 필요 없다"], ["3", "강도(범위)는 적당한가?", "작게 시작해서 키운다 (±10° → ±15°…)"], ["4", "눈으로 확인했는가?", "증강된 배치를 직접 그려 본다"]];
    items.forEach((it, i) => {
      const y = 1.55 + i * 1.2;
      s.addText(it[0], { shape: pres.shapes.OVAL, x: 0.7, y: y + 0.08, w: 0.7, h: 0.7, fill: { color: C.teal }, color: C.white, fontFace: FONT, fontSize: 22, bold: true, align: "center", valign: "middle", margin: 0 });
      d.text(s, it[1], { x: 1.7, y, w: 10.8, h: 0.5, size: 20, bold: true });
      d.text(s, it[2], { x: 1.7, y: y + 0.47, w: 10.8, h: 0.5, size: 16, color: C.muted });
    });
    d.card(s, { x: 0.6, y: 6.1, w: 12.1, h: 0.75, fill: C.tealL, body: "2교시 정리: 변환은 '위치 바꾸기'와 '값 바꾸기'. 모두 NumPy로 구현할 수 있고, 작게·여러 개·눈으로 확인!", size: 15 });
  }
};
