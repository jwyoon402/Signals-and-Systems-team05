# 음성 잡음 제거 챌린지 — 2주 차

2026 신호 및 시스템 Speech Noise Reduction Challenge 프로젝트입니다.

교수님이 제공한 기본 실습 코드(`noise_reduction_baselines.ipynb`)를 바탕으로, 1주 차에는 Wiener Filter를 사용하였으며 2주 차에는 **MMSE-STSA 알고리즘**을 적용하여 잡음 제거 성능을 개선하였습니다.

프로그램은 잡음이 포함된 음성(Noisy)만 입력받아 처리하며, 원본 음성(Clean)은 잡음 제거 과정에서 사용하지 않습니다.

## 1. 실행 환경

- Python 3.13 환경에서 테스트
- 입력 음성: 16 kHz WAV 파일
- 모노 및 다채널 음성 지원

필요한 라이브러리는 다음 명령어로 설치합니다.

```bash
pip install -r requirements.txt
```

## 2. 실행 방법

### 입력 및 출력 폴더

GitHub에는 `.gitignore` 설정으로 인해 `input/`과 `output/` 폴더가 포함되어 있지 않습니다.

실행 전에 프로젝트 폴더에 `input/` 폴더를 생성하고 잡음이 포함된 WAV 파일을 넣어주세요.

`output/` 폴더는 프로그램 실행 시 자동으로 생성됩니다.

### 기본 실행

프로젝트 폴더에서 다음 명령어를 실행합니다.

```bash
python enhance.py
```

프로그램은 `input/`에 포함된 모든 WAV 파일을 처리하여 `output/`에 저장합니다.

### 입력 및 출력 경로 직접 지정

```bash
python enhance.py --input_dir input --output_dir output
```

출력 파일은 입력과 동일한 파일명, 샘플링 주파수, 채널 수, 샘플 수를 유지합니다. 하위 폴더 구조도 유지됩니다.

## 3. 적용 알고리즘

### 1주 차: Wiener Filter

교수님이 제공한 Wiener Filter를 기반으로 다음 파라미터를 조정하였습니다.

- `wiener_smoothing`: 0.98 → 0.95
- `wiener_gain_floor`: 0.05 → 0.03

### 2주 차: MMSE-STSA

2주 차에는 MMSE-STSA(Minimum Mean Square Error – Short-Time Spectral Amplitude) 알고리즘을 적용하였습니다.

기존 Wiener Filter와 동일한 STFT, 잡음 파워 추정, Decision-Directed 사전 SNR 추정 방식을 유지하면서 필터 이득 계산 부분을 MMSE-STSA 방식으로 변경하였습니다.

주요 설정값은 다음과 같습니다.

- 샘플링 주파수: 16000 Hz
- FFT 크기: 512
- Hop 크기: 128
- 잡음 추정 비율: 0.20
- Decision-Directed 평활화 계수: 0.95
- 최소 이득: 0.03

과도한 잡음 제거와 음성 왜곡을 방지하기 위해 필터 이득을 제한하였습니다.

## 4. Validation 성능 비교

교수님이 제공한 Validation 데이터 164개를 이용해 PESQ-WB 점수를 평가하였습니다.

| 알고리즘 | 평균 PESQ |
| 기존 Wiener Filter | 1.6003 |
| MMSE-STSA | **1.6349** |

- 평균 PESQ 개선량: **+0.0346**
- 기존 Wiener Filter보다 성능이 좋은 음성: 104개 / 164개
- PESQ 평가 실패: 0개

또한 별도로 작성한 제출용 `enhance.py`로 Validation 전체 164개를 처리한 후 PESQ를 재측정하여 평균 1.6349가 재현되는 것을 확인하였습니다.

위 점수는 공개 Validation 데이터 기준이며, 비공개 Hidden Test 결과와는 다를 수 있습니다.

## 5. 파일 구성

```text
Signals-and-Systems-team05/
├── enhance.py          # MMSE-STSA 잡음 제거 코드
├── requirements.txt    # 필요한 라이브러리 및 버전
├── README.md           # 실행 방법 및 성능 결과
├── AI_usage_log.md     # AI 활용 내역
└── .gitignore          # 데이터 및 임시 파일 제외
```

`input/`, `output/`은 로컬 실행 시 사용하는 폴더입니다.

## 6. 코드 출처 및 AI 활용

교수님이 제공한 noise_reduction_baselines.ipynb 파일의 STFT 변환, 잡음 추정 및 음성 복원 방식을 참고하여 MMSE-STSA 알고리즘을 구현하였습니다.

ChatGPT를 활용하여 MMSE-STSA 알고리즘의 원리 이해, Python 코드 작성, 파라미터 설정 및 PESQ 결과 비교를 지원받았습니다.

### 참고 문헌

Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984.

DOI: https://doi.org/10.1109/TASSP.1984.1164453

자세한 AI 활용 내역은 `AI_usage_log.md`를 참고해 주세요.
