# 음성 잡음 제거 챌린지 — 1주 차

교수님이 제공한 기본 실습 코드(`noise_reduction_baselines.ipynb`)의 **Wiener Filter(위너 필터)**를 바탕으로 만든 1주 차 잡음 제거 프로그램입니다.

이 프로그램은 **잡음이 포함된 음성 파일만** 입력받아 처리하며, 잡음이 없는 원본 음성(`clean`) 파일은 사용하지 않습니다.

## 1. 실행 환경

- Python 3.10 이상 권장
- 입력 음성: **16 kHz, 모노(1채널) WAV 파일**

프로젝트 폴더에서 다음 명령어로 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

## 2. 실행 방법

### 기본 실행

1. 잡음이 포함된 WAV 파일을 `input/` 폴더에 넣습니다.
2. 터미널에서 프로젝트 폴더로 이동한 뒤 아래 명령어를 실행합니다.

```bash
python enhance.py
```

3. 잡음이 제거된 음성 파일은 `output/` 폴더에 저장됩니다.

입력 파일과 **동일한 파일 이름 및 하위 폴더 구조**를 유지합니다. 입력 폴더에는 하나 이상의 WAV 파일이 있어야 합니다.

### 입력·출력 폴더 직접 지정

다른 경로를 사용하려면 다음과 같이 실행할 수 있습니다.

```bash
python enhance.py --input_dir /path/to/input --output_dir /path/to/output
```

> **주의:** 16 kHz 모노 WAV 파일만 지원하며, 조건에 맞지 않는 파일은 오류가 발생합니다. 실행 시 `clean` 음성 파일은 필요하지 않습니다.

## 3. 적용 알고리즘

**STFT(단시간 푸리에 변환)**를 이용한 Wiener Filter 기반 잡음 제거를 사용합니다.

- 창 함수: Hann window
- FFT 크기: `512`
- Hop 크기: `128`
- 잡음 추정: 에너지가 가장 낮은 프레임의 20%를 이용해 잡음 파워 추정
- SNR 추정: 이전 프레임의 결과를 반영하는 decision-directed 방식

1주 차에는 교수님이 제공한 기본 알고리즘의 구조는 유지하고, 아래 파라미터만 변경했습니다.

| 파라미터 | 기본 설정 | 1주 차 수정값 |
|---|---:|---:|
| `wiener_smoothing` | 0.98 | **0.95** |
| `wiener_gain_floor` | 0.05 | **0.03** |

## 4. Validation 성능 비교

교수님이 제공한 `noise_reduction_baselines.ipynb`를 사용하여 **Validation 음성 164개**를 평가한 결과입니다.

| 평가 지표 | 수정 전 Wiener Filter | 수정 후 Wiener Filter |
|---|---:|---:|
| 평균 PESQ-WB | 1.5984 | **1.6003** |
| 평균 SNR (dB) | 13.1196 | **13.6455** |

- 평균 PESQ: **+0.0019** 상승
- 평균 SNR: **+0.5259 dB** 상승

PESQ 개선폭은 작으므로 음질이 크게 향상되었다고 단정할 수는 없습니다.

위 점수는 **실습 노트북에서 계산한 평가 결과**입니다. 별도 저장된 `output/` WAV 파일에 대해 PESQ를 다시 측정한 결과는 아니며, 비공개 테스트셋 점수도 포함되지 않습니다.

## 5. 파일 구성

```text
SpeechNoiseReduction_week1/
├── enhance.py          # 음성 잡음 제거 실행 코드
├── requirements.txt    # 필요한 라이브러리
├── README.md           # 프로젝트 설명 및 실행 방법
├── AI_usage_log.md     # AI 활용 내역서
├── input/              # 처리할 WAV 파일을 넣는 폴더
└── output/             # 처리된 WAV 파일이 저장되는 폴더
```

## 6. 코드 출처 및 AI 활용

- **기본 알고리즘:** 교수님이 제공한 `noise_reduction_baselines.ipynb`의 STFT, 잡음 추정 및 Wiener Filter 구현을 기반으로 함
- **1주 차 변경 사항:** Wiener Filter 파라미터 두 개 조정
- **AI 활용:** ChatGPT를 활용해 평가 결과를 해석하고, 일괄 처리용 `enhance.py`와 제출 문서 작성을 지원받음

자세한 AI 활용 내용은 `AI_usage_log.md`를 참고해 주세요.
