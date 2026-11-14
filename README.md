# 음성 잡음 제거 챌린지 — 2주 차 두 번째 제출

2026 신호 및 시스템 **Speech Noise Reduction Challenge** 5조(오버클록)의 잡음 제거 프로그램입니다.

이번 버전은 앞서 제출한 **MMSE-STSA**를 **MMSE-LSA(Log-MMSE)** 방식으로 개선하고, 잡음 파워 추정 관련 파라미터를 조정한 결과입니다. 머신러닝·딥러닝 모델은 사용하지 않습니다.

## 1. 실행 환경

- Python 3.13 환경에서 검증
- 입력: **16 kHz WAV** 파일(모노 또는 다채널)
- 필요 패키지: `numpy`, `scipy`, `soundfile` (`requirements.txt` 참고)

프로젝트 폴더에서 다음 명령어로 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## 2. 실행 방법

1. 저장소를 내려받은 후, `enhance.py`와 같은 위치에 `input/` 폴더를 만듭니다.
2. `input/` 안에 잡음이 포함된 16 kHz WAV 파일을 넣습니다.
3. 프로젝트 폴더에서 아래 명령어를 실행합니다.

```bash
python enhance.py --input_dir input --output_dir output
```

경로를 생략한 `python enhance.py` 명령도 같은 기본 폴더를 사용합니다.

`output/` 폴더는 실행 시 자동으로 생성되며, 입력 파일마다 같은 파일명과 상대 폴더 구조로 결과 WAV 파일 하나가 저장됩니다. 출력의 샘플링 주파수·채널 수·샘플 수는 입력과 동일하게 유지됩니다.

**실제 잡음 제거는 Noisy 음성만으로 수행합니다.** Clean 음성은 개발 과정의 성능 평가에만 사용하며 `enhance.py`의 입력으로 사용하지 않습니다. `input/`과 `output/`은 GitHub에서 제외될 수 있으므로 실행 시 위 안내에 따라 준비합니다.

## 3. 적용 알고리즘

- 입력 신호를 **STFT**로 변환
- 낮은 에너지의 완전한 STFT 프레임 일부를 선택해 주파수별 잡음 파워 추정
- **Decision-Directed** 방식으로 사전 SNR 추정
- **MMSE-LSA** 이득 함수로 주파수 성분을 조절
- 입력 신호의 위상을 유지하면서 **ISTFT**로 복원

기존 MMSE-STSA와 비교하여 음성 스펙트럼의 **로그 진폭 추정**을 사용하는 것이 핵심 차이입니다. 현재 코드는 통계적 신호 처리와 수치 필터링으로만 동작합니다.

### 최종 설정값

| 파라미터 | 값 |
|---|---:|
| `SAMPLE_RATE` | 16000 Hz |
| `N_FFT` | 512 |
| `HOP` | 128 |
| `NOISE_FRACTION` | 0.10 |
| `NOISE_SCALE` | 1.80 |
| `DD_SMOOTHING` | 0.95 |
| `GAIN_FLOOR` | 0.03 |

`NOISE_FRACTION`은 잡음 파워 추정에 활용하는 저에너지 프레임의 비율이고, `NOISE_SCALE`은 추정한 잡음 파워에 곱하는 보정 계수입니다.

## 4. Validation 성능 비교

제공된 **Validation 164개 파일**에서 16 kHz PESQ-WB 평균을 비교했습니다.

| 알고리즘 | Validation 평균 PESQ-WB |
|---|---:|
| 기존 Wiener Filter | 1.6003 |
| 2주 차 첫 제출 MMSE-STSA | 1.6349 |
| MMSE-LSA (잡음 비율 0.20, 보정 계수 1.00) | 1.6674 |
| MMSE-LSA (잡음 비율 0.10, 보정 계수 1.00) | 1.6976 |
| **이번 제출: MMSE-LSA (잡음 비율 0.10, 보정 계수 1.80)** | **1.7516** |

- **직전 제출 대비 개선:** +0.1167 PESQ (1.6349 → 1.7516)
- **Validation 평균 SNR:** Noisy 9.6341 dB, Wiener Filter 13.6455 dB, MMSE-STSA 13.2891 dB, 최종 MMSE-LSA **13.4182 dB** (직전 제출 대비 +0.1291 dB).
- **청취 및 시각화:** Validation 샘플 1개에서 MMSE-LSA는 MMSE-STSA보다 잡음이 훨씬 적게 들렸지만 약간 먹먹하게 느껴졌고, 스펙트로그램에서도 배경 잡음의 강한 억제가 관찰되었습니다. 한 파일의 주관적 관찰을 전체 음성에 일반화하지 않습니다.

- 독립 실행한 **제출용 `enhance.py`로 164개 전부 처리하고 평균 1.7516을 재현**했습니다.
- 제출용 WAV와 노트북 계산 결과의 최대 샘플 오차는 약 `2.98 × 10^-8`이었으며, 샘플링 주파수·채널 수·샘플 수 검사가 통과했습니다.

위 수치는 **공개 Validation 데이터**에서의 결과입니다. 비공개 Hidden Test에서 동일한 점수를 보장하지는 않습니다.

## 5. 저장소 파일

```text
Signals-and-Systems-team05/
├── enhance.py          # MMSE-LSA 잡음 제거 실행 코드
├── requirements.txt    # 패키지 및 버전
├── README.md           # 실행 방법과 실험 결과
├── AI_usage_log.md     # AI 활용 내역
└── .gitignore          # 입력·출력 데이터 등 제외 설정
```

## 6. 개발 기반 및 참고 문헌

교수님 제공 `noise_reduction_baselines.ipynb`의 STFT, 잡음 추정 및 평가 구조를 참고하였고, 기존 MMSE-STSA 제출 코드에서 이득 계산식을 MMSE-LSA 방식으로 변경했습니다. 이후 파라미터를 Train 데이터에서 비교하고 Validation 데이터로 성능을 확인했습니다.

- Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984. https://doi.org/10.1109/TASSP.1984.1164453
- Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 33, no. 2, pp. 443–445, 1985. https://doi.org/10.1109/TASSP.1985.1164550

코드 작성 및 실험 검토에 사용한 AI 관련 사항은 `AI_usage_log.md`와 회차별 제출 내역서에 기록합니다.
