# 음성 잡음 제거 챌린지 — 3주 차 제출

2026 신호 및 시스템 **Speech Noise Reduction Challenge** 5조(오버클록)의 잡음 제거 프로그램입니다.

이번 버전은 2주 차 두 번째 제출의 **MMSE-LSA(Log-MMSE)** 방식을 발전시켜 **Multi-resolution MMSE-LSA**를 적용하고, **Automatic Notch Filter**와 **Time-varying Spectral Gate**를 추가한 결과입니다. 머신러닝·딥러닝 모델은 사용하지 않습니다.

## 1. 실행 환경

- Python 3.13.18 (Google Colab 환경에서 검증)
- 입력: **16 kHz WAV** 파일(모노 또는 다채널)
- 필요 패키지: `numpy`, `scipy`, `soundfile` (`requirements.txt` 참고)

프로젝트 폴더에서 다음 명령어로 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```


**다채널 처리 검증:** Validation의 Noisy 음성을 결합하여 모노·스테레오(2채널)·3채널 WAV를 생성한 후 제출용 `enhance.py`를 독립 실행하였다. 모든 파일에서 16 kHz 샘플레이트, 48,000개 샘플, 입력 채널 수가 유지되었으며, 다채널 출력과 채널별 독립 처리 결과의 최대 샘플 차이는 0이었다.

## 2. 실행 방법

1. 저장소를 내려받은 후 `enhance.py`와 같은 위치에 `input/` 폴더를 만듭니다.
2. `input/` 안에 잡음이 포함된 16 kHz WAV 파일을 넣습니다.
3. 프로젝트 폴더에서 아래 명령어를 실행합니다.

```bash
python enhance.py --input_dir input --output_dir output
```

`output/` 폴더는 실행 시 자동으로 생성되며, 입력 파일마다 **동일한 파일명과 상대 폴더 구조**로 결과 WAV 파일이 저장됩니다. 출력의 샘플링 주파수·채널 수·샘플 수는 입력과 동일하게 유지합니다. 출력 WAV는 `FLOAT` 형식으로 저장합니다.

**잡음 제거 과정에서는 Noisy 음성만 사용합니다.** Clean 음성은 개발 과정의 성능 평가에만 사용하며 `enhance.py`의 입력으로 사용하지 않습니다. 16 kHz가 아닌 입력 파일은 자동 변환하지 않고 오류로 처리합니다.

## 3. 적용 알고리즘

이번 제출에서는 **Multi-resolution MMSE-LSA → Automatic Notch Filter → Time-varying Spectral Gate** 순서로 음성을 처리합니다.

- **Multi-resolution MMSE-LSA:** STFT를 두 해상도에서 각각 수행하고, 저에너지 프레임으로 잡음 파워를 추정한 뒤 Decision-Directed 방식과 MMSE-LSA 이득 함수를 이용합니다. 두 출력 신호를 50:50으로 혼합합니다.
- **Automatic Notch Filter:** Noisy 신호의 저에너지 프레임에서 주파수별 잡음을 추정하고, 주변 대역보다 두드러진 협대역 피크를 자동 검출합니다. 최대 두 개의 검출 주파수에만 Notch Filter를 적용하며, 검출되지 않으면 이 단계를 건너뜁니다.
- **Time-varying Spectral Gate:** Noisy 신호의 시간·주파수별 에너지를 이용해 시간에 따라 달라지는 잡음 바닥을 추정하고, 스펙트럼 이득을 계산해 잔여 잡음을 추가로 억제합니다.

### 최종 설정값

| 항목 | 값 |
|---|---:|
| `SAMPLE_RATE` | 16000 Hz |
| Multi-resolution FFT / Hop | 1024 / 256, 1536 / 384 |
| Multi-resolution 혼합 비율 | 0.5 : 0.5 |
| `NOISE_FRACTION` | 0.05 |
| `NOISE_SCALE` | 2.20 |
| `DD_SMOOTHING` | 0.95 |
| `GAIN_FLOOR` | 0.03 |
| Automatic Notch 최대 피크 | 2개 |
| Automatic Notch `Q` | 35 |
| Automatic Notch 혼합 비율 | 0.9 |
| Spectral Gate FFT / Hop | 512 / 128 |
| Spectral Gate 혼합 비율 | 0.35 |

모든 단계는 전통적인 디지털 신호처리와 수치 계산으로 구현했습니다.

## 4. Validation 성능 비교

제공된 **Validation 164개 파일**에서 16 kHz PESQ-WB 평균을 비교했습니다.

| 알고리즘 | Validation 평균 PESQ-WB |
|---|---:|
| 2주 차 두 번째 제출 MMSE-LSA | 1.751625 |
| 3주 차 Multi-resolution MMSE-LSA | 1.886391 |
| Multi-resolution + Automatic Notch Filter | 1.914650 |
| **이번 제출: Multi-resolution + Automatic Notch + Spectral Gate** | **1.920658** |

- **2주 차 두 번째 제출 대비 개선:** 약 +0.1690 PESQ (1.751625 → 1.920658).
- **Automatic Notch만 적용한 버전 대비 개선:** 약 +0.0060 PESQ (1.914650 → 1.920658).
- **독립 실행 검증:** 제출 후보의 `enhance.py`를 별도 Python 프로세스로 실행하여 Validation WAV 164개를 모두 생성했습니다. 샘플링 주파수, 채널 수, 샘플 수 및 저장 형식 검사에 통과했습니다.
- **재현성:** 독립 실행으로 생성한 WAV를 다시 읽어 평균 PESQ-WB **1.920658**을 확인했으며, 앞서 검증한 WAV와의 최대 샘플 차이는 **0**이었습니다.
### 개별 음성 PESQ 및 청취 비교

전체 평균뿐 아니라 Validation의 음성 3개를 직접 청취하고, **Clean / Noisy / 2주 차 MMSE-LSA / 3주 차 최종 출력**의 스펙트로그램을 동일한 색상 범위로 비교했습니다. 아래 값은 **각 파일의 PESQ-WB**이며 전체 평균과는 구분됩니다.

| 음성 파일 | Noisy PESQ | 2주 차 PESQ | 3주 차 최종 PESQ |
|---|---:|---:|---:|
| `speaker015_utt01_seg0001_noisy.wav` | 1.119869 | 1.769526 | **1.958461** |
| `speaker016_utt02_seg0009_noisy.wav` | 1.210322 | 1.427355 | **1.431136** |
| `speaker007_utt02_seg0005_noisy.wav` | 1.131854 | **1.174842** | 1.160982 |

- **speaker015:** 3주 차 출력에서 배경에 넓게 분포한 잡음 성분이 줄었으며, 주요 저주파 음성 구조가 유지되는 모습을 확인했습니다. 직접 청취에서도 **잡음이 확실히 줄고 목소리가 자연스럽게 들렸습니다.**
- **speaker016:** 약 3 kHz 부근의 반복적인 주파수 성분이 처리 후에도 관찰됐습니다. 3주 차 PESQ가 2주 차보다 소폭 높았지만, **직접 청취했을 때 잡음 제거 효과는 거의 느껴지지 않았습니다.** 해당 성분이 모두 잡음인지 여부는 스펙트로그램만으로 단정하기 어렵습니다.
- **speaker007:** 약 2.2초 이후 넓은 주파수 대역의 성분이 처리 후에도 남아 있었습니다. 3주 차 PESQ는 2주 차보다 소폭 낮았고, **직접 청취 시에도 잡음 제거 효과가 거의 느껴지지 않았습니다.**

이 비교를 통해 3주 차 알고리즘이 **전체 Validation 평균 PESQ를 개선**했으나, 파일별 잡음 특성에 따라 성능과 청취상 효과가 다를 수 있음을 확인했습니다. **데이터 재배포 범위를 고려해 스펙트로그램 PNG와 원본 WAV는 이 공개 README에 첨부하지 않았습니다.**

위 점수는 **공개 Validation 데이터**에서의 결과이며, 비공개 Hidden Test의 성능을 보장하지 않습니다.

## 5. 저장소 파일

```text
Signals-and-Systems-team05/
├── enhance.py          # Multi-resolution + Notch + Gate 실행 코드
├── requirements.txt    # 필요한 라이브러리
└── README.md           # 실행 방법과 성능 결과
```

AI 활용 내역서는 교수님이 지정한 양식과 제출 절차에 따라 별도로 제출합니다.

## 6. 개발 기반 및 참고 문헌

교수님 제공 `noise_reduction_baselines.ipynb`의 STFT, 잡음 추정 및 평가 구조와 기존 2주 차 **MMSE-LSA 제출 코드**를 기반으로 개선했습니다. Train 데이터에서 해상도, 잡음 추정 파라미터, 후처리 후보를 비교하고 Validation 데이터로 최종 결과를 확인했습니다.

- Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984. https://doi.org/10.1109/TASSP.1984.1164453
- Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 33, no. 2, pp. 443–445, 1985. https://doi.org/10.1109/TASSP.1985.1164550

ChatGPT는 코드 작성 보조, 실험 결과 해석 및 문서 초안 작성에 활용했으며, 실제 코드 실행·음성 청취·성능 검증은 팀에서 수행했습니다. AI 활용 내역은 별도 제출 문서에 기록합니다.
