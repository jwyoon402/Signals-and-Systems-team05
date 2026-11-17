# 신호 및 시스템 AI 활용 내역서 — 3주 차 제출

**팀명:** 5조 (오버클록)  
**팀원:** 윤종원, 강은비, 이준수  
**제출 내용:** 2주 차 MMSE-LSA를 Multi-resolution 방식으로 개선하고, Automatic Notch Filter와 Time-varying Spectral Gate를 결합하여 음성 품질 향상

## Part 1. AI 활용 개요

- **활용 AI 모델:** ChatGPT (OpenAI)
- **주요 활용 목적:** 기존 MMSE-LSA의 한계 분석, STFT 해상도 및 잡음 추정 파라미터 실험 설계, 협대역 잡음 제거용 자동 Notch Filter 구현, 시간 가변 Spectral Gate 구현, Train/Validation PESQ·SNR 비교 코드 작성, 오류 해결, 독립 실행형 제출 코드 통합, 청취·스펙트로그램 분석 및 문서 작성 보조
- **활용 원칙:** 머신러닝·딥러닝은 사용하지 않고 전통적인 DSP 기법만 사용하였다. 최종 `enhance.py`는 잡음이 포함된 Noisy WAV만으로 동작하며, Clean 음성은 공개 Train/Validation의 성능 **평가와 오차 분석**에만 이용하였다. AI가 제안한 코드는 Google Colab에서 직접 실행해 비교하고, 성능이 낮은 후보는 채택하지 않았다.
- **이전 회차 대비 변화:** 2주 차 최종 제출의 단일 STFT MMSE-LSA(FFT 512, Hop 128)에서 **두 해상도(FFT 1024/1536)의 MMSE-LSA 출력 혼합 → 자동 협대역 잡음 억제 → 시간 가변 스펙트럼 억제**의 세 단계 구조로 변경하였다.

## Part 2. 프롬프트 및 검증 로그

### 사례 1. Multi-resolution MMSE-LSA로 STFT 해상도 변경

**나의 질문(대화 요지)**  
“2주 차 MMSE-LSA보다 PESQ를 높이고 싶다. 신호와 시스템에서 배운 STFT 및 전통적인 DSP 범위에서 분석 해상도와 잡음 추정값을 변경해 비교하자.”

**AI 답변 요약**  
기존 MMSE-LSA의 Decision-Directed 사전 SNR 계산과 로그 스펙트럼 이득 함수를 유지하면서, FFT 크기를 다르게 설정해 동일한 Train 음성에서 비교하도록 제안하였다. 서로 다른 시간·주파수 해상도가 상호 보완적일 수 있으므로 두 FFT 결과를 시간 영역에서 혼합하는 방식도 제시하였다.

**알고리즘 이해 및 선정 이유**  
STFT의 FFT 크기는 시간·주파수 해상도에 영향을 준다. FFT 1024와 1536으로 각각 MMSE-LSA를 적용한 뒤 출력 신호를 1:1로 섞어, 한 가지 해상도만 사용했을 때의 한계를 보완하고자 하였다. 잡음 추정에는 에너지가 작은 프레임 일부를 활용하며, 사전 SNR은 이전 프레임 결과를 반영하는 Decision-Directed 방식으로 계산한다.

**나의 검증 및 수정**

| 알고리즘 | Validation 164개 평균 PESQ-WB | 평균 SNR (dB) |
|---|---:|---:|
| 2주 차 MMSE-LSA (FFT 512) | 1.751625 | 13.418173 |
| FFT 1024 설정 | 1.863572 | 미측정 |
| FFT 1536 설정 | 1.876696 | 미측정 |
| **Multi-resolution (1024 + 1536, 1:1)** | **1.886391** | **13.534871** |

FFT 1024/1536을 혼합한 방식이 Validation 평균 PESQ가 가장 높아 기본 알고리즘으로 선택하였다. `NOISE_FRACTION=0.05`, `NOISE_SCALE=2.20`, `DD_SMOOTHING=0.95`, `GAIN_FLOOR=0.03`을 사용하였다. 독립 실행한 `enhance.py`가 Validation 164개에 대해 **PESQ 1.886391**을 재현하는지 확인하였다.

또한 MMSE-LSA의 고정된 음성 부재 확률, 간소화 IMCRA 계열 잡음 추정, 다중 대역 Spectral Subtraction, 간소화 KLT Subspace, 이득 평활화 후처리 및 적응형 원본 혼합 등 여러 실험을 수행하였다. 일부는 Train에서 소폭 개선되었지만 다른 Train 표본이나 Validation에서는 효과가 유지되지 않아 최종 구조에서 제외하였다. 이는 **시험한 개별 구현의 결과**이지 해당 알고리즘 계열 전반에 대한 결론은 아니다.

### 사례 2. Automatic Notch Filter로 협대역 잔여 성분 억제

**나의 질문(대화 요지)**  
“처리 후에도 ‘쉬이이—’ 하는 잡음과 ‘윙—/삐이—’ 하는 소리가 들린다. 스펙트로그램을 분석해서 기존 MMSE-LSA 뒤에 추가 필터를 적용할 수 있는가?”

**AI 답변 요약**  
고정된 주파수를 무조건 제거하면 음성이 손상될 수 있으므로, 입력 Noisy WAV의 저에너지 구간에서 잡음 스펙트럼을 추정하고 주변 주파수보다 두드러진 협대역 피크를 자동 검출하는 방법을 제시하였다. 검출한 피크에만 `scipy.signal.iirnotch` 필터를 적용하고 기존 출력과 혼합해 감쇠 강도를 조절하도록 하였다.

**알고리즘 이해 및 선정 이유**  
Notch Filter는 선택된 좁은 주파수 대역을 억제한다. 입력의 STFT(FFT 2048)에서 낮은 에너지 프레임 20%를 선택하고, 주파수별 잡음 프로파일이 주변 대역보다 뚜렷하게 큰 위치를 찾았다. 대략 250~7500 Hz 범위에서 최대 두 피크를 선택해 `Q=35`의 IIR Notch Filter를 적용했다. 최종 혼합 비율은 **0.9**로 설정하였다. 검출 피크가 없는 파일은 Notch 처리를 건너뛰도록 했다.

**나의 검증 및 수정**

| 평가 데이터 | 기존 Multi-resolution PESQ | + Auto Notch 0.9 PESQ | 변화 |
|---|---:|---:|---:|
| Train 100개 | 1.932208 | **1.956133** | **+0.023925** |
| 이전 Train과 겹치지 않는 Train 150개 | 1.795900 | **1.815046** | **+0.019146** |
| Validation 164개 | 1.886391 | **1.914650** | **+0.028259** |

Validation에서는 **54개 파일에서 필터가 활성화**되었고, 전체 164개 중 개선 43개·하락 11개·변화 없음 110개였다. 평균 변화의 부트스트랩 95% 구간은 **+0.013820 ~ +0.045536**이었다. 실제 출력 WAV 164개를 저장 후 재평가해 평균 PESQ **1.914650**을 재현하였다.

처음 문제를 발견한 두 파일 중 `speaker016_utt02_seg0009_noisy.wav`는 검출 피크가 없어 Notch가 적용되지 않았다. `speaker007_utt02_seg0005_noisy.wav`는 적용됐지만 PESQ 변화가 작았다. 따라서 이 필터가 **모든 종류의 잔여 잡음을 해결한 것은 아니다**.

### 사례 3. Time-varying Spectral Gate 추가 및 실패한 대안 비교

**나의 질문(대화 요지)**  
“자동 Notch를 추가한 뒤에도 넓은 대역의 잡음이 남는다. Noisy 음성만을 사용하고 학습 모델 없이 시간에 따라 변하는 잡음을 더 억제할 수 있는가?”

**AI 답변 요약**  
입력 Noisy 음성의 STFT 파워에서 시간 방향의 백분위수 기반 잡음 바닥을 추정하고, 시간·주파수별 이득을 계산해 Auto Notch 결과에 부분적으로 적용하는 **실험용 Spectral Gate** 구현을 제시하였다. 서로 다른 혼합 비율 0.35와 0.65를 같은 파일에서 비교하도록 했다.

**알고리즘 이해 및 선정 이유**  
협대역 피크에 집중하는 Notch와 달리 Spectral Gate는 시간·주파수마다 다른 감쇠량을 적용한다. STFT는 FFT 512, Hop 128을 사용하였다. 입력 파워에서 시간 방향 41프레임 구간의 20번째 백분위수를 구하고 Gaussian 평활화를 통해 잡음 바닥을 추정했다. 이를 이용한 이득을 이미 향상된 신호의 스펙트럼에 적용하고, 기존 신호와 `blend=0.35`로 혼합하였다. 본 구현은 **실험용 규칙 기반 DSP 후처리**로, 특정 논문의 알고리즘을 그대로 재현했다고 주장하지 않는다.

**나의 검증 및 수정**

| 평가 데이터 | Auto Notch PESQ | Gate 0.35 추가 PESQ | 변화 |
|---|---:|---:|---:|
| Train 120개 | 2.047045 | **2.060029** | **+0.012984** |
| 별도 Train 150개 | 1.852402 | **1.855975** | **+0.003573** |
| Validation 164개 (노트북 내 계산) | 1.914650 | **1.920661** | **+0.006011** |
| Validation 164개 (저장 WAV 재측정) | 1.914650 | **1.920658** | **+0.006008** |

Gate 0.65는 첫 Train 120개에서는 소폭 높았지만 별도 Train 150개에서는 평균 개선이 약 **+0.000413**에 그쳐, 0.35를 채택하였다. Gate 0.35의 Validation에서는 164개 중 **107개 개선·57개 하락**을 확인하였다. 저장 WAV 기준 평균 PESQ는 **1.920658**이었다.

그 외 Residual Wiener 후처리와 새로운 시간 가변 MMSE-LSA 핵심 잡음 추정 방식을 실험했으나 기존 후보 대비 평균 PESQ 개선을 확인하지 못했다. 특히 시간 가변 MMSE-LSA의 별도 Train 60개 결과는 기존 **1.857570**, `adaptive_weight=0.3` **1.849973**, `0.6` **1.855627**로 나타나 채택하지 않았다.

### 사례 4. 제출 코드 재현성, 다채널 입출력, 청취 및 SNR 직접 검증

**나의 질문(대화 요지)**  
“최종 알고리즘을 하나의 `enhance.py`로 통합하고, 파일 출력 결과가 기존 실험과 같게 나오는지 검증하자. 직접 음성을 듣고 스펙트로그램을 비교하자. 교수님 지침에 맞게 Validation SNR도 직접 측정해 기록하자.”

**AI 답변 요약**  
독립 실행용 `enhance.py` 통합 및 CLI 실행 검사, 저장된 WAV를 다시 읽어 PESQ/SNR을 계산하는 코드, 동일 음량 기준의 청취 코드와 공통 색상 범위의 스펙트로그램 작성 코드, 모노·스테레오·3채널 입출력 검증 코드를 제공하였다. SNR 측정은 `10 log10(Σclean² / Σ(clean−processed)²)`를 파일마다 구한 뒤 164개를 평균하도록 작성하였다.

**나의 직접 검증 및 수정**

1. **제출용 코드 재현:** 기존 제출 코드는 보존하고 최종 `enhance.py`를 별도 폴더에 생성하였다. 별도 Python 프로세스에서 Validation WAV **164/164개**를 처리한 후 저장 WAV 평균 PESQ **1.920658**을 재현했다. 노트북에서 앞서 생성한 기준 WAV와 독립 실행 출력의 **최대 샘플 차이는 0.0000000000**이었다. 출력의 16 kHz, 채널 수, 샘플 수 및 `FLOAT` WAV 저장 형식을 확인했다.
2. **실행 환경 및 다채널:** Google Colab의 Python **3.13.18**에서 확인했다. Validation의 Noisy 모노 파일을 조합해 **모노 3개, 2채널 1개, 3채널 1개**의 시험 WAV를 만든 뒤 독립 실행했다. 모든 파일에서 16,000 Hz, 48,000샘플 및 채널 수가 유지되었고, 다채널 출력과 채널별 독립 처리 출력의 **최대 샘플 차이는 모두 0**이었다. 이는 합성한 다채널 테스트 입력에 대한 **처리 정확성 검증**이지 실제 다채널 음질 개선의 일반적 보장을 의미하지 않는다.
3. **직접 청취:** `speaker015_utt01_seg0001_noisy.wav`는 최종 출력에서 잡음이 확실히 줄고 목소리가 자연스럽게 들렸다. `speaker016_utt02_seg0009_noisy.wav`와 `speaker007_utt02_seg0005_noisy.wav`는 원본 Noisy 대비 잡음 제거 효과가 거의 느껴지지 않았다. 기존 3주 차와 Auto Notch 출력만 비교한 청취에서도 세 사례의 체감 차이는 작았다.
4. **스펙트로그램 및 개별 PESQ:** Clean·Noisy·2주 차·3주 차 출력의 스펙트로그램을 같은 범위로 비교하였다. `speaker016`에서는 약 3 kHz의 반복적인 성분이, `speaker007`에서는 약 2.2초 이후 광대역 성분이 처리 후에도 관찰되었다. 다만 그 성분을 스펙트로그램만으로 전부 잔여 잡음이라고 단정하지 않았다.

| Validation 음성 | Noisy PESQ | 2주 차 PESQ | 3주 차 최종 PESQ |
|---|---:|---:|---:|
| `speaker015_utt01_seg0001_noisy.wav` | 1.119869 | 1.769526 | **1.958461** |
| `speaker016_utt02_seg0009_noisy.wav` | 1.210322 | 1.427355 | **1.431136** |
| `speaker007_utt02_seg0005_noisy.wav` | 1.131854 | **1.174842** | 1.160982 |

**직접 측정한 최종 Validation PESQ·SNR(164개 평균)**

| 알고리즘 | 평균 PESQ-WB | 평균 SNR (dB) |
|---|---:|---:|
| Noisy 입력 | 1.3947 *(2주 차 로그에서 측정한 값)* | 9.634146 |
| 2주 차 MMSE-LSA | 1.751625 | 13.418173 |
| Multi-resolution MMSE-LSA | 1.886391 | **13.534871** |
| Multi-resolution + Auto Notch | 1.914650 | 13.408592 |
| **최종: Multi-resolution + Auto Notch + Gate 0.35** | **1.920658** | **13.186665** |

- **2주 차 대비 최종 PESQ 변화:** **+0.169033**
- **2주 차 대비 최종 SNR 변화:** **−0.231508 dB**
- **Noisy 대비 최종 SNR 변화:** **+3.552519 dB**

SNR은 Clean 신호의 에너지와 출력 오차 에너지를 이용해 계산했으며, Clean은 **평가 코드에만** 사용하였다. 이번 결과에서 최종 모델은 2주 차보다 **PESQ가 개선됐지만 SNR은 감소**하였다. 추가 억제 과정에서 오차 에너지의 특성이 달라졌을 수 있으나, 그 원인을 확정했다고 주장하지 않는다. AI가 제시한 측정 코드를 Google Colab에서 직접 실행하고 CSV 및 `SNR_AI_usage_log_section.md`로 결과를 저장하였다.

### 참고 문헌 및 실제 활용한 구현 자료

**[1] Decision-Directed SNR 및 기존 MMSE-STSA 구조의 기반 문헌**  
Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, **1984**.  
DOI: https://doi.org/10.1109/TASSP.1984.1164453

**[2] MMSE-LSA 이득 함수의 기반 문헌**  
Y. Ephraim and D. Malah, “Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 33, no. 2, pp. 443–445, **1985**.  
DOI: https://doi.org/10.1109/TASSP.1985.1164550  
연구자 공개 원문: https://malah.net.technion.ac.il/files/2017/08/Ephraim_Speech_Enhancement_Minimum_ASSP85.pdf

**[3] Auto Notch 구현 함수 참고 자료**  
SciPy 공식 문서, `scipy.signal.iirnotch`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html  
구현에서 검출한 주파수와 품질 계수 `Q=35`로 2차 IIR Notch 필터를 구성하는 방법에 참고하였다.

**[4] Spectral Gate 구현 함수 참고 자료**  
SciPy 공식 문서, `scipy.ndimage.percentile_filter`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html  
SciPy 공식 문서, `scipy.ndimage.gaussian_filter`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html  
시간 방향 백분위수 기반 잡음 바닥 추정과 평활화의 구현에 참고하였다.

**참고 범위에 관한 주의:** 위 논문은 MMSE 계열 추정 방식의 근거이며, 이번에 작성한 Auto Notch의 피크 검출 규칙과 Spectral Gate 혼합 규칙 전체가 해당 논문에 그대로 수록되어 있다는 의미가 아니다. 문헌의 서지정보와 공개 자료 및 라이브러리 문서는 ChatGPT의 도움으로 확인하였다. 직접 읽지 않은 부분을 직접 검증했다고 주장하지 않는다. 외부의 타인 작성 잡음 제거 코드를 그대로 복사해 사용하지 않았으며, 교수님 제공 노트북·이전 제출 코드·SciPy 라이브러리 함수를 기반으로 구현했다.

## Part 3. 종합 고찰

이번 3주 차 제출에서는 2주 차의 단일 해상도 MMSE-LSA를 Multi-resolution 방식으로 발전시켰다. FFT 1024/1536 출력의 혼합으로 Validation PESQ **1.886391**을 얻었으며, 저에너지 프레임에서 협대역 피크를 검출하는 Automatic Notch Filter를 추가해 **1.914650**, 시간 가변 Spectral Gate를 추가해 최종 저장 WAV 기준 **1.920658**을 기록하였다.

반면 기존 후처리를 더 강하게 만들거나 시간 가변 잡음 추정을 핵심 MMSE-LSA 내부에 추가하는 시도는 기대한 성능 향상을 보이지 않았다. 이를 통해 **Train에서의 개선이 다른 데이터에서도 유지되는지 따로 검증하는 과정**이 필요함을 확인하였다. Validation을 여러 번 참조한 만큼 비공개 Hidden Test에서의 일반화 성능은 확인되지 않았다.

실제 청취에서는 일부 음성의 잡음 감소와 자연스러운 목소리를 확인했지만, 다른 음성에서는 광대역 또는 협대역 잔여 성분과 잡음 제거 한계가 남았다. **평균 PESQ는 2주 차보다 +0.169033 상승한 반면, 평균 SNR은 −0.231508 dB 낮아졌다.** 따라서 두 지표를 함께 기록하고, 향후에는 음성 왜곡과 잔여 잡음의 균형을 개선할 필요가 있다고 판단했다.

최종 제출 코드는 Noisy WAV만으로 동작하며, 별도 프로세스에서 Validation 164개 전부의 결과를 재현하였다. 이 문서는 ChatGPT의 코드·분석 보조와 팀의 직접 실행·청취·검증을 구분해 기록한 것이다.

## 서약 및 조원

본 조는 위 내역이 실제 수행한 실험 및 확인한 자료에 근거하도록 작성하였으며, AI의 제안을 그대로 수용하지 않고 코드를 직접 실행하고 성능을 검증하였음을 확인한다.

**5조 / 팀원:** 윤종원, 강은비, 이준수
