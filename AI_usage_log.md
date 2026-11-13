## 2주 차 AI 활용 내역

### 1. AI 활용 도구
- ChatGPT (OpenAI)

### 2. AI 활용 목적
기존 Wiener Filter의 잡음 제거 성능을 개선하기 위해 새로운 신호 처리 알고리즘을 탐색하고, PESQ 평가 결과를 비교하는 과정에서 AI를 활용하였다.

### 3. AI 활용 내용
- 기존 Decision-Directed Wiener Filter의 동작 원리 분석
- 적응형 잡음 추정(Adaptive Noise Estimation) 방법 탐색 및 Python 코드 작성 지원
- 적응형 Wiener Filter의 파라미터 조정 및 평가 결과 분석
- MMSE-STSA 알고리즘의 핵심 원리와 수식 이해
- MMSE-STSA 알고리즘의 Python 코드 구현 지원
- 기존 Wiener Filter와 MMSE-STSA의 PESQ 성능 비교 코드 작성
- GitHub 제출용 `enhance.py` 제작 및 실행 검증 지원
- README 및 실행 환경 정리

### 4. 알고리즘 실험 과정

**(1) 적응형 Wiener Filter**

기존 Wiener Filter에 시간에 따라 잡음 파워를 갱신하는 방식을 추가하였다.

Train 데이터 30개를 이용한 평가 결과는 다음과 같다.

| 알고리즘 | 평균 PESQ |
| 기존 Wiener Filter | 1.6056 |
| 적응형 Wiener (0.98 / 1.5) | 1.6007 |
| 적응형 Wiener (0.995 / 1.2) | 1.6043 |

적응형 잡음 추정 방법은 파라미터를 조정하면서 성능이 일부 개선되었으나 기존 Wiener Filter보다 높은 점수를 얻지 못하여 최종 알고리즘으로 채택하지 않았다.

**(2) MMSE-STSA 알고리즘**

기존 Wiener Filter의 STFT 및 잡음 추정 구조를 유지하면서 MMSE-STSA 방식으로 필터 이득을 계산하도록 변경하였다.

Train 데이터 30개와 Validation 데이터 164개를 이용하여 성능을 비교하였다.

| 알고리즘 | Train 30개 PESQ | Validation 164개 PESQ |
| 기존 Wiener Filter | 1.6056 | 1.6003 |
| MMSE-STSA | 1.6905 | 1.6349 |

Validation 기준 평균 PESQ가 0.0346 상승하였다.

### 5. 코드 검증

AI의 도움을 받아 작성한 MMSE-STSA 코드를 Google Colab에서 실행하고 결과를 직접 검증하였다.

- 입력 음성과 출력 음성의 샘플링 주파수, 채널 수, 샘플 수 일치 확인
- 기존 코랩 MMSE-STSA 함수와 제출용 코드의 계산 결과 비교
- Validation 전체 164개 음성 처리 성공
- 제출용 코드의 평균 PESQ 1.6349 재현 확인

### 6. AI 활용 결과

AI를 활용하여 기존 Wiener Filter와 적응형 Wiener Filter의 한계를 분석하고, MMSE-STSA 알고리즘을 구현하였다.

제안된 알고리즘과 파라미터의 성능을 Google Colab에서 직접 검증하였으며, 그 결과를 바탕으로 MMSE-STSA를 2주 차 제출 알고리즘으로 선정하였다.

### 7. 참고 문헌

Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator," IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 32, no. 6, pp. 1109–1121, 1984.

DOI: https://doi.org/10.1109/TASSP.1984.1164453
