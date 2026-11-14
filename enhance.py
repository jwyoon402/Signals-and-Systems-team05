"""2주 차 두 번째 제출 후보: MMSE-LSA 기반 음성 잡음 제거 프로그램.

기존 제출용 WAV 입출력 구조를 유지하고, MMSE-STSA 대신 MMSE-LSA
(Log-MMSE) 이득을 적용합니다. 잡음 제거 과정에서 clean 음성을 사용하지 않습니다.

실행 예시:
    python enhance.py --input_dir input --output_dir output

입력: 16 kHz WAV 파일 (모노 또는 여러 채널)
출력: 입력과 같은 파일명·샘플링 주파수·채널 수·샘플 수의 WAV 파일
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

# Train에서 후보를 찾고 Validation에서 비교한 MMSE-LSA 실험 설정.
SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128
NOISE_FRACTION = 0.10
NOISE_SCALE = 1.80
DD_SMOOTHING = 0.95
GAIN_FLOOR = 0.03
EPS = 1e-18


def estimate_noise_power(z: np.ndarray, times: np.ndarray,
                         signal_length: int) -> np.ndarray:
    """에너지가 낮은 완전한 STFT 프레임을 이용하여 잡음 파워를 추정합니다."""
    power = np.abs(z) ** 2

    # 경계의 제로 패딩이 잡음 구간 선택에 영향을 주지 않도록 제외합니다.
    centers = np.rint(times * SAMPLE_RATE).astype(int)
    full = np.flatnonzero(
        (centers >= N_FFT // 2)
        & (centers + N_FFT // 2 <= signal_length)
    )
    if len(full) == 0:
        full = np.arange(power.shape[1])

    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    chosen = full[np.argsort(energy, kind="stable")[:count]]
    noise = power[:, chosen].mean(axis=1)
    return np.maximum(uniform_filter1d(noise, size=3, mode="nearest"), EPS)


def mmse_lsa_filter(noisy: np.ndarray) -> np.ndarray:
    """모노 음성에 MMSE-LSA 이득을 적용하고 원래 샘플 수로 복원합니다."""
    x = np.asarray(noisy, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("유효한 1차원 모노 음성이 필요합니다.")
    original_length = len(x)

    # 짧은 음성도 512-point FFT로 처리합니다.
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))

    _, times, z = signal.stft(
        x, fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT,
        boundary="zeros", padded=True,
    )
    power = np.abs(z) ** 2

    # 이전 방식의 잡음 파워 추정에 보정 계수를 적용합니다.
    noise = estimate_noise_power(z, times, len(x))
    noise = np.maximum(noise * NOISE_SCALE, EPS)

    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)

    for frame in range(z.shape[1]):
        # 사후 SNR 및 Decision-Directed 사전 SNR.
        gamma = np.clip(power[:, frame] / noise, 1e-8, 1e6)
        instantaneous = np.maximum(gamma - 1.0, 0.0)
        if frame == 0:
            xi = instantaneous
        else:
            xi = (
                DD_SMOOTHING * previous_clean_power / noise
                + (1.0 - DD_SMOOTHING) * instantaneous
            )
        xi = np.clip(xi, 0.0, 1e6)

        # Ephraim–Malah MMSE-LSA 이득: 지수적분 E1(v)를 사용합니다.
        v = gamma * xi / (1.0 + xi)
        v = np.maximum(v, 1e-8)
        gain = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))

        # 과도한 감쇠·증폭 방지를 위한 실험용 이득 제한.
        gain = np.clip(gain, GAIN_FLOOR, 1.0)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2

    _, output = signal.istft(
        enhanced, fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT,
        input_onesided=True, boundary=True,
    )
    if len(output) < original_length:
        raise RuntimeError("복원된 음성의 샘플 수가 부족합니다.")
    return np.asarray(output[:original_length], dtype=np.float64)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """input 폴더의 WAV 파일을 처리하고 상대 경로·파일명을 유지합니다."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾지 못했습니다: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not files:
        raise FileNotFoundError(f"입력 폴더에 WAV 파일이 없습니다: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(files, start=1):
        audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{source}: 입력은 16 kHz WAV여야 합니다 (현재 {sample_rate} Hz)."
            )

        # 각 채널을 독립적으로 처리하여 입력 형태를 유지합니다.
        channels = [mmse_lsa_filter(audio[:, ch]) for ch in range(audio.shape[1])]
        result = np.column_stack(channels)
        if result.shape != audio.shape:
            raise RuntimeError(f"음성 형태가 달라졌습니다: {source}")

        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 부동소수점 WAV 사용: 디스크 저장 후 양자화 오차 최소화.
        sf.write(
            destination,
            result[:, 0] if result.shape[1] == 1 else result,
            sample_rate,
            subtype="FLOAT",
        )
        print(f"[{idx}/{len(files)}] {source.name} -> {destination}")
    print(f"처리 완료: {len(files)}개, 출력 폴더: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMSE-LSA 음성 잡음 제거 (2주 차 두 번째 제출 후보)"
    )
    parser.add_argument("--input_dir", type=Path, default=Path("input"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
