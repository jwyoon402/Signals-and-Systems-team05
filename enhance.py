"""3주 차 최종 후보: 다중 해상도 MMSE-LSA 음성 잡음 제거.

16 kHz WAV 파일들을 읽어 1024/1536-point MMSE-LSA 결과를 절반씩 혼합합니다.
잡음 제거 추론에 clean 음성을 사용하지 않습니다.

실행:
    python enhance.py --input_dir input --output_dir output

입력과 출력의 상대 경로, 파일명, 샘플링 주파수, 채널 수 및 샘플 수를 유지합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

SAMPLE_RATE = 16000
FFT_SETTINGS = ((1024, 256), (1536, 384))
MIX_ALPHA = 0.50   # FFT 1536의 출력 비율. FFT 1024의 비율은 1-alpha.
NOISE_FRACTION = 0.05
NOISE_SCALE = 2.20
DD_SMOOTHING = 0.95
GAIN_FLOOR = 0.03
EPS = 1e-18


def estimate_noise_power(
    z: np.ndarray, times: np.ndarray, signal_length: int, n_fft: int
) -> np.ndarray:
    """프레임 에너지가 낮은 완전한 STFT 프레임에서 잡음 파워 추정."""
    power = np.abs(z) ** 2
    centers = np.rint(times * SAMPLE_RATE).astype(int)
    full = np.flatnonzero(
        (centers >= n_fft // 2)
        & (centers + n_fft // 2 <= signal_length)
    )
    if len(full) == 0:
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    chosen = full[np.argsort(energy, kind="stable")[:count]]
    noise = power[:, chosen].mean(axis=1)
    return np.maximum(uniform_filter1d(noise, size=3, mode="nearest"), EPS)


def mmse_lsa_filter(noisy: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """기존 2주 차 MMSE-LSA 공식을 STFT 길이를 지정해 수행."""
    x = np.asarray(noisy, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("유효한 1차원 모노 음성이 필요합니다.")
    original_length = len(x)

    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    _, times, z = signal.stft(
        x, fs=SAMPLE_RATE, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, nfft=n_fft,
        boundary="zeros", padded=True,
    )
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(x), n_fft)
    noise = np.maximum(noise * NOISE_SCALE, EPS)

    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)
    for frame in range(z.shape[1]):
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
        v = gamma * xi / (1.0 + xi)
        v = np.maximum(v, 1e-8)
        gain = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))
        gain = np.clip(gain, GAIN_FLOOR, 1.0)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2

    _, output = signal.istft(
        enhanced, fs=SAMPLE_RATE, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, nfft=n_fft,
        input_onesided=True, boundary=True,
    )
    if len(output) < original_length:
        raise RuntimeError("복원된 음성의 샘플 수가 부족합니다.")
    return np.asarray(output[:original_length], dtype=np.float64)


def multires_lsa_filter(noisy: np.ndarray) -> np.ndarray:
    """Validation 기준 mix_50: 1024 및 1536 FFT 결과를 50:50 혼합."""
    (n1, h1), (n2, h2) = FFT_SETTINGS
    y1024 = mmse_lsa_filter(noisy, n_fft=n1, hop=h1)
    y1536 = mmse_lsa_filter(noisy, n_fft=n2, hop=h2)
    return (1.0 - MIX_ALPHA) * y1024 + MIX_ALPHA * y1536


def process_folder(input_dir: Path, output_dir: Path) -> None:
    """입력 폴더의 모든 WAV를 처리하고 상대 경로와 음성 형태 유지."""
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
        channels = [multires_lsa_filter(audio[:, ch]) for ch in range(audio.shape[1])]
        result = np.column_stack(channels)
        if result.shape != audio.shape:
            raise RuntimeError(f"음성 형태가 달라졌습니다: {source}")
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
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
        description="다중 해상도 MMSE-LSA 잡음 제거 (3주 차 후보)"
    )
    parser.add_argument("--input_dir", type=Path, default=Path("input"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)






# =========================================================
# Week 3 Final Candidate
# Multi-resolution MMSE-LSA + Auto Notch + Spectral Gate
# =========================================================

import argparse as _argparse
from pathlib import Path as _Path

import numpy as _np
import soundfile as _sf

from scipy import signal as _signal
from scipy.ndimage import (
    median_filter as _median_filter,
    percentile_filter as _percentile_filter,
    gaussian_filter as _gaussian_filter,
)


# Validation 저장 WAV와 동일한 형식
OUTPUT_SUBTYPE = 'FLOAT'


def automatic_notch_filter(noisy, enhanced, mix=0.9):
    fs = 16000

    freqs, times, Z = _signal.stft(
        noisy,
        fs=fs,
        window="hann",
        nperseg=2048,
        noverlap=1536
    )

    power = _np.abs(Z) ** 2

    frame_energy = _np.mean(power, axis=0)

    n_quiet = max(
        4, int(len(frame_energy) * 0.2)
    )

    quiet_idx = _np.argsort(
        frame_energy
    )[:n_quiet]

    noise_profile = _np.median(
        power[:, quiet_idx],
        axis=1
    )

    background = _median_filter(
        noise_profile,
        size=31,
        mode="nearest"
    )

    peak_db = 10 * _np.log10(
        (noise_profile + 1e-14) /
        (background + 1e-14)
    )

    peaks, properties = _signal.find_peaks(
        peak_db,
        height=10.0,
        prominence=5.0,
        distance=15
    )

    peaks = [
        p for p in peaks
        if 250 <= freqs[p] <= 7500
    ]

    peaks = sorted(
        peaks,
        key=lambda p: peak_db[p],
        reverse=True
    )[:2]

    if not peaks:
        return _np.asarray(enhanced).copy()

    filtered = _np.asarray(
        enhanced, dtype=_np.float64
    ).copy()

    for p in peaks:
        frequency = float(freqs[p])

        b, a = _signal.iirnotch(
            frequency,
            Q=35,
            fs=fs
        )

        sos = _signal.tf2sos(b, a)

        filtered = _signal.sosfiltfilt(
            sos, filtered
        )

    return (
        (1.0 - mix) * enhanced +
        mix * filtered
    )


def time_varying_spectral_gate(
    noisy, enhanced, blend=0.35
):
    noisy = _np.asarray(
        noisy, dtype=_np.float64
    )
    enhanced = _np.asarray(
        enhanced, dtype=_np.float64
    )

    fs = 16000
    fft_size = 512
    hop = 128

    stft_args = dict(
        fs=fs,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size-hop,
        nfft=fft_size,
        boundary="zeros",
        padded=True
    )

    _, _, X = _signal.stft(
        noisy, **stft_args
    )

    _, _, Y = _signal.stft(
        enhanced, **stft_args
    )

    power = _np.abs(X) ** 2

    noise_power = _percentile_filter(
        power,
        percentile=20,
        size=(1, 41),
        mode="nearest"
    )

    noise_power = _gaussian_filter(
        noise_power,
        sigma=(1.0, 1.5)
    )

    smooth_power = _gaussian_filter(
        power,
        sigma=(1.0, 1.0)
    )

    ratio = noise_power / (
        smooth_power + 1e-12
    )

    gain = _np.sqrt(
        _np.clip(
            1.0 - 0.8 * ratio,
            0.16,
            1.0
        )
    )

    _, filtered = _signal.istft(
        Y * gain,
        fs=fs,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size-hop,
        nfft=fft_size,
        input_onesided=True,
        boundary=True
    )

    if len(filtered) < len(noisy):
        filtered = _np.pad(
            filtered,
            (0, len(noisy)-len(filtered))
        )

    filtered = filtered[:len(noisy)]

    return (
        (1.0 - blend) * enhanced +
        blend * filtered
    )


def enhance_final_mono(noisy):
    noisy = _np.asarray(
        noisy, dtype=_np.float64
    )

    # 1. 기존 검증된 Multi-resolution MMSE-LSA
    baseline = multires_lsa_filter(noisy)

    # 2. Auto Notch 0.9
    notch = automatic_notch_filter(
        noisy, baseline, mix=0.9
    )

    # 3. Time-varying Gate 0.35
    final = time_varying_spectral_gate(
        noisy, notch, blend=0.35
    )

    assert final.shape == noisy.shape
    assert _np.isfinite(final).all()

    return final


def enhance_final(audio):
    if audio.ndim == 1:
        return enhance_final_mono(audio)

    # 다채널 입력은 채널별 독립 처리
    channels = [
        enhance_final_mono(audio[:, ch])
        for ch in range(audio.shape[1])
    ]

    return _np.column_stack(channels)


def process_wav(input_path, output_path):
    audio, sr = _sf.read(
        input_path, dtype="float64"
    )

    if sr != 16000:
        raise ValueError(
            f"16kHz 입력이 필요합니다: {input_path}"
        )

    enhanced = enhance_final(audio)

    assert enhanced.shape == audio.shape

    output_path = _Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    _sf.write(
        output_path,
        enhanced,
        sr,
        subtype=OUTPUT_SUBTYPE
    )


def main():
    parser = _argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", "--input-dir", "--input",
        dest="input_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir", "--output-dir", "--output",
        dest="output_dir",
        required=True
    )

    args = parser.parse_args()

    source = _Path(args.input_dir)
    destination = _Path(args.output_dir)

    if source.is_file():
        if source.suffix.lower() != ".wav":
            raise ValueError("WAV 파일을 입력해 주세요.")

        output_file = (
            destination
            if destination.suffix.lower() == ".wav"
            else destination / source.name
        )

        process_wav(source, output_file)
        print("Processed:", output_file)

    else:
        if not source.is_dir():
            raise FileNotFoundError(source)

        files = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() == ".wav"
        )

        if not files:
            raise ValueError("입력 WAV가 없습니다.")

        for i, input_file in enumerate(files, 1):
            relative = input_file.relative_to(source)

            output_file = destination / relative

            process_wav(
                input_file,
                output_file
            )

            if i % 20 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}")


if __name__ == "__main__":
    main()
