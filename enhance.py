"""1주차 음성 노이즈 감소 챌린지 제출물.

제공된 Wiener filter를 바탕으로 파라미터를 조정하면:
    wiener_smoothing: 0.98 -> 0.95
    wiener_gain_floor: 0.05 -> 0.03

입력/에서 WAV 파일을 읽고, 해당 향상된 WAV 파일을 출력/에 씁니다.
구현은 깨끗한 참조 파일이 아닌 각 노이즈 입력 파일에만 접근합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128
NOISE_FRACTION = 0.20
WIENER_SMOOTHING = 0.95
WIENER_GAIN_FLOOR = 0.03


def estimate_noise_power(stft_matrix: np.ndarray, times: np.ndarray,
                         signal_length: int) -> np.ndarray:
    """Estimate stationary noise using lowest-energy complete STFT frames."""
    power = np.abs(stft_matrix) ** 2
    centers = np.rint(times * SAMPLE_RATE).astype(int)
    full = np.flatnonzero(
        (centers >= N_FFT // 2) &
        (centers + N_FFT // 2 <= signal_length)
    )
    if len(full) == 0:
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    noise = power[:, selected].mean(axis=1)
    return np.maximum(uniform_filter1d(noise, size=3, mode="nearest"), 1e-18)


def wiener_filter(noisy: np.ndarray) -> np.ndarray:
    noisy = np.asarray(noisy, dtype=np.float64)
    if noisy.ndim != 1 or len(noisy) == 0 or not np.isfinite(noisy).all():
        raise ValueError("Expected a nonempty finite mono audio signal")
    original_length = len(noisy)
    if original_length < N_FFT:
        noisy = np.pad(noisy, (0, N_FFT - original_length))

    _, times, z = signal.stft(
        noisy, fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT,
        boundary="zeros", padded=True,
    )
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(noisy))
    posterior_snr = power / noise[:, None]

    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (
                WIENER_SMOOTHING * previous_clean_power / noise
                + (1.0 - WIENER_SMOOTHING) * instantaneous
            )
        gain = np.clip(prior_snr / (1.0 + prior_snr), WIENER_GAIN_FLOOR, 1.0)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2

    _, output = signal.istft(
        enhanced, fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT,
        input_onesided=True, boundary=True,
    )
    if len(output) < original_length:
        raise RuntimeError("Inverse STFT produced too few samples")
    return np.asarray(output[:original_length], dtype=np.float64)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    audio_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not audio_files:
        raise FileNotFoundError(f"No .wav files in: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(audio_files, start=1):
        audio, sr = sf.read(source, dtype="float64")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{source}: expected {SAMPLE_RATE} Hz, got {sr} Hz")
        if audio.ndim != 1:
            raise ValueError(f"{source}: expected mono WAV")
        enhanced = wiener_filter(audio)
        target = output_dir / source.relative_to(input_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Original notebook's export uses floating-point WAV, without normalizing.
        sf.write(target, enhanced, SAMPLE_RATE, subtype="FLOAT")
        print(f"[{idx}/{len(audio_files)}] {source.name} -> {target}")
    print(f"Done: {len(audio_files)} files, output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 Wiener-filter denoiser")
    parser.add_argument("--input_dir", type=Path, default=Path("input"))
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
