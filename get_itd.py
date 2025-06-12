import numpy as np
from numpy.typing import NDArray

def calculate_itd(hrir: NDArray[np.float32], fs: float, min_freq: int, max_freq: int) -> float:
    hfft = np.fft.rfft(hrir, axis=0)
    pleft = np.unwrap(np.angle(hfft[:, 0]))
    pright = np.unwrap(np.angle(hfft[:, 1]))
    deltap = pleft - pright
    freqs = np.fft.rfftfreq(hrir.shape[0], d=1 / fs)
    mask = (freqs > min_freq) & (freqs < max_freq)
    slope, _ = np.polyfit(freqs[mask], deltap[mask], 1)
    itd_phase = slope / (2 * np.pi)
    
    correlation = np.correlate(hrir[:, 0], hrir[:, 1], mode="full")
    max_index = np.argmax(np.abs(correlation))
    delay_samples = max_index - (len(correlation) // 2)
    itd_corr = delay_samples / fs
    
    if abs(itd_phase - itd_corr) <= 0.002: 
        return itd_phase
    return 0.7 * itd_phase + 0.3 * itd_corr

def remove_itd(hrir_fft: NDArray[np.complex64], n: int, itd: float, fs: float) -> NDArray[np.float32]:
    left = hrir_fft[:, 0]
    right = hrir_fft[:, 1]
    
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    
    if itd > 0:
        right = right * np.exp(2j * np.pi * freqs * itd)
    else:
        left = left * np.exp(2j * np.pi * freqs * abs(itd))
    
    return np.column_stack((left, right)).astype(np.complex64)