import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

np.random.seed(0)
num_measurments = 200
true_signal = np.sin(np.linspace(0, 4 * np.pi, num_measurments))
measured_range = true_signal + np.random.normal(0, 0.2, num_measurments)

def apply_kalman_filter(R, c):
    x = np.zeros(num_measurments)
    p = np.zeros(num_measurments)
    x_mns = np.zeros(num_measurments)
    p_mns = np.zeros(num_measurments)
    k_gain = np.zeros(num_measurments)

    x[0] = 0
    p[0] = 1

    for i in range(1, num_measurments):
        x_mns[i] = x[i - 1]
        p_mns[i] = p[i - 1] + c

        k_gain[i] = p_mns[i] / (p_mns[i] + R)
        x[i] = x_mns[i] + k_gain[i] * (measured_range[i] - x_mns[i])
        p[i] = (1 - k_gain[i]) * p_mns[i]Q

    return x, p

def apply_fourier_denoise(data, cutoff):
    denoised = []

    for i in range(1, len(data) + 1):
        window_data = data[:i]
        t = len(window_data)

        fft_data = np.fft.fft(window_data)
        freq = np.fft.fftfreq(t, d=1)

        fft_filtered = fft_data.copy()
        fft_filtered[np.abs(freq) > cutoff] = 0

        smoothed = np.fft.ifft(fft_filtered).real
        denoised.append(smoothed[-1])

    return denoised

def compare(R=0.01, c=0.001, cutoff=0.05):
    fft_data = apply_fourier_denoise(measured_range, cutoff)
    x, p = apply_kalman_filter(R, c)
    shift = 1

    plt.figure(figsize=(10, 5))
    plt.plot(measured_range[shift:], label="Measured Signal", alpha=0.5)
    plt.plot(true_signal[shift:], label="True Signal", linestyle="--", color="black")
    plt.plot(x[shift:], label="Kalman Filter", linewidth=2)
    plt.plot(fft_data[shift:], label="FFT Filter", color="red", linewidth=2)
    plt.title(f"Kalman vs FFT Filtering\nR={R:.4f}, c={c:.4f}, cutoff={cutoff:.3f}")
    plt.xlabel("Time step")
    plt.ylabel("Signal value")
    plt.legend()
    plt.grid(True)
    plt.show()

interact(
    compare,
    R=widgets.FloatLogSlider(value=0.01, base=10, min=-4, max=-1, step=0.1, description='Meas. Cov (R)'),
    c=widgets.FloatLogSlider(value=0.001, base=10, min=-5, max=-1, step=0.1, description='Proc. Cov (c)'),
    cutoff=widgets.FloatSlider(value=0.05, min=0.001, max=0.5, step=0.01, description='FFT Cutoff Freq'),
)
