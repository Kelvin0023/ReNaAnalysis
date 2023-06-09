import numpy as np
from matplotlib import pyplot as plt


def simulate_eeg(start_time, end_time, srate=128, num_channels=64, freq=10, amplitude=1, noise=0.5):
    """
    Simulates an EEG signal with an oscillation of a given frequency and amplitude,
    along with white noise of a specified standard deviation.

    Parameters:
        duration (float): Duration of the signal in seconds (default=1).
        srate (int): Sampling frequency in Hz (default=1000).
        freq (float): Frequency of the oscillation in Hz (default=10).
        amplitude (float): Amplitude of the oscillation in microvolts (default=1).
        noise (float): Standard deviation of the white noise in microvolts (default=0.5).

    Returns:
        tuple: A tuple containing the time vector and the simulated EEG signal.
    """
    if end_time < start_time:
        raise ValueError("End time must be greater than start time.")
    duration = end_time - start_time
    # Create time vector
    t = np.linspace(start_time, end_time, int(duration * srate), endpoint=False)

    # Generate oscillation
    oscillation = amplitude * np.sin(2 * np.pi * freq * t)
    oscillation = np.tile(oscillation, (num_channels, 1))
    # Generate noise
    noise = noise * np.random.randn(num_channels, len(t))

    # Combine oscillation and noise
    eeg = oscillation + noise

    # Plot the simulated EEG signal
    # plt.plot(t, eeg)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (uV)')
    # plt.show()

    # Return time vector and simulated EEG signal
    return t, eeg

if __name__ == "__main__":
# Simulate EEG signal
    t, eeg = simulate_eeg(10, 20, srate=128, freq=10, amplitude=1, noise=0.5)

    # Plot the simulated EEG signal
    # plt.plot(t, eeg)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude (uV)')
    # plt.show()
