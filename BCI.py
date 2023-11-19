import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

Data = mne.io.read_epochs_eeglab('E:\DATA Hadi\Data\Data\Recordings\Phase 1\PreProcessedData\P2\P2.set')

df = Data.to_data_frame()


T7_Electrode = df[df.columns[9:10]].to_numpy()[:1123]
O2_Electrode = df[df.columns[18:19]].to_numpy()[:1123]


plt.figure(figsize=(10, 8))


plt.subplot(2, 1, 1)
plt.plot(T7_Electrode, label='T7')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('EEG Data - Channel T7')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(O2_Electrode, label='O2', color='red')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('EEG Data - Channel O2')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))


plt.subplot(2, 1, 1)
frequencies, psd_t7 = welch(T7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies, psd_t7, color='b', linestyle='-', label='T7 PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density (PSD) - T7')
plt.legend()


plt.subplot(2, 1, 2)
frequencies, psd_o2 = welch(O2_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies, psd_o2, color='r', linestyle='-', label='O2 PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density (PSD) - O2')
plt.legend()

plt.tight_layout()
plt.show()
