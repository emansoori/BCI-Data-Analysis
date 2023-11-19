import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

Data = mne.io.read_epochs_eeglab('E:\DATA Hadi\Data\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set')

df = Data.to_data_frame()


F7_Electrode = df[df.columns[10:11]].to_numpy()[:1123]
T4_Electrode = df[df.columns[31:32]].to_numpy()[:1123]


plt.figure(figsize=(10, 6))


plt.plot(F7_Electrode, label='F7')
plt.plot(T4_Electrode, label='T4')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('EEG Data - Channels F7 and T4')
plt.legend()



plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

frequencies, psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies, psd_f7, color='b', linestyle='-', label='F7 PSD')

frequencies, psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies, psd_t4, color='r', linestyle='-', label='T4 PSD')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density (PSD) - Welch Method')
plt.legend()

plt.tight_layout()
plt.show()
