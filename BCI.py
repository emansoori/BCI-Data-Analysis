import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

Data = mne.io.read_epochs_eeglab('D:\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set')
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

plt.figure(figsize=(10, 6))
frequencies_f7, psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies_f7, psd_f7, color='b', linestyle='-', label='F7 PSD')

frequencies_t4, psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
plt.semilogy(frequencies_t4, psd_t4, color='r', linestyle='-', label='T4 PSD')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density (PSD) - Welch Method')
plt.legend()

mean_f7 = np.mean(F7_Electrode[:, 0])
std_f7 = np.std(F7_Electrode[:, 0])

mean_t4 = np.mean(T4_Electrode[:, 0])
std_t4 = np.std(T4_Electrode[:, 0])

print(f'Mean and Standard Deviation for F7: {mean_f7}, {std_f7}')
print(f'Mean and Standard Deviation for T4: {mean_t4}, {std_t4}')

plt.tight_layout()

freq_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 30), 'Gamma': (30, 50)}

mean_amplitudes_f7 = []
mean_amplitudes_t4 = []

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(freq_bands)))

for color, (band, (f_low, f_high)) in zip(colors, freq_bands.items()):
    indices_f7 = np.where((frequencies_f7 >= f_low) & (frequencies_f7 <= f_high))
    avg_amplitude_f7 = np.mean(psd_f7[indices_f7])
    mean_amplitudes_f7.append(avg_amplitude_f7)

ax.bar(freq_bands.keys(), mean_amplitudes_f7, alpha=0.7, color=colors, label='F7')

highlight_bands = ['Beta', 'Gamma']
for bar, band, color in zip(ax.patches, freq_bands.keys(), colors):
    if band in highlight_bands:
        bar.set_facecolor('red')

plt.xlabel('Frequency Band')
plt.ylabel('Average Amplitude')
plt.title('Average Amplitude in Different Frequency Bands - F7 Electrode')
plt.legend()

fig, ax = plt.subplots(figsize=(10, 6))
for color, (band, (f_low, f_high)) in zip(colors, freq_bands.items()):
    indices_t4 = np.where((frequencies_t4 >= f_low) & (frequencies_t4 <= f_high))
    avg_amplitude_t4 = np.mean(psd_t4[indices_t4])
    mean_amplitudes_t4.append(avg_amplitude_t4)

ax.bar(freq_bands.keys(), mean_amplitudes_t4, alpha=0.7, color=colors, label='T4')

for bar, band, color in zip(ax.patches, freq_bands.keys(), colors):
    if band in highlight_bands:
        bar.set_facecolor('red')

plt.xlabel('Frequency Band')
plt.ylabel('Average Amplitude')
plt.title('Average Amplitude in Different Frequency Bands - T4 Electrode')
plt.legend()

plt.show()
