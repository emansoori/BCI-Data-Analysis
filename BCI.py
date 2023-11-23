import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def calculate_extreme_statistics_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    extreme_values = data[(data < lower_bound) | (data > upper_bound)]

    return {
        'Definition': 'IQR',
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Number of Extreme Values': len(extreme_values),
        'Mean of Extreme Values': np.mean(extreme_values),
        'Max of Extreme Values': np.max(extreme_values),
        'Extreme Values': extreme_values,
    }


freq_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 30), 'Gamma': (30, 50)}

Data = mne.io.read_epochs_eeglab('D:\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set')
df = Data.to_data_frame()

F7_Electrode = df[df.columns[10:11]].to_numpy()[:1123]
T4_Electrode = df[df.columns[31:32]].to_numpy()[:1123]


fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(F7_Electrode, label='F7', color='b')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('EEG Data - Channel F7')
axs[0].legend()

axs[1].plot(T4_Electrode, label='T4', color='r')
axs[1].set_xlabel('Sample Number')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('EEG Data - Channel T4')
axs[1].legend()

plt.tight_layout()


frequencies_f7, psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
frequencies_t4, psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')


fig, ax = plt.subplots(figsize=(10, 6))
plt.semilogy(frequencies_f7, psd_f7, color='b', linestyle='-', label='F7 PSD')
plt.semilogy(frequencies_t4, psd_t4, color='r', linestyle='-', label='T4 PSD')

beta_low, beta_high = freq_bands['Beta']
plt.axvline(x=beta_low, color='green', linestyle='--', label='Beta Region')
plt.axvline(x=beta_high, color='green', linestyle='--')

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


mean_amplitudes_f7 = []

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


mean_amplitudes_t4 = []

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


result_f7_iqr = calculate_extreme_statistics_iqr(F7_Electrode[:, 0])
result_t4_iqr = calculate_extreme_statistics_iqr(T4_Electrode[:, 0])


print("\nResults for F7 Electrode (IQR):")
for key, value in result_f7_iqr.items():
    print(f"{key}: {value}")

print("\nResults for T4 Electrode (IQR):")
for key, value in result_t4_iqr.items():
    print(f"{key}: {value}")

# Show the plots
plt.show()
