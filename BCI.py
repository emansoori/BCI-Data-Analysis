import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def calculate_extreme_statistics_iqr(data, multiplier=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    extreme_values = data[(data < lower_bound) | (data > upper_bound)]
    sample_numbers = np.where((data < lower_bound) | (data > upper_bound))[0]

    return {
        'Definition': f'IQR (Multiplier: {multiplier})',
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Number of Extreme Values': len(extreme_values),
        'Sample Numbers of Extreme Values': sample_numbers,
        'Mean of Extreme Values': np.mean(extreme_values),
        'Max of Extreme Values': np.max(extreme_values),
        'Extreme Values': extreme_values,
    }

freq_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 30), 'Gamma': (30, 50)}

Data = mne.io.read_epochs_eeglab(r'D:\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set')
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

original_frequencies_f7, original_psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
original_frequencies_t4, original_psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')


fig, ax2_original = plt.subplots(figsize=(10, 6))
ax2_original.semilogy(original_frequencies_f7, original_psd_f7, color='b', linestyle='-', label='Original F7 PSD')
ax2_original.semilogy(original_frequencies_t4, original_psd_t4, color='r', linestyle='-', label='Original T4 PSD')  
ax2_original.set_ylabel('Power/Frequency (dB/Hz)')
ax2_original.set_title('Original Power Spectral Density (PSD) - Welch Method')
ax2_original.legend()


frequencies_f7, psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
frequencies_t4, psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.semilogy(frequencies_f7, psd_f7, color='b', linestyle='-', label='F7 PSD')
ax1.semilogy(frequencies_t4, psd_t4, color='r', linestyle='-', label='T4 PSD')  
ax1.set_ylabel('Power/Frequency (dB/Hz)')
ax1.set_title('Trends in Power Spectral Density (PSD) - Welch Method')
ax1.legend()

beta_low, beta_high = freq_bands['Beta']
ax1.axvline(x=beta_high, color='green', linestyle='--')
ax1.axvline(x=10.5, color='green', linestyle='--')
ax1.axvline(x=100, color='green', linestyle='--', label='Additional Line at 100 Hz')
ax1.axvline(x=125, color='green', linestyle='--')
ax1.text(18, 30e-5, 'Motor\nImagery\nPeriod', fontsize=12, color='black')
ax1.text(45, 45e-5, 'Still focusd but not Motor Imagery Period', fontsize=12, color='black')


upward_trend_start = 10.5
upward_trend_end = 30
downward_trend_1_start = 30
downward_trend_1_end = 100
downward_trend_2_start = 101
downward_trend_2_end = 125


arrow_properties_upward = dict(facecolor='blue', edgecolor='blue', arrowstyle='->', linewidth=2)
ax1.annotate('', xy=(upward_trend_end, psd_f7[np.argmin(np.abs(frequencies_f7 - upward_trend_end))]),
             xytext=(upward_trend_start, psd_f7[np.argmin(np.abs(frequencies_f7 - upward_trend_start))]),
             arrowprops=arrow_properties_upward)

arrow_properties_downward_1 = dict(facecolor='blue', edgecolor='blue', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_1_start, psd_f7[np.argmin(np.abs(frequencies_f7 - downward_trend_1_start))]),
             xytext=(downward_trend_1_end, psd_f7[np.argmin(np.abs(frequencies_f7 - downward_trend_1_end))]),
             arrowprops=arrow_properties_downward_1)

arrow_properties_downward_2 = dict(facecolor='blue', edgecolor='blue', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_2_start, psd_f7[np.argmin(np.abs(frequencies_f7 - downward_trend_2_start))]),
             xytext=(downward_trend_2_end, psd_f7[np.argmin(np.abs(frequencies_f7 - downward_trend_2_end))]),
             arrowprops=arrow_properties_downward_2)



downward_trend_t4_1_start = 10.5
downward_trend_t4_1_end = 30
downward_trend_t4_2_start = 30
downward_trend_t4_2_end = 100
downward_trend_t4_3_start = 101
downward_trend_t4_3_end = 125


arrow_properties_downward_t4_1 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_t4_1_start, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_1_start))]),
             xytext=(downward_trend_t4_1_end, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_1_end))]),
             arrowprops=arrow_properties_downward_t4_1)

arrow_properties_downward_t4_2 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_t4_2_start, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_2_start))]),
             xytext=(downward_trend_t4_2_end, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_2_end))]),
             arrowprops=arrow_properties_downward_t4_2)

arrow_properties_downward_t4_3 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_t4_3_start, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_3_start))]),
             xytext=(downward_trend_t4_3_end, psd_t4[np.argmin(np.abs(frequencies_t4 - downward_trend_t4_3_end))]),
             arrowprops=arrow_properties_downward_t4_3)

mean_f7, std_f7 = np.mean(F7_Electrode[:, 0]), np.std(F7_Electrode[:, 0])
mean_t4, std_t4 = np.mean(T4_Electrode[:, 0]), np.std(T4_Electrode[:, 0])

print(f'Mean and Standard Deviation for F7: {mean_f7}, {std_f7}')
print(f'Mean and Standard Deviation for T4: {mean_t4}, {std_t4}')


mean_amplitudes_f7 = []
mean_amplitudes_t4 = []

colors = plt.cm.viridis(np.linspace(0, 1, len(freq_bands)))

fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})

for color, (band, (f_low, f_high)) in zip(colors, freq_bands.items()):
    indices_f7 = np.where((frequencies_f7 >= f_low) & (frequencies_f7 <= f_high))
    avg_amplitude_f7 = np.mean(psd_f7[indices_f7])
    mean_amplitudes_f7.append(avg_amplitude_f7)

    indices_t4 = np.where((frequencies_t4 >= f_low) & (frequencies_t4 <= f_high))
    avg_amplitude_t4 = np.mean(psd_t4[indices_t4])
    mean_amplitudes_t4.append(avg_amplitude_t4)

ax2.bar(freq_bands.keys(), mean_amplitudes_f7, alpha=0.7, color=colors)
ax2.set_xlabel('Frequency Band')
ax2.set_ylabel('Average Amplitude')
ax2.set_title('Avg. Amplitude in Different Frequency Bands - F7 Electrode')
ax2.legend()

ax3.bar(freq_bands.keys(), mean_amplitudes_t4, alpha=0.7, color=colors)
ax3.set_xlabel('Frequency Band')
ax3.set_ylabel('Average Amplitude')
ax3.set_title('Avg. Amplitude in Different Frequency Bands - T4 Electrode')
ax3.legend()

highlight_bands = ['Beta', 'Gamma']
for ax, mean_amplitudes, band, color in zip([ax2, ax3], [mean_amplitudes_f7, mean_amplitudes_t4], ['F7', 'T4'], colors):
    for bar, band_name in zip(ax.patches, freq_bands.keys()):
        if band_name in highlight_bands:
            bar.set_facecolor('red')


result_f7_iqr = calculate_extreme_statistics_iqr(F7_Electrode[:, 0])


result_f7_iqr_2 = calculate_extreme_statistics_iqr(F7_Electrode[:, 0], multiplier=2)


fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})
ax5.plot(F7_Electrode[:, 0], label='F7 Electrode')
ax5.axhline(result_f7_iqr['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 1.5)')
ax5.axhline(result_f7_iqr['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 1.5)')
ax5.scatter(result_f7_iqr['Sample Numbers of Extreme Values'], result_f7_iqr['Extreme Values'], color='g', label='Outliers (Multiplier: 1.5)')
ax5.set_title('F7 Electrode IQR Outliers (Multiplier: 1.5)')
ax5.set_xlabel('Sample Numbers')
ax5.set_ylabel('Amplitude')
ax5.legend()


ax6.plot(F7_Electrode[:, 0], label='F7 Electrode')
ax6.axhline(result_f7_iqr_2['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 2)')
ax6.axhline(result_f7_iqr_2['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 2)')
ax6.scatter(result_f7_iqr_2['Sample Numbers of Extreme Values'], result_f7_iqr_2['Extreme Values'], color='b', label='Outliers (Multiplier: 2)')
ax6.set_title('F7 Electrode IQR Outliers (Multiplier: 2)')
ax6.set_xlabel('Sample Numbers')
ax6.set_ylabel('Amplitude')
ax6.legend()


result_t4_iqr = calculate_extreme_statistics_iqr(T4_Electrode[:, 0])


result_t4_iqr_2 = calculate_extreme_statistics_iqr(T4_Electrode[:, 0], multiplier=2)


fig, (ax7, ax8) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})
ax7.plot(T4_Electrode[:, 0], label='T4 Electrode')
ax7.axhline(result_t4_iqr['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 1.5)')
ax7.axhline(result_t4_iqr['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 1.5)')
ax7.scatter(result_t4_iqr['Sample Numbers of Extreme Values'], result_t4_iqr['Extreme Values'], color='g', label='Outliers (Multiplier: 1.5)')
ax7.set_title('T4 Electrode IQR Outliers (Multiplier: 1.5)')
ax7.set_xlabel('Sample Numbers')
ax7.set_ylabel('Amplitude')
ax7.legend()


ax8.plot(T4_Electrode[:, 0], label='T4 Electrode')
ax8.axhline(result_t4_iqr_2['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 2)')
ax8.axhline(result_t4_iqr_2['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 2)')
ax8.scatter(result_t4_iqr_2['Sample Numbers of Extreme Values'], result_t4_iqr_2['Extreme Values'], color='b', label='Outliers (Multiplier: 2)')
ax8.set_title('T4 Electrode IQR Outliers (Multiplier: 2)')
ax8.set_xlabel('Sample Numbers')
ax8.set_ylabel('Amplitude')
ax8.legend()

result_t4_iqr = calculate_extreme_statistics_iqr(T4_Electrode[:, 0])

# Access the sample numbers array
sample_numbers_t4 = result_t4_iqr['Sample Numbers of Extreme Values']

# Print the sample numbers for T4
print("Sample Numbers of Extreme Values for T4:", sample_numbers_t4)

plt.tight_layout()
plt.show()
