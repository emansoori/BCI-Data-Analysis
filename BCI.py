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
print (df)
print (df[df.columns[4]])

CP1_Electrode = df[df.columns[10:11]].to_numpy()[:1123]
AF7_Electrode = df[df.columns[31:32]].to_numpy()[:1123]


fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].plot(CP1_Electrode, label='CP1', color='b')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('EEG Data - Channel CP1')
axs[0].legend()

axs[1].plot(AF7_Electrode, label='AF7', color='r')
axs[1].set_xlabel('Sample Number')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('EEG Data - Channel AF7')
axs[1].legend()

original_frequencies_CP1, original_psd_CP1 = welch(CP1_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
original_frequencies_AF7, original_psd_AF7 = welch(AF7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')


fig, ax2_original = plt.subplots(figsize=(10, 6))
ax2_original.semilogy(original_frequencies_CP1, original_psd_CP1, color='b', linestyle='-', label='Original CP1 PSD')
ax2_original.semilogy(original_frequencies_AF7, original_psd_AF7, color='r', linestyle='-', label='Original AF7 PSD')  
ax2_original.set_ylabel('Power/Frequency (dB/Hz)')
ax2_original.set_title('Original Power Spectral Density (PSD) - Welch Method')
ax2_original.legend()


frequencies_CP1, psd_CP1 = welch(CP1_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
frequencies_AF7, psd_AF7 = welch(AF7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.semilogy(frequencies_CP1, psd_CP1, color='b', linestyle='-', label='CP1 PSD')
ax1.semilogy(frequencies_AF7, psd_AF7, color='r', linestyle='-', label='AF7 PSD')  
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
ax1.annotate('', xy=(upward_trend_end, psd_CP1[np.argmin(np.abs(frequencies_CP1 - upward_trend_end))]),
             xytext=(upward_trend_start, psd_CP1[np.argmin(np.abs(frequencies_CP1 - upward_trend_start))]),
             arrowprops=arrow_properties_upward)

arrow_properties_downward_1 = dict(facecolor='blue', edgecolor='blue', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_1_start, psd_CP1[np.argmin(np.abs(frequencies_CP1 - downward_trend_1_start))]),
             xytext=(downward_trend_1_end, psd_CP1[np.argmin(np.abs(frequencies_CP1 - downward_trend_1_end))]),
             arrowprops=arrow_properties_downward_1)

arrow_properties_downward_2 = dict(facecolor='blue', edgecolor='blue', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_2_start, psd_CP1[np.argmin(np.abs(frequencies_CP1 - downward_trend_2_start))]),
             xytext=(downward_trend_2_end, psd_CP1[np.argmin(np.abs(frequencies_CP1 - downward_trend_2_end))]),
             arrowprops=arrow_properties_downward_2)



downward_trend_AF7_1_start = 10.5
downward_trend_AF7_1_end = 30
downward_trend_AF7_2_start = 30
downward_trend_AF7_2_end = 100
downward_trend_AF7_3_start = 101
downward_trend_AF7_3_end = 125


arrow_properties_downward_AF7_1 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_AF7_1_start, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_1_start))]),
             xytext=(downward_trend_AF7_1_end, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_1_end))]),
             arrowprops=arrow_properties_downward_AF7_1)

arrow_properties_downward_AF7_2 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_AF7_2_start, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_2_start))]),
             xytext=(downward_trend_AF7_2_end, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_2_end))]),
             arrowprops=arrow_properties_downward_AF7_2)

arrow_properties_downward_AF7_3 = dict(facecolor='red', edgecolor='red', arrowstyle='<-', linewidth=2)
ax1.annotate('', xy=(downward_trend_AF7_3_start, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_3_start))]),
             xytext=(downward_trend_AF7_3_end, psd_AF7[np.argmin(np.abs(frequencies_AF7 - downward_trend_AF7_3_end))]),
             arrowprops=arrow_properties_downward_AF7_3)

mean_CP1, std_CP1 = np.mean(CP1_Electrode[:, 0]), np.std(CP1_Electrode[:, 0])
mean_AF7, std_AF7 = np.mean(AF7_Electrode[:, 0]), np.std(AF7_Electrode[:, 0])

print(f'Mean and Standard Deviation for CP1: {mean_CP1}, {std_CP1}')
print(f'Mean and Standard Deviation for AF7: {mean_AF7}, {std_AF7}')


mean_amplitudes_CP1 = []
mean_amplitudes_AF7 = []

colors = plt.cm.viridis(np.linspace(0, 1, len(freq_bands)))

fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})

for color, (band, (f_low, f_high)) in zip(colors, freq_bands.items()):
    indices_CP1 = np.where((frequencies_CP1 >= f_low) & (frequencies_CP1 <= f_high))
    avg_amplitude_CP1 = np.mean(psd_CP1[indices_CP1])
    mean_amplitudes_CP1.append(avg_amplitude_CP1)

    indices_AF7 = np.where((frequencies_AF7 >= f_low) & (frequencies_AF7 <= f_high))
    avg_amplitude_AF7 = np.mean(psd_AF7[indices_AF7])
    mean_amplitudes_AF7.append(avg_amplitude_AF7)

ax2.bar(freq_bands.keys(), mean_amplitudes_CP1, alpha=0.7, color=colors)
ax2.set_xlabel('Frequency Band')
ax2.set_ylabel('Average Amplitude')
ax2.set_title('Avg. Amplitude in Different Frequency Bands - CP1 Electrode')
ax2.legend()

ax3.bar(freq_bands.keys(), mean_amplitudes_AF7, alpha=0.7, color=colors)
ax3.set_xlabel('Frequency Band')
ax3.set_ylabel('Average Amplitude')
ax3.set_title('Avg. Amplitude in Different Frequency Bands - AF7 Electrode')
ax3.legend()

highlight_bands = ['Beta', 'Gamma']
for ax, mean_amplitudes, band, color in zip([ax2, ax3], [mean_amplitudes_CP1, mean_amplitudes_AF7], ['CP1', 'AF7'], colors):
    for bar, band_name in zip(ax.patches, freq_bands.keys()):
        if band_name in highlight_bands:
            bar.set_facecolor('red')


result_CP1_iqr = calculate_extreme_statistics_iqr(CP1_Electrode[:, 0])


result_CP1_iqr_2 = calculate_extreme_statistics_iqr(CP1_Electrode[:, 0], multiplier=2)


fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})
ax5.plot(CP1_Electrode[:, 0], label='CP1 Electrode')
ax5.axhline(result_CP1_iqr['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 1.5)')
ax5.axhline(result_CP1_iqr['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 1.5)')
ax5.scatter(result_CP1_iqr['Sample Numbers of Extreme Values'], result_CP1_iqr['Extreme Values'], color='g', label='Outliers (Multiplier: 1.5)')
ax5.set_title('CP1 Electrode IQR Outliers (Multiplier: 1.5)')
ax5.set_xlabel('Sample Numbers')
ax5.set_ylabel('Amplitude')
ax5.legend()


ax6.plot(CP1_Electrode[:, 0], label='CP1 Electrode')
ax6.axhline(result_CP1_iqr_2['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 2)')
ax6.axhline(result_CP1_iqr_2['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 2)')
ax6.scatter(result_CP1_iqr_2['Sample Numbers of Extreme Values'], result_CP1_iqr_2['Extreme Values'], color='b', label='Outliers (Multiplier: 2)')
ax6.set_title('CP1 Electrode IQR Outliers (Multiplier: 2)')
ax6.set_xlabel('Sample Numbers')
ax6.set_ylabel('Amplitude')
ax6.legend()


result_AF7_iqr = calculate_extreme_statistics_iqr(AF7_Electrode[:, 0])


result_AF7_iqr_2 = calculate_extreme_statistics_iqr(AF7_Electrode[:, 0], multiplier=2)


fig, (ax7, ax8) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.5})
ax7.plot(AF7_Electrode[:, 0], label='AF7 Electrode')
ax7.axhline(result_AF7_iqr['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 1.5)')
ax7.axhline(result_AF7_iqr['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 1.5)')
ax7.scatter(result_AF7_iqr['Sample Numbers of Extreme Values'], result_AF7_iqr['Extreme Values'], color='g', label='Outliers (Multiplier: 1.5)')
ax7.set_title('AF7 Electrode IQR Outliers (Multiplier: 1.5)')
ax7.set_xlabel('Sample Numbers')
ax7.set_ylabel('Amplitude')
ax7.legend()


ax8.plot(AF7_Electrode[:, 0], label='AF7 Electrode')
ax8.axhline(result_AF7_iqr_2['Lower Bound'], color='r', linestyle='--', label='Lower Bound (Multiplier: 2)')
ax8.axhline(result_AF7_iqr_2['Upper Bound'], color='r', linestyle='--', label='Upper Bound (Multiplier: 2)')
ax8.scatter(result_AF7_iqr_2['Sample Numbers of Extreme Values'], result_AF7_iqr_2['Extreme Values'], color='b', label='Outliers (Multiplier: 2)')
ax8.set_title('AF7 Electrode IQR Outliers (Multiplier: 2)')
ax8.set_xlabel('Sample Numbers')
ax8.set_ylabel('Amplitude')
ax8.legend()

result_AF7_iqr = calculate_extreme_statistics_iqr(AF7_Electrode[:, 0])


sample_numbers_AF7 = result_AF7_iqr['Sample Numbers of Extreme Values']


print("Sample Numbers of Extreme Values for AF7:", sample_numbers_AF7)

outliers_CP1 = result_CP1_iqr['Extreme Values']
std_outliers_CP1 = np.std(outliers_CP1)


outliers_AF7 = result_AF7_iqr['Extreme Values']
std_outliers_AF7 = np.std(outliers_AF7)

print("Standard Deviation for Outliers of CP1:", std_outliers_CP1)
print("Standard Deviation for Outliers of AF7:", std_outliers_AF7)

std_outliers_CP1 = np.std(outliers_CP1)
std_outliers_AF7 = np.std(outliers_AF7)


fig, ax9 = plt.subplots(figsize=(8, 6))
std_deviations = [std_outliers_CP1, std_outliers_AF7]
labels = ['CP1 Outliers', 'AF7 Outliers']

ax9.bar(labels, std_deviations, color=['blue', 'red'])
ax9.set_ylabel('Standard Deviation')
ax9.set_title('Standard Deviation of Outliers for CP1 and AF7 (Multiplier is 1.5)')

plt.tight_layout()
plt.show()
