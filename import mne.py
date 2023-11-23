import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

freq_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 30), 'Gamma': (30, 50)}

Data = mne.io.read_epochs_eeglab('D:\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set')
df = Data.to_data_frame()

F7_Electrode = df[df.columns[10:11]].to_numpy()[:1123]
T4_Electrode = df[df.columns[31:32]].to_numpy()[:1123]

# Calculate power spectral density using Welch method
frequencies_f7, psd_f7 = welch(F7_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')
frequencies_t4, psd_t4 = welch(T4_Electrode[:, 0], fs=Data.info['sfreq'], nperseg=256, scaling='density')

# Compute statistics for extreme values using IQR
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

# Test sensitivity to IQR
result_f7_iqr = calculate_extreme_statistics_iqr(F7_Electrode[:, 0])
result_t4_iqr = calculate_extreme_statistics_iqr(T4_Electrode[:, 0])

# Display results for IQR
print("\nSensitivity to IQR Definition of Extreme Values for F7 Electrode:")
print(result_f7_iqr)

print("\nSensitivity to IQR Definition of Extreme Values for T4 Electrode:")
print(result_t4_iqr)

# Plot EEG data with outliers
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].plot(F7_Electrode, label='F7', color='b')
extreme_indices_f7 = np.where((F7_Electrode[:, 0] < result_f7_iqr['Lower Bound']) | (F7_Electrode[:, 0] > result_f7_iqr['Upper Bound']))
axs[0].scatter(extreme_indices_f7, F7_Electrode[extreme_indices_f7, 0], color='green', label='Outliers')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('EEG Data - Channel F7 with Outliers')
axs[0].legend()

axs[1].plot(T4_Electrode, label='T4', color='r')
extreme_indices_t4 = np.where((T4_Electrode[:, 0] < result_t4_iqr['Lower Bound']) | (T4_Electrode[:, 0] > result_t4_iqr['Upper Bound']))
axs[1].scatter(extreme_indices_t4, T4_Electrode[extreme_indices_t4, 0], color='green', label='Outliers')
axs[1].set_xlabel('Sample Number')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('EEG Data - Channel T4 with Outliers')
axs[1].legend()

plt.tight_layout()

# Plot outliers in a separate figure
fig_outliers, axs_outliers = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs_outliers[0].scatter(extreme_indices_f7, F7_Electrode[extreme_indices_f7, 0], color='green', label='Outliers')
axs_outliers[0].set_ylabel('Amplitude')
axs_outliers[0].set_title('Outliers - Channel F7')
axs_outliers[0].legend()

axs_outliers[1].scatter(extreme_indices_t4, T4_Electrode[extreme_indices_t4, 0], color='green', label='Outliers')
axs_outliers[1].set_xlabel('Sample Number')
axs_outliers[1].set_ylabel('Amplitude')
axs_outliers[1].set_title('Outliers - Channel T4')
axs_outliers[1].legend()

plt.tight_layout()

# Show plots
plt.show()
