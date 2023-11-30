# CMSC 6950
# Final Project

This provided code takes the EEG signal (Brain Signal) of a participant performing several motor imagery tasks. it then takes two different electrodes, namely AF7 (Electrode number 31) and CP1 (Electrode Number 10) from the exisiting 64 channels to process. AF7 is in the frontal lobe, and CP1 is in the parietal lobe. Therefore, this project aims to see the effect of these two electrodes in the motor imagery of the left hand.

---------------------------------------------------------------------------------------------------------

Please follow these steps to reproduce each figure in my project report:
---------------------------------------------------------------------------------------------------------
Figure 1: EEG Data Plots

    Load EEG Data:
        Save the EEG data.set file in your preffered location and change my default address at 'D:\Data\Recordings\Phase 1\PreProcessedData\P3\P3.set' to your own address.

    Plot EEG Data:
        Generate line plots for channels CP1 and AF7.
        X-axis: Sample Number
        Y-axis: Amplitude
        Use blue for CP1 and red for AF7.
---------------------------------------------------------------------------------------------------------

Figure 2: Original Power Spectral Density (PSD)

    Calculate Original PSD:
        Apply the Welch method to obtain the original PSD for channels CP1 and AF7.

    Plot Original PSD:
        Create a log-scaled line plot.
        X-axis: Frequency
        Y-axis: Power/Frequency (dB/Hz)
        Use blue for CP1 and red for AF7.

---------------------------------------------------------------------------------------------------------

Figure 3: Trends in Power Spectral Density (PSD)

    Calculate PSD with Trends:
        Apply the Welch method to obtain the PSD for channels CP1 and AF7.
        Include annotations for relevant trends.

    Plot PSD with Trends:
        Create a log-scaled line plot.
        X-axis: Frequency
        Y-axis: Power/Frequency (dB/Hz)
        Use blue for CP1 and red for AF7.
        Highlight specific frequency regions.

---------------------------------------------------------------------------------------------------------


Figure 4: Average Amplitude in Different Frequency Bands

    Calculate Average Amplitude:
        Determine the average amplitude for CP1 and AF7 in different frequency bands.

    Plot Average Amplitude:
        Generate bar charts for each frequency band.
        X-axis: Frequency Band
        Y-axis: Average Amplitude
        Use a color gradient.

---------------------------------------------------------------------------------------------------------

Figure 5 AND 6: IQR Outliers

    Calculate IQR Outliers:
        Apply the Interquartile Range (IQR) method to identify outliers for CP1 with a multiplier of 1.5.

    Plot IQR Outliers:
        Create a line plot of CP1.
        Include dashed lines for the lower and upper bounds.
        Mark outliers with green dots.

    Repeat for AF7:
        Apply the IQR method with a multiplier of 2.
        Plot AF7 with outliers marked in blue.
---------------------------------------------------------------------------------------------------------

Figure 7: Standard Deviation of Outliers

    Calculate Standard Deviation:
        Determine the standard deviation of outliers for CP1 and AF7.

    Plot Standard Deviation:
        Generate a bar chart with blue for CP1 and red for AF7.
        X-axis: Outlier Type (CP1/AF7)
        Y-axis: Standard Deviation
Best regards,
Erfan

---------------------------------------------------------------------------------------------------------