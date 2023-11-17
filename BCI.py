import mne
import time
import time
import numpy as np



Data = mne.io.read_epochs_eeglab('d:\Data\Recordings\Phase 1\PreProcessedData\P2 Processed\P2.set')
df = Data.to_data_frame()
x = df[df.columns[3:4]].to_numpy()
x = x[:1123]
x = np.transpose(x)
print (x.shape)
