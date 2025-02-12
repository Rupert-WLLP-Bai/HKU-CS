import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class task_2_2:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_2_2 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            spt_fn (str): Filename for the sum of pure tone signals (task_2_2_1).
            chirp_fn (str): Filename for the chirp signal (task_2_2_2).
            ecg_fn (str): Filename for the ECG signal (task_2_2_3).
            spt_data (dict): Loaded data for the sum of pure tone signals.
            chirp_data (dict): Loaded data for the chirp signal.
            ecg_data (dict): Loaded data for the ECG signal.
        """
        self.data_root = data_root
        self.spt_fn = "task_2_2_1.pickle"
        self.chirp_fn = "task_2_2_2.pickle"
        self.ecg_fn = "task_2_2_3.pickle"
        
        with open(osp.join(self.data_root, self.spt_fn), "rb") as f:
            self.spt_data = pickle.load(f)
        with open(osp.join(self.data_root, self.chirp_fn), "rb") as f:
            self.chirp_data = pickle.load(f)
        with open(osp.join(self.data_root, self.ecg_fn), "rb") as f:
            self.ecg_data = pickle.load(f)
        
    def get_freq_spt(self):
        """
        Analyze the sum of pure tone signals to determine the primary frequency components.

        Returns:
            freq (np.float64): An array of the three primary frequency components in descending order.
        
        >>> test = task_2_2()
        >>> f = test.get_freq_spt()
        >>> len(f) == 3
        True
        """
        s_t = self.spt_data["values"] # signal values
        fs = self.spt_data["fs"] # sampling frequency
        
        freq = np.zeros(3, dtype=np.float64) # (3,)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        N = len(s_t)
        freqs = fftfreq(N, d=1/fs)
        fft_vals = fft(s_t)
        
        fft_mag = np.abs(fft_vals)
        
        peaks, _ = find_peaks(fft_mag[:N//2], height=0)
        peak_freqs = freqs[peaks]
        
        freq[:3] = np.sort(np.abs(peak_freqs))[-3:][::-1]
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        freq = np.sort(freq)[::-1]
        freq = np.squeeze(freq[:3]).astype(np.float64)
        return freq
    
    def get_bw_chirp(self):
        """
        Compute the bandwidth of the chirp signal.

        Returns:
            bw (float64): The bandwidth of the chirp signal in Hz. Format: float64.
            
        >>> test = task_2_2()
        >>> bw = test.get_bw_chirp()
        >>> (bw >= 100) & (bw <= 1000)
        True
        """
        s_t = self.chirp_data["values"] # signal values
        fs = self.chirp_data["fs"] # sampling frequency
        
        bw = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # Compute the FFT of the signal
        N = len(s_t)
        freqs = fftfreq(N, d=1/fs)  # Frequency bins
        fft_vals = fft(s_t)  # Compute FFT
        
        # Get the magnitude of the FFT values
        fft_mag = np.abs(fft_vals)
        
        # Consider only the positive frequency part
        positive_freqs = freqs[:N//2]
        positive_fft_mag = fft_mag[:N//2]
        
        # Compute the maximum and minimum frequencies in the spectrum
        max_freq = np.max(positive_freqs)  # Maximum frequency
        min_freq = np.min(positive_freqs[positive_fft_mag > 0])  # Minimum non-zero frequency
        
        # Bandwidth is defined as the difference between the maximum and minimum frequencies
        bw = max_freq - min_freq
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        return bw 
    
    def get_heart_rate(self):
        """
        Determine the heart rate from the ECG signal.

        Returns:
            hr (float64): The heart rate in beats per minute (BPM).
        
        >>> test = task_2_2()
        >>> hr = test.get_heart_rate()
        >>> (hr >= 60) & (hr <= 90)
        True
        """
        s_t = self.ecg_data["values"] # signal values
        fs = self.ecg_data["fs"] # sampling frequency
        
        hr = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        r_peaks, _ = find_peaks(s_t, distance=fs*0.6)
        rr_intervals = np.diff(r_peaks) / fs
        hr = 60 / np.mean(rr_intervals)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Make sure hr is a float64
        if isinstance(hr, np.ndarray):
            if hr.size > 1:
                hr = hr[0]
            hr = hr.item()
        if isinstance(hr, list):
            if len(hr) > 1:
                hr = hr[0]
        hr = float(hr)
        return hr

if __name__ == "__main__":
    data_root = "./data/" # Change this to the directory where you store the data
    test = task_2_2(data_root=data_root)
    # ...