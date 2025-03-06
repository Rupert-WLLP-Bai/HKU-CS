import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import pickle

class task_4_1:
    def __init__(self, data_root="./data/"):
        """
        Initializes the task_4_1 class by loading signal data from a specified path.

        Parameters:
            data_root (str): The root directory where the signal data file is located.
                             The default value is "./data/".

        Attributes:
            data (np.ndarray): Loaded signal data.
            fs (int): Sampling rate in Hz, initialized from the loaded data.
        """
        file_n = "task_4_1.pickle"
        with open(osp.join(data_root, file_n), "rb") as f:
            data = pickle.load(f)
        self.data, self.fs = data["values"], data["fs"]
    
    def get_data(self):
        """
        Returns the loaded signal data and its sampling frequency.

        Returns:
            np.ndarray: The loaded signal data.
            int: The sampling frequency of the signal.
        """
        return self.data, self.fs

    def apply_highpass(self, data, fs, cutoff=20, order=5):
        """
        Applies a 20Hz high-pass filter to the loaded signal.
        
        Parameters:
            cutoff (int): Cutoff frequency for the high-pass filter. 

        Returns:
            np.ndarray: The high-pass filtered signal.
        
        >>> test = task_4_1(data_root="./data/")
        >>> data, fs = test.get_data()
        >>> filtered = test.apply_highpass(data, fs)
        >>> np.all(filtered != None)
        True
        >>> round(filtered[10], 2), round(filtered[-1], 2)
        (0.23, 0.08)
        """
        filtered = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Implement the high-pass filter using the butter and sosfiltfilt functions
        # sos = ...
        # filtered = 
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        filtered = np.array(filtered, dtype=np.float64)
        return filtered

    def apply_lowpass(self, data, fs, cutoff=10, order=5):
        """
        Applies a 10Hz low-pass filter to the loaded signal.
        
        Parameters:
            cutoff (int): Cutoff frequency for the low-pass filter. 

        Returns:
            np.ndarray: The low-pass filtered signal.
        
        >>> test = task_4_1(data_root="./data/")
        >>> data, fs = test.get_data()
        >>> filtered = test.apply_lowpass(data, fs)
        >>> np.all(filtered != None)
        True
        >>> round(filtered[10], 2), round(filtered[-1], 2)
        (0.51, -0.64)
        """
        filtered = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Implement the low-pass filter using the butter and sosfiltfilt functions
        # sos = ...
        # filtered = ...
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        filtered = np.array(filtered, dtype=np.float64)
        return filtered

    def apply_bandpass(self, data, fs, cutoff_low=10, cutoff_high=20, order=5):
        """
        Applies a band-pass filter to the loaded signal by combining high-pass and low-pass filters.
        
        Returns:
            np.ndarray: The band-pass filtered signal.
            cutoff1 (int): Cutoff frequency for the high-pass filter.
            cutoff2 (int): Cutoff frequency for the low-pass filter.
        
        >>> test = task_4_1(data_root="./data/")
        >>> data, fs = test.get_data()
        >>> filtered, cutoff1, cutoff2 = test.apply_bandpass(data, fs)
        >>> np.all(filtered != None), cutoff1 != None, cutoff2 != None
        (True, True, True)
        >>> round(filtered[10], 2), round(filtered[-1], 2)
        (0.05, -0.0)
        """
        
        filtered, cutoff1, cutoff2 = None, None, None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Set the cutoff frequencies for the low-pass and high-pass filters
        # cutoff1 = ...
        # cutoff2 = ...
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        high_passed = self.apply_highpass(cutoff=cutoff1, data=data, fs=fs, order=order)
        filtered = self.apply_lowpass(cutoff=cutoff2, data=high_passed, fs=fs, order=order)
        return filtered, cutoff1, cutoff2


