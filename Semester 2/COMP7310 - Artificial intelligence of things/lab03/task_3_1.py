import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

class task_3_1:
    def __init__(self, fs=500):
        """
        Initializes the task_3_1 class for performing ACF on different signals.

        Attributes:
            fs (int): Sampling rate in Hz, default is 500 Hz.
        """
        self.fs = fs

    def apply_acf_pt(self):
        """
        Computes the ACF of a pure tone signal.

        The pure tone signal is defined as:
        s(t) = 2.5 * cos(2 * pi * 12.3 * t + pi/3)
        where 0 <= t < 10 with sampling rate of self.fs.
        
        You need to first generate the signal in time domain and then compute its ACF.

        Returns:
            - acf (np.float64): the ACF of the signal.
        
        >>> test = task_3_1()
        >>> acf = test.apply_acf_pt()
        >>> np.round(acf[0], 5)
        1.0
        """
        
        N = 10 * self.fs
        s_t = np.zeros(N, dtype=np.float64)
        acf = np.zeros(N, dtype=np.float64)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        t = np.arange(0, 10, 1/self.fs)
        s_t = 2.5 * np.cos(2 * np.pi * 12.3 * t + np.pi/3)
        acf = correlate(s_t, s_t, mode='full')
        acf = acf[acf.size // 2:] / np.max(acf)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<

        
        acf = np.array(acf).astype(np.float64)
        return acf
    
    def apply_acf_pulse(self):
        """
        Computes the ACF of a pulse signal.

        The pulse signal is defined as:
        s(t) =
            - 0 for 0 <= t < 0.3
            - 3 for 0.3 <= t < 0.9
            - 0 for 0.9 <= t < 1.2
            - 2 for 1.2 <= t < 1.8
            - 0 for 1.8 <= t < 2

        You need to first generate the signal in time domain and then compute its ACF.
        
        Returns:
            - acf (np.float64): the ACF of the signal.
            
        >>> test = task_3_1()
        >>> acf = test.apply_acf_pulse()
        >>> np.round(acf[0], 5)
        1.0
        
        """
        N = 2 * self.fs
        s_t = np.zeros(N, dtype=np.float64)
        acf = np.zeros(N, dtype=np.float64)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        t = np.arange(0, 2, 1/self.fs)
        s_t[(t >= 0.3) & (t < 0.9)] = 3
        s_t[(t >= 1.2) & (t < 1.8)] = 2
        acf = correlate(s_t, s_t, mode='full')
        acf = acf[acf.size // 2:] / np.max(acf)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<

        
        acf = np.array(acf).astype(np.float64)
        return acf
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)