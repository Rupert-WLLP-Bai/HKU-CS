import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

class task_2_1:
    def __init__(self, fs=2000):
        """
        Initializes the task_2_1 class for performing FFT on different signals.

        Attributes:
            fs (int): Sampling rate in Hz, default is 2000 Hz.
            t (np.ndarray): Time array ranging from -1 to 1 second.
            N (int): Length of the time array.
        """
        self.fs = fs
        self.t = np.arange(-1, 1, 1/fs)
        self.N = len(self.t)
    
    def apply_fft_pt(self):
        """
        Computes the frequency spectrum of a pure tone signal.

        The pure tone signal is defined as:
        s(t) = 2.025 * cos(2 * pi * 20.25 * t + pi/3)
        
        You need to first generate the signal in time domain and then compute its frequency spectrum.

        Returns:
            tuple: (s_f_freq, s_f)
            - s_f_freq (np.float64): the frequency axis 
            - s_f (np.float64): the magnitude of the frequency spectrum.
        
        >>> test = task_2_1()
        >>> freq, mag = test.apply_fft_pt()
        >>> np.round(mag[-1], 1), np.round(freq[-1], 1)
        (1.7, 999.5)
        """
        s_t = np.zeros(self.N, dtype=np.float64)
        s_f = np.zeros(self.N // 2, dtype=np.float64)
        s_f_freq = np.zeros(self.N // 2, dtype=np.float64)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<

        
        s_f_freq = np.array(s_f_freq).astype(np.float64)
        s_f = np.array(s_f).astype(np.float64)
        
        return s_f_freq, s_f
    
