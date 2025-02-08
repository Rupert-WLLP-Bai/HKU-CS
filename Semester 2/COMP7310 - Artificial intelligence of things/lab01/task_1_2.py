import numpy as np
from vis import *

class task_1_2:
    def __init__(self, fs=1000):
        """
        Initialize the class with a specific sampling rate.

        Args:
            fs (int): Sampling rate in Hz. Defaults to 1000Hz.
        """
        self.fs = fs
    
    def generate_linear_chirp(self, amplitude, period, duration, start_freq, end_freq, init_phase):
        """
        Generate a linear chirp signal.

        Args:
            amplitude (float): Amplitude of the generated signal.
            period (float): Period of the generated signal (s).
            duration (float): Duration of the generated signal (s).
            start_freq (float): Start frequency of the generated signal (Hz).
            end_freq (float): End frequency of the generated signal (Hz).
            init_phase (float): Initial phase of the generated signal (radius).

        Returns:
            t   numpy.array: Array of timestamps in seconds. Data type must be float.
            f_t numpy.array: Array of generated frequency values. Data type must be float.
            s_t numpy.array: Array of generated signal values. Daíta type must be float.
        
        >>> gen = task_1_2(1000)
        >>> t, f_t, s_t = gen.generate_linear_chirp(amplitude=1, period=1, duration=2, start_freq=1, end_freq=10, init_phase=0)
        >>> np.round(t[10], 5)
        0.01
        >>> np.round(f_t[10], 5)
        1.09
        >>> np.round(s_t[10], 5)
        0.99785
        
        """
        t = None
        f_t = None
        s_t = None

        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # todo: YOUR CODE HERE
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        
        t = np.array(t).astype(float)
        f_t = np.array(f_t).astype(float)
        s_t = np.array(s_t).astype(float)
        return t, f_t, s_t
    
    def generate_quar_chirp(self, amplitude, period, duration, start_freq, end_freq, init_phase):
        """
        Generate a quadratic chirp signal.

        Args:
            amplitude (float): Amplitude of the generated signal.
            period (float): Period of the generated signal (s).
            duration (float): Duration of the generated signal (s).
            start_freq (float): Start frequency of the generated signal (Hz).
            end_freq (float): End frequency of the generated signal (Hz).
            init_phase (float): Initial phase of the generated signal (radius).

        Returns:
            t   numpy.array: Array of timestamps in seconds. Data type must be float.
            f_t numpy.array: Array of generated frequency values. Data type must be float.
            s_t numpy.array: Array of generated signal values. Daíta type must be float.
        
        >>> gen = task_1_2(1000)
        >>> t, f_t, s_t = gen.generate_quar_chirp(amplitude=1.0, period=3, duration=10, start_freq=1, end_freq=10, init_phase=0)
        >>> np.round(t[10], 5)
        0.01
        >>> np.round(f_t[10], 5)
        1.0001
        >>> np.round(s_t[10], 5)
        0.99803
        
        """
        t = None
        f_t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # todo: YOUR CODE HERE
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        t = np.array(t).astype(float)
        f_t = np.array(f_t).astype(float)
        s_t = np.array(s_t).astype(float)
        return t, f_t, s_t
    
    def visualize(self):
        """
        Visualize the generated signals.
        """
        # Generate the linear chirp signal
        t, f_t, s_t = self.generate_linear_chirp(amplitude=1, period=1, duration=2, start_freq=1, end_freq=10, init_phase=0)
        plot1D_single(t, s_t, '2-1 L-Chirp', 'Time (s)', 'Amplitude', grid=True)
        plot1D_single(t, f_t, '2-1 f_t', 'Time (s)', 'Frequency (Hz)', grid=True)
        
        # Generate the quadratic chirp signal
        t, f_t, s_t = self.generate_quar_chirp(amplitude=1, period=3, duration=10, start_freq=1, end_freq=10, init_phase=0)
        plot1D_single(t, s_t, '2-2 Q-Chirp', 'Time (s)', 'Amplitude', grid=True)
        plot1D_single(t, f_t, '2-2 f_t', 'Time (s)', 'Frequency (Hz)', grid=True)
