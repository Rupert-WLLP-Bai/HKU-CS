import numpy as np
from vis import *


class task_1_3:
    def __init__(self, fs=1000):
        """
        Initialize the class with a specific sampling rate.

        Args:
            fs (int): Sampling rate in Hz. Defaults to 1000Hz.
        """
        self.fs = fs

    def generate_am_signal(self, Ac: float, mu: float, fm: float, fc: float, phase: float):
        """
        Generate an amplitude-modulated (AM) signal.

        Args:
            Ac (float): Amplitude of the carrier wave.
            mu (float): Modulation index (0 < mu <= 1).
            fm (float): Frequency of the message signal (Hz).
            fc (float): Frequency of the carrier signal (Hz).
            phase (float): Initial phase of the carrier wave (radians).

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of message signal values. Data type must be float.
            numpy.array: Array of carrier signal values. Data type must be float.
            numpy.array: Array of modulated signal values. Data type must be float.

        Note:
            - The returned signal arrays must strictly be of type float.
            - Time range: 0 <= t < 2 seconds.

        Example:
            >>> gen = task_1_3(1000)
            >>> t, m_t, c_t, s_t = gen.generate_am_signal(Ac=1, mu=0.5, fm=5, fc=50, phase=0)
            >>> np.round(t[10], 5), np.round(m_t[10], 5), np.round(c_t[10], 5), np.round(s_t[10], 5)
            (0.01, 0.95106, -1.0, -1.47553)
        """
        
        t = None
        m_t = None
        c_t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # Generate the time vector from 0 (inclusive) to 2 seconds (exclusive)
        t = np.arange(0, 2, 1/self.fs)
        
        # Generate the message signal m_t as a cosine wave at frequency fm.
        m_t = np.cos(2 * np.pi * fm * t)
        
        # Generate the carrier signal c_t with amplitude Ac, frequency fc and initial phase.
        c_t = Ac * np.cos(2 * np.pi * fc * t + phase)
        
        # Generate the amplitude modulated signal: s(t) = (1 + mu * m(t)) * c(t)
        s_t = (1 + mu * m_t) * c_t
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Ensure all arrays are of type float
        t = np.array(t).astype(float)
        m_t = np.array(m_t).astype(float)
        c_t = np.array(c_t).astype(float)
        s_t = np.array(s_t).astype(float)

        return t, m_t, c_t, s_t
    
    def visualize(self):
        t, m_t, c_t, s_t = self.generate_am_signal(Ac=1, mu=0.5, fm=5, fc=50, phase=0)
        plot1D_multiple(t, [m_t, c_t, s_t], ['Message Signal', 'Carrier Signal', 'Modulated Signal'], '3 AM', 'Time (s)', 'Amplitude')
