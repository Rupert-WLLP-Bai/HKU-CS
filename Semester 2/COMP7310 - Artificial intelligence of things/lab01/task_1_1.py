import numpy as np
from vis import *

class task_1_1:
    def __init__(self, fs=1000):
        """
        Initialize the class with a specific sampling rate.

        Args:
            fs (int): Sampling rate in Hz. Defaults to 1000Hz.
        """
        self.fs = fs
    
    # TODO: Implement this function
    def generate_signal_1(self):
        """
        Generate the first signal: a pure tone with a specified frequency and phase offset.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated signal values. Data type must be float.

        Note:
            - The returned signal array must strictly be of type float.

        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_1()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, 0.6046)
        """  
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        t = np.linspace(-1, 1, 2 * self.fs, endpoint=False)
        # (-0.25, -0.5) (0,0.5) (1/12, 1)
        # y = sin(ax + b) -> a = 4*pi, b = pi/6
        s_t = np.sin(4*np.pi*t + np.pi/6)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(float)
        t = np.array(t).astype(float)
        return t, s_t

    # TODO: Implement this function
    def generate_signal_2(self):
        """
        Generate the second signal: a combination of rectangle and triangle waves.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated signal values. Data type must be float.

        Note:
            - The returned signal array must strictly be of type float.

        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_2()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, 0.0)
        """
        
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # Generate the time axis: from -1 to 1, with 2*self.fs sample points, interval is 1/self.fs
        t = np.linspace(-1, 1, 2*self.fs, endpoint=False)
        
        # Initialize the signal to all zeros
        s_t = np.zeros_like(t)
        
        # -------------------------------
        # Rectangle wave part
        # -------------------------------
        # Corresponding points: (-0.7,0) -> (-0.7,1) -> (-0.3,1) -> (-0.3,0)
        # In the interval [-0.7, -0.3), the rectangle wave is 1
        mask_rect = (t >= -0.7) & (t < -0.3)
        s_t[mask_rect] = 1.0
        
        # -------------------------------
        # Triangle wave part
        # -------------------------------
        # Corresponding points: (-0.3,0) -> (0.5,1) -> (0.7,0)
        # Rising segment: t in [-0.3, 0.5), rising from 0 to 1
        mask_tri_rise = (t >= -0.3) & (t < 0.5)
        s_t[mask_tri_rise] = (t[mask_tri_rise] + 0.3) / 0.8  # 0.8 = 0.5 - (-0.3)
        
        # Falling segment: t in [0.5, 0.7), falling from 1 to 0
        mask_tri_fall = (t >= 0.5) & (t < 0.7)
        s_t[mask_tri_fall] = (0.7 - t[mask_tri_fall]) / 0.2  # 0.2 = 0.7 - 0.5
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(float)
        t = np.array(t).astype(float)
        return t, s_t
        
    # TODO: Implement this function
    def generate_signal_3(self):
        """
        
        Generate the third signal: a complex signal based on real and imaginary parts.

        Returns:
            numpy.array: Array of timestamps in seconds. Data type must be float.
            numpy.array: Array of generated complex signal values. Data type must be np.complex64.

        Note:
            - The returned signal array must strictly be of type np.complex64.
            
        Example:
            >>> gen = task_1_1(1000)
            >>> t, s_t = gen.generate_signal_3()
            >>> np.round(t[10], 5), np.round(s_t[10], 5)
            (-0.99, (0.99211+0.12533j))
        """
        t = None
        s_t = None
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        t = np.linspace(-1, 1, 2*self.fs, endpoint=False)
        # real part = cos(4*pi*t)
        # imaginary part = sin(4*pi*t)
        s_t = np.cos(4*np.pi*t) + 1j*np.sin(4*np.pi*t)
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_t = np.array(s_t).astype(np.complex64)
        t = np.array(t).astype(float)
        return t, s_t
    
    def visualize(self):
        # Generate the first signal
        t, s_t = self.generate_signal_1()
        plot1D_single(t, s_t, '1-1', 'Time (s)', 'Amplitude')
        
        # Generate the second signal
        t, s_t = self.generate_signal_2()
        plot1D_single(t, s_t, '1-2', 'Time (s)', 'Amplitude')
        
        # Generate the third signal
        t, s_t = self.generate_signal_3()
        plot1D_multiple(t, [np.real(s_t), np.imag(s_t)], ['Real', 'Imaginary'], '1-3', 'Time (s)', 'Amplitude')
    