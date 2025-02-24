import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

class task_3_2:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_3_2 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            br_fn (str): Filename for the breath signal (task_3_2_1).
            ecg_fn (str): Filename for the ECG signal (task_3_2_2).
            ecg_data (dict): Loaded data for the ECG signal.
            br_data (dict): Loaded data for the breath signal.
        """
        self.data_root = data_root
        
        self.br_fn = "task_3_2_1.pickle"
        self.ecg_fn = "task_3_2_2.pickle"
        
        
        with open(osp.join(self.data_root, self.ecg_fn), "rb") as f:
            self.ecg_data = pickle.load(f)
        with open(osp.join(self.data_root, self.br_fn), "rb") as f:
            self.br_data = pickle.load(f)
        
    def get_br_1(self):
        """
        Calculate the breathing rate from the breath signal.

        Returns:
            br (float64): The breathing rate in breaths per minute (BPM).
        
        >>> test = task_3_2()
        >>> br = test.get_br_1()
        >>> br != 0
        True
        """
        s_t = self.br_data["values"] # signal values
        fs = self.br_data["fs"] # sampling frequency
        
        br = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Make sure br is a float64
        if isinstance(br, np.ndarray):
            if br.size > 1:
                br = br[0]
            br = br.item()
        if isinstance(br, list):
            if len(br) > 1:
                br = br[0]
        br = float(br)
        return br
    
    def get_br_2(self):
        """
        Calculate the breathing rate over time from the breath signal.
        
        You should use choose the window length as short as possible with 
        time resolution of 1s. 
        Your window length should be chosen from [1, 10]s and 
        we assume the window length here is an integer.

        Returns:
            - b_t (np.float64): The breathing rate b(t) in BPM.
            - window_length (int): The length of the window used to compute the breathing rate.
            - window_step (float): The step size used to compute the breathing rate.
        >>> test = task_3_2()
        >>> b_t, wl, ws = test.get_br_2()
        >>> b_t.any() != 0, wl != 0, ws != 0
        (True, True, True)
        """
        s_t = self.br_data["values"] # signal values
        fs = self.br_data["fs"] # sampling frequency
        
        b_t = np.array([], dtype=np.float64) # breathing rate over time
        window_length = 0 # TODO: Set the window length (in seconds)
        window_step = 0.0 # TODO: Set the window step (in seconds)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Make sure br is a float64
        b_t = np.array(b_t).astype(np.float64)
        window_length = int(window_length)
        window_step = float(window_step)
        return b_t, window_length, window_step
    
    
    
    def get_hr_1(self):
        """
        Determine the heart rate from the ECG signal over time.

        Returns:
            - h_t (float64): The heart rate h(t) in BPM.
        
        >>> test = task_3_2()
        >>> h_t = test.get_hr_1()
        >>> h_t.any() != 0
        True
        """
        s_t = self.ecg_data["values"] # signal values
        fs = self.ecg_data["fs"] # sampling frequency
        
        h_t = np.array([], dtype=np.float64)
        window_length = 5 # Window length in seconds
        window_step = 2 # Window step in seconds
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Make sure hr is a float64
        h_t = np.array(h_t).astype(np.float64)
        return h_t
    
    def get_hr_2(self):
        """
        Determine the heart rate from the ECG signal over time.
        
        You should adjust your window_length and window_step to make sure 
            - the frequency resolution is 0.5 Hz, and 
            - time resolution is 0.1s

        Returns:
            - h_t (float64): The heart rate h(t) in BPM.
            - window_length (float): The length of the window used to compute the heart rate.
            - window_step (float): The step size used to compute the heart rate.
        
        >>> test = task_3_2()
        >>> h_t, wl, ws = test.get_hr_2()
        >>> h_t.any() != 0, wl != 0, ws != 0
        (True, True, True)
        """
        s_t = self.ecg_data["values"] # signal values
        fs = self.ecg_data["fs"] # sampling frequency
        
        h_t = np.array([], dtype=np.float64)
        
        window_length = 0.0 # TODO: Set the window length in seconds
        window_step = 0.0 # TODO: Set the window step in seconds
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        
        # Make sure hr is a float64
        h_t = np.array(h_t).astype(np.float64)
        h_t = np.array(h_t).astype(np.float64)
        window_length = float(window_length)
        window_step = float(window_step)
        return h_t, window_length, window_step
    

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)