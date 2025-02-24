import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks

class task_3_3:
    def __init__(self, data_root="./data/"):
        """
        Initializes the task_3_3 class, loading signal data from pickle files.

        Args:
            data_root (str): Directory path where 'task_3_3.pickle' is stored. Defaults to './data/'.

        Attributes:
            m1, m2 (np.ndarray): Motion sensor signals (1 Hz, 10 samples).
            t1, t2 (np.ndarray): Temperature sensor signals (1 Hz, 10 samples).
            s1, s2 (np.ndarray): Sound sensor signals (16 kHz, 160,000 samples).
            s, p (np.ndarray): Sound signal and music pattern (16 kHz, 160,000 and 16,000 samples).
        """
        with open(osp.join(data_root, 'task_3_3.pickle'), 'rb') as f:
            data = pickle.load(f)
        self.m1, self.m2 = data['m1'], data['m2']
        self.t1, self.t2 = data['t1'], data['t2']
        self.s1, self.s2 = data['s1'], data['s2']
        self.s, self.p = data['s'], data['p']

    def get_pcc(self, s1, s2):
        """
        Calculate the Pearson Correlation Coefficient (PCC) between two signals using np.correlate.

        Args:
            s1 (np.ndarray): First signal array.
            s2 (np.ndarray): Second signal array, same length as s1.

        Returns:
            np.float64: PCC value between -1 and 1.

        Examples:
            >>> t = task_3_3()
            >>> s1 = np.array([1, 2, 3])
            >>> s2 = np.array([2, 4, 6])
            >>> pcc = t.get_pcc(s1, s2)
            >>> np.isclose(pcc, 1.0, atol=1e-6)
            True
            >>> s3 = np.array([3, 2, 1])
            >>> pcc = t.get_pcc(s1, s3)
            >>> np.isclose(pcc, -1.0, atol=1e-6)
            True
        """
        pcc = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Implement PCC using np.correlate:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        pcc = np.float64(pcc)
        return pcc

    def check_motion_sensors(self, m1, m2):
        """
        Check if motion sensors are negatively correlated, indicating proper function.

        Args:
            m1 (np.ndarray): Living room motion sensor signal.
            m2 (np.ndarray): Bedroom motion sensor signal.

        Returns:
            tuple: (PCC value as np.float64, Boolean indicating proper functio).

        Examples:
            >>> t = task_3_3()
            >>> pcc, res = t.check_motion_sensors(t.m1, t.m2)
            >>> isinstance(pcc, np.float64), isinstance(res, bool)
            (True, True)
        """
        pcc, res = None, None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Use get_pcc to compute PCC, set res True if PCC < 0
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        pcc = np.float64(pcc)
        res = bool(res)
        return pcc, res

    def check_temperature_sensors(self, t1, t2):
        """
        Check if temperature sensors are positively correlated, indicating proper function.

        Args:
            t1 (np.ndarray): Living room temperature sensor signal.
            t2 (np.ndarray): Bedroom temperature sensor signal.

        Returns:
            tuple: (PCC value as np.float64, Boolean proper function).

        Examples:
            >>> t = task_3_3()
            >>> pcc, res = t.check_temperature_sensors(t.t1, t.t2)
            >>> isinstance(pcc, np.float64), isinstance(res, bool)
            (True, True)
        """
        pcc, res = None, None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: Use get_pcc to compute PCC, set res True if PCC > 0
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        pcc = np.float64(pcc)
        res = bool(res)
        return pcc, res

    def sync_event_signals(self, s1, s2):
        """
        Synchronize two sound sensor signals and check if delay allows alarm triggering.

        Args:
            s1 (np.ndarray): Living room sound sensor signal.
            s2 (np.ndarray): Front door sound sensor signal.

        Returns:
            tuple: (Delay in seconds as np.float64, Boolean indicating if alarm can be triggered).

        Notes:
            - Sampling rate: 10 Hz.
            - Threshold: 0.1 seconds.

        Examples:
            >>> t = task_3_3()
            >>> delay, res = t.sync_event_signals(t.s1, t.s2)
            >>> delay >= 0, isinstance(delay, np.float64), isinstance(res, bool)
            (True, True, True)
        """
        delay, res = None, None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: 
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        delay = np.abs(delay).astype(np.float64)
        res = bool(res)
        return delay, res

    def detect_music_patterns(self, s, p):
        """
        Detect the starting index of a music pattern in a sound signal using PCC. Sample rate is 16 kHz.

        Args:
            s (np.ndarray): Living room sound signal
            p (np.ndarray): Music pattern.

        Returns:
            int: Starting index of the strongest pattern match.

        Examples:
            >>> t = task_3_3()
            >>> idx = t.detect_music_patterns(t.s, t.p)
            >>> idx >= 0, isinstance(idx, int)
            (True, True)
        """
        start_idx = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO: 
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        start_idx = int(start_idx)
        return start_idx

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)