import pandas as pd
import numpy as np

def classify_reaction(series : pd.Series, threshold : float) -> np.ndarray:
    """
        Takes in the return column (ret_1d,3d,etc) 

        Returns a column of a classification of the return to
        1 being "Up", -1 being "Down" or 0 "No Change"
    """
    
    return np.select(
        [series > threshold, series < -threshold],
        [1, -1],
        default=0
    )