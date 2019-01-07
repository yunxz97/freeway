import numpy as np
from constants import SMALL_NON_ZERO, SMALL_LOG_NON_ZERO

def one_hot(value, classes):
	res = np.zeros(classes, dtype=np.int)
	res[value] = 1
	return res

def to_log_probability(pot, min_val=SMALL_NON_ZERO, max_val=1):
    a = np.clip(pot, min_val, max_val)
    idx = np.where(a == min_val)
    a = np.log(a)
    a[idx] = SMALL_LOG_NON_ZERO
    return a