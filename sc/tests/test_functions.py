import os
import numpy as np
from sc.utils.functions import kendall_constraint

class TestKendallConstraint():
    
    x = np.array([1, 3, 4, 7, 11])
    y = np.array([2, 10, 17, 50, 122]) # x**2+1
    
    def return_one_if_identical(self):
        kendall_constraint(x, y, )
    def return_neg_one_if_opposite(self):
        pass