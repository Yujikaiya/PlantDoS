import numpy as np
import pandas as pd

class hellinger_distance(object):
    
    def __init__(self, true_DoS):
        self.true_DoS = true_DoS

    def hellinger_distance(self, estimate_DoS, threshold=1.0):
        num_DoS_points = len(self.true_DoS)
        threshold_index = int(threshold * num_DoS_points)
        if sum(self.true_DoS[:threshold_index]) != 0:
            DoS_1 = self.true_DoS[:threshold_index]/sum(self.true_DoS[:threshold_index])
        else:
            DoS_1 = self.true_DoS[:threshold_index]
        if sum(estimate_DoS[:threshold_index]) != 0:
            DoS_2 = estimate_DoS[:threshold_index]/sum(estimate_DoS[:threshold_index])
        else:
            DoS_2 = estimate_DoS[:threshold_index]
        hellinger = 0
        for true_x, estimate_x in zip(DoS_1, DoS_2):
            hellinger += (true_x**0.5 - estimate_x**0.5)**2
        return hellinger

    def hellinger_for_plot(self, estimate_DoS_list, thres_list):
        hellinger_array = np.zeros((len(estimate_DoS_list),len(thres_list)))
        for doslist_i, estimate_DoS in enumerate(estimate_DoS_list):
            for threslist_i, thres in enumerate(thres_list):
                hellinger_array[doslist_i,threslist_i] = self.hellinger_distance(estimate_DoS, thres)

        return pd.DataFrame(hellinger_array, columns=thres_list)
    
    