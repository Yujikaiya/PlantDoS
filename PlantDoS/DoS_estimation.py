import numpy as np

class dos_estimation(object):
    
    def __init__(self, energy_max, energy_min, energy_level):
        self.energy_max = energy_max + 1e-10
        self.energy_min = energy_min
        self.width = energy_max - energy_min + 1e-10
        self.energy_level = energy_level
        
        energy_state = np.zeros(energy_level)
        for i in range(energy_level):
            low = energy_min + i*self.width/energy_level
            high = energy_min + (i+1)*self.width/energy_level
            energy_state[i] = (high+low)/2
            
        self.energy_state = energy_state

    def dos_from_true(self,energy_data):
        true_DoS = np.zeros(self.energy_level)
        for energy in energy_data:
            id_num = np.floor((energy-self.energy_min)/(self.width/self.energy_level)).astype(int)
            true_DoS[id_num] = true_DoS[id_num]+1
        true_DoS = true_DoS/sum(true_DoS)
        
        return true_DoS
    
    def dos_from_random(self,energy_data):
        rs_DoS = np.zeros(self.energy_level)
        for energy in energy_data:
            id_num = np.floor((energy-self.energy_min)/(self.width/self.energy_level)).astype(int)
            rs_DoS[id_num] = rs_DoS[id_num]+1
        rs_DoS = rs_DoS/sum(rs_DoS)

        return rs_DoS
    
    def dos_from_epa(self, betas, EPA_samples, energy_column, mhm_iter):
        bnum = len(betas)
        estdists = self.make_histogram(betas, EPA_samples, energy_column)   #get histgram of energy
        f = np.zeros(bnum)                                                  #initial free energy
        estsum = np.sum(estdists,axis = 0)
        for fiter in range(mhm_iter):                                       #iteration of calculating f and DoS by MHM
            es_DoS = np.zeros(self.energy_level)                 
            for energy_index in range(self.energy_level):
                res = 0
                for beta_index in range(bnum):
                    res = res + sum(estdists[beta_index,:]) * np.exp(-betas[beta_index] * self.energy_state[energy_index] + f[beta_index])
                es_DoS[energy_index] = estsum[energy_index]/res

            for beta_index in range(bnum):
                f[beta_index] = -np.log(np.dot(es_DoS, np.exp(-betas[beta_index]*self.energy_state)))

        es_DoS = es_DoS/np.sum(es_DoS)

        return es_DoS, f
    
    def make_histogram(self, betas, EPA_samples, energy_column):
        bnum = len(betas)
        estdists = np.zeros([bnum,self.energy_level])
        for beta_iter in range(bnum):
            selected_df = EPA_samples[EPA_samples['beta']==betas[beta_iter]]
            for energy_value in selected_df[energy_column]:
                id_num = np.floor((energy_value-self.energy_min)/self.width*self.energy_level).astype(int) #assign energy level number
                estdists[beta_iter,id_num] = estdists[beta_iter,id_num]+1
        return estdists
        