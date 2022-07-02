import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        
    def gray_plot(self, true_distribution_df, energy_variable, gray_variable, energy_levels, gray_levels_num, xlabel, ylabel, title, figname=None, samples=None, betas=None, f=None, imshow=False, vmin=-7, vmax=-3):
        
        #parameter setting for graybox dos estimation
        energy_threshold = energy_levels[-1]
        energy_levels[0] = energy_levels[0] - 1e-10
        energies = true_distribution_df[energy_variable]
        grays = true_distribution_df[gray_variable]
        gray_max = grays[energies<=energy_threshold].max() + 1e-10
        gray_min = grays[energies<=energy_threshold].min()
        gray_width = gray_max - gray_min
            
        #gray box dos estimation
        if samples is not None:
            energies = samples[energy_variable]
            grays = samples[gray_variable]
            
            if 'beta' in samples.columns:
                estdists = self.graybox_epa_dos(energy_levels, gray_levels_num, energies, grays, gray_min, gray_width, energy_threshold, betas, f, samples, energy_variable)
            else:
                estdists = self.graybox_dos(energy_levels, gray_levels_num, energies, grays, gray_min, gray_width, energy_threshold)
        else:
            energies = true_distribution_df[energy_variable]
            grays = true_distribution_df[gray_variable]
            estdists = self.graybox_dos(energy_levels, gray_levels_num, energies, grays, gray_min, gray_width, energy_threshold)

        #dataframe for counter plot
        gray_label_list = [round(gray_min+i*gray_width/gray_levels_num,3) for i in range(gray_levels_num)]
        energy_label_list = ['{}%'.format(round(i*100,1)) for i in energy_levels[1:]]
        counter_df = self.make_counter_df(estdists, gray_label_list, energy_label_list, imshow=imshow)

        #plot
        if imshow==False:
            plt.figure(figsize=(12,8)) 
            ax = sns.heatmap(counter_df,annot=True,cmap="Blues",vmin=vmin,vmax=vmax)
        
        else:
            counter_plot = counter_df.to_numpy()
            xticks = [i for i in range(len(gray_label_list))]
            yticks = [i for i in range(len(energy_label_list))]
            fig, ax = plt.subplots(figsize=(12,8))
            im = ax.imshow(counter_plot, cmap="Blues",vmin=vmin,vmax=vmax, interpolation='bicubic', aspect=len(gray_label_list)/len(energy_label_list))
            fig.colorbar(im, ax=ax)
            ax.set_xticks(xticks)
            ax.set_xticklabels(gray_label_list)
            ax.set_yticks(yticks)
            ax.set_yticklabels(energy_label_list)
            
        ax.invert_yaxis()
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        if figname is not None:
            plt.savefig(figname)

        
    def make_counter_df(self, estdists, gray_label_list, energy_label_list, imshow=False):
        if imshow==True:
            estdists[estdists==0] = 1.0e-10     #avoid error when imshow is used
        estdists = np.log10(estdists)
        counter_df = pd.DataFrame(estdists)
        counter_df = counter_df.iloc[:-1,:-1] #remove out of scope
        counter_df.index = energy_label_list
        counter_df.columns = gray_label_list
        
        return counter_df
        
    def graybox_dos(self, energy_levels, gray_levels_num, energies, grays, gray_min, gray_width, energy_threshold):
        energy_levels_num = len(energy_levels) - 1
        estdists = np.zeros([energy_levels_num + 1, gray_levels_num+1])
        for sc1, sc2 in zip(energies,grays):
            for energy_level_index in range(energy_levels_num):
                if energy_levels[energy_level_index] < sc1 <= energy_levels[energy_level_index+1]:
                    id_num1 = energy_level_index       #index of descrete space of energy
                if sc1 > energy_threshold:
                    id_num1 = energy_levels_num        #out of range (bigger than energy threshold)
            id_num2 = np.floor((sc2-gray_min)/(gray_width/gray_levels_num)).astype(int) #index of descrete space of a graybox varialble
            if id_num2 >= gray_levels_num:
                id_num2 = gray_levels_num              #out of range (bigger than max graybox values within the thres energy)
            elif id_num2 < 0:
                id_num2 = gray_levels_num              #out of range (smaller than min graybox values within the thres energy)
            estdists[id_num1,id_num2] = estdists[id_num1,id_num2]+1
        estdists = estdists/np.sum(estdists)
        
        return estdists
    
    def graybox_epa_dos(self, energy_levels, gray_levels_num, energies, grays, gray_min, gray_width, energy_threshold, betas, f, samples, energy_variable):
        energy_levels_num = len(energy_levels) - 1
        gray_step_value = gray_width/gray_levels_num
        gray_step_values = [gray_min+i*gray_step_value for i in range(gray_levels_num + 1)]
        weight = self.weight_calc(betas, f, samples, energy_variable)
        estdists = np.zeros([energy_levels_num + 1, gray_levels_num+1])
        for sc1, sc2, wg in zip(energies,grays,weight):
            for energy_level_index in range(energy_levels_num):
                if energy_levels[energy_level_index] < sc1 <= energy_levels[energy_level_index+1]:
                    id_num1 = energy_level_index       #index of descrete space of energy
                if sc1 > energy_threshold:
                    id_num1 = energy_levels_num        #out of range (bigger than energy threshold)
            id_num2 = np.floor((sc2-gray_min)/(gray_width/gray_levels_num)).astype(int) #index of descrete space of a graybox varialble
            if id_num2 >= gray_levels_num:
                id_num2 = gray_levels_num              #out of range (bigger than max graybox values within the thres energy)
            elif id_num2 < 0:
                id_num2 = gray_levels_num              #out of range (smaller than min graybox values within the thres energy)
            estdists[id_num1,id_num2] = estdists[id_num1,id_num2]+wg  #sum weight
        estdists = estdists/np.sum(estdists)

        return estdists
    
    def weight_calc(self, betas, f, EPA_samples, energy_column):
        
        R = np.zeros([self.energy_level,len(betas)])
        estdists = self.make_histogram(betas, EPA_samples, energy_column)
        for i in range(self.energy_level):
            for j in range(len(betas)):
                R[i][j] = sum(estdists[j,:])*np.exp(-betas[j]*self.energy_state[i] + f[j])
            R[i] = R[i]/np.sum(R[i])
            
        weights = np.zeros(len(EPA_samples))
        for i, row in enumerate(EPA_samples.iterrows()):
            beta_index = np.where(betas==row[1]['beta']) 
            energy_index = int((row[1][energy_column]-self.energy_min)/self.width*len(self.energy_state))
            weights[i] = (R[energy_index][beta_index]*np.exp(betas[beta_index]*self.energy_state[energy_index]-f[beta_index]))
            
        return weights
    
    def observable_dos(self, observables, observable_levels_num, observable_max, observable_min):
        
        energy_max = observable_max+0.00001
        energy_min = observable_min
        width = energy_max - energy_min

        observable_dos = np.zeros(observable_levels_num)
        for score in observables:
            id_num = np.floor((score-energy_min)/(width/observable_levels_num)).astype(int)
            observable_dos[id_num] = observable_dos[id_num]+1
        observable_dos = observable_dos/sum(observable_dos)

        return observable_dos
    
    def observable_epa_dos(self, observables, weights, observable_levels_num, observable_max, observable_min):

        energy_max = observable_max+0.00001
        energy_min = observable_min
        width = energy_max - energy_min

        observable_dos = np.zeros(observable_levels_num)
        for score, weight in zip(observables, weights):
            id_num = np.floor((score-energy_min)/(width/observable_levels_num)).astype(int)
            observable_dos[id_num] = observable_dos[id_num] + weight
        observable_dos = observable_dos/sum(observable_dos)

        return observable_dos
    
    
