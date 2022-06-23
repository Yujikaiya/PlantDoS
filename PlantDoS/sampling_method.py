import numpy as np
import pandas as pd
import pickle
import copy

#For base example, random sampling was used
class random_sampling(object):
    
    column_name = ['Conc', 'SM2_eq', 'Temp', 'Time', 'Total_failure', 'SM1_failure', 'SM2_failure', 'Iso_failure', 'Bis_failure', 'Yield', 'E_factor', 'Material_cost', 'STY']
    
    def __init__(self, true_distribution_df, range_num, random_num):
        self.true_distribution_df = true_distribution_df
        self.range_num = range_num
        self.random_num = random_num
        
    def random_sample_from_df(self, trial_num):

        random_sampling_all_trial = pd.DataFrame()
        for trial in range(trial_num):    
            random_sampling_df = pd.DataFrame(columns = self.column_name)
            for siter in range(self.random_num):
                param = np.random.randint(low=0,high=self.range_num,size=4)
                df_idx = param[0]*self.range_num**3+param[1]*self.range_num**2+param[2]*self.range_num+param[3]  #index of true_distribution_df(this is only feasible only when true_distribution_df is sampled by calc_true_ditribution.ipynb)
                results = self.true_distribution_df.iloc[df_idx,:]
                random_sampling_df = random_sampling_df.append(results)
            random_sampling_df['trial'] = trial
            random_sampling_all_trial = random_sampling_all_trial.append(random_sampling_df)

        return random_sampling_all_trial
    
class population_annealing(object):
    
    column_name = ['Conc', 'SM2_eq', 'Temp', 'Time', 'Total_failure', 'SM1_failure', 'SM2_failure', 'Iso_failure', 'Bis_failure', 'Yield', 'E_factor', 'Material_cost', 'STY']
    
    def __init__(self,true_distribution_df,range_num,num_population,mcmc_num,walk_step,betas):
        self.true_distribution_df = true_distribution_df
        self.range_num = range_num
        self.num_population = num_population
        self.mcmc_num = mcmc_num
        self.walk_step = walk_step
        self.betas = betas

    def resampling(self, curr_beta, old_beta, curr_points):
        weights = np.zeros(self.num_population)
        z_score = np.zeros(self.num_population) 
        for siter in range(self.num_population):
            df_idx = curr_points[siter,0]*self.range_num**3+curr_points[siter,1]*self.range_num**2+curr_points[siter,2]*self.range_num+curr_points[siter,3]
            results = self.true_distribution_df.iloc[df_idx,:]
            z_score[siter] = results["Total_failure"] #optimized outputs
            weights[siter] = np.exp(-(curr_beta-old_beta)*z_score[siter]) #minimize
        weights = weights / np.sum(weights)
        rR = np.random.choice(np.arange(0,self.num_population),p=weights,size=self.num_population,replace=True)
        newpoints = curr_points[rR,:]
        
        return newpoints
    
    def mcmc(self,pos,beta,old_results):
        old_score = old_results["Total_failure"] 
        newnode = copy.copy(pos)
        newnode = self.random_walk(newnode, pos)
        df_idx = newnode[0]*self.range_num**3+newnode[1]*self.range_num**2+newnode[2]*self.range_num+newnode[3]
        results = self.true_distribution_df.iloc[df_idx,:]
        _scores = results["Total_failure"] 
        q = np.exp(-beta*(_scores-old_score))     #minimize    
        if (q > np.random.rand(1)):
            pos = newnode
            flag = 1
            old_results = results
        else:
            flag = 0
        
        return old_results, results, pos, flag
    
    def random_walk(self, newnode, pos):
        move = (np.random.randint(0,2)*2 - 1) * (np.random.randint(0,self.walk_step) + 1) #random selection of step 
        input_idx = np.random.randint(0,4)                                                #random selection of input
        new_pos = pos[input_idx]+move
        if new_pos<0 or new_pos>(self.range_num-1):
            newnode[input_idx] = (self.range_num-1) - np.mod(new_pos,(self.range_num-1))
        else:
            newnode[input_idx] = new_pos
        
        return newnode
    
    def pupolation_annealing_from_df(self,trial_num):
        
        EPA_sampling_all_trial = pd.DataFrame()
        _EPA_sampling_all_trial = pd.DataFrame() #save rejected samples

        for trial in range(trial_num):
            curr_points = np.random.randint(0,self.range_num,(self.num_population,4))      #random sampling of intial points
            old_beta=0
            EPA_sampling_df = pd.DataFrame(columns = self.column_name)
            _EPA_sampling_df = pd.DataFrame(columns = self.column_name)     #save rejected samples
            flag_list = []
            beta_list = []
            
            #increasing inverse temperature
            for beta_iter in range(len(self.betas)):
                #set inverse temperature
                beta = self.betas[beta_iter]
                #resamping
                curr_points = self.resampling(beta, old_beta, curr_points)
                #Metropolis                            
                for siter in range(self.num_population):
                    df_idx = curr_points[siter,0]*self.range_num**3 + curr_points[siter,1]*self.range_num**2 + curr_points[siter,2]*self.range_num + curr_points[siter,3]
                    old_results = self.true_distribution_df.iloc[df_idx,:]
                    pos = curr_points[siter,:]
                    for i in range(self.mcmc_num):
                        old_results, results, pos, flag = self.mcmc(pos,beta,old_results)
                        EPA_sampling_df = EPA_sampling_df.append(old_results)        #old_results is updated only if accepted
                        _EPA_sampling_df = _EPA_sampling_df.append(results)          #results of suggested points 
                        flag_list.append(flag)
                        beta_list.append(beta)                    
                    curr_points[siter,:] = pos
                oldbeta = beta
            EPA_sampling_df['beta'] = beta_list
            _EPA_sampling_df['beta'] = beta_list
            EPA_sampling_df['flag'] = flag_list
            _EPA_sampling_df['flag'] = flag_list
            EPA_sampling_df['trial'] = trial
            _EPA_sampling_df['trial'] = trial
            
            EPA_sampling_all_trial = EPA_sampling_all_trial.append(EPA_sampling_df)
            _EPA_sampling_all_trial = _EPA_sampling_all_trial.append(_EPA_sampling_df)
            
            
        return EPA_sampling_all_trial, _EPA_sampling_all_trial
    
