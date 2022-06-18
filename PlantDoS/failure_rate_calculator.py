import numpy as np
import pandas as pd


class failure_rate_calculator(object):
    
    #fixed parameter
    R = 8.314
    T_ref = 90 + 273.71

    #Parameter distribution
    Es = [33.3*1000, 35.3*1000, 38.9*1000, 44.8*1000]   #estimated activation energy
    E_SEs = [0.3*1000, 0.5*1000, 1.5*1000, 1.8*1000]    #estimated activation energy standard error
    k_90s = [57.9/100, 2.7/100, 0.865/100, 1.63/100]    #estimated rate constant
    k_SEs = [0.7/100, 0.06/100, 0.004/100, 0.11/100]    #estimated rate constant standard error

    def __init__(self,reaction_simulator):
        self.reaction_simulator = reaction_simulator

    #generate kinetic paramter from parameter distributions
    def gen_param(self,sample_num=1000):
        gen_Es = np.zeros([len(self.Es),sample_num])
        gen_As = np.zeros([len(self.k_90s),sample_num])
        for sample_iter in range(len(self.Es)):
            gen_k_90 = np.random.normal(loc=self.k_90s[sample_iter], scale=self.k_SEs[sample_iter], size=sample_num)
            gen_E = np.random.normal(loc=self.Es[sample_iter], scale=self.E_SEs[sample_iter], size=sample_num)
            gen_Es[sample_iter,:] = gen_E
            gen_A = gen_k_90 * np.exp(gen_E/(self.R*self.T_ref)) #calculate pre-exponential factors
            gen_As[sample_iter,:] = gen_A
        return gen_Es, gen_As

    #monte-carlo simulation
    def monte_carlo_sim(self,A_conc,B_eq,Temp,Time,sample_num=1000):
        gen_Es, gen_As = self.gen_param(sample_num=sample_num)
        results = []
        for mont_iter in range(sample_num):
            gen_A = gen_As[:,mont_iter]
            gen_E = gen_Es[:,mont_iter]
            self.reaction_simulator.run_rxn(A_conc,B_eq,Temp,Time, gen_A, gen_E) #run single simulation
            deterministic_results = self.reaction_simulator.get_product_ratio()
            deterministic_results.append(self.reaction_simulator.calc_product_yield())
            deterministic_results.append(self.reaction_simulator.clac_Efactor())
            results.append(deterministic_results)
            result_df = pd.DataFrame(results)
        return result_df

    #calculate failure rate
    def calc_failure_rate(self,result_df, limit_set):
        trial_num = len(result_df)
        SM1_fail = result_df[result_df[0] > limit_set[0]]   #SM1: 2,4-difluoronitrobenzene
        SM1_prob = len(SM1_fail)/trial_num
        SM2_fail = result_df[result_df[1] > limit_set[1]]   #SM2: pyrrolidine
        SM2_prob = len(SM2_fail)/trial_num
        Iso_fail = result_df[result_df[3] > limit_set[2]]   #Iso: para-substituted
        Iso_prob = len(Iso_fail)/trial_num
        Bis_fail = result_df[result_df[4] > limit_set[3]]   #Bis: bis-substituted
        Bis_prob = len(Bis_fail)/trial_num
        all_prob = len(result_df[(result_df[0] > limit_set[0]) #Total failure: at least one of the impuirites exceeds the limit
                 | (result_df[1] > limit_set[1])
                 | (result_df[3] > limit_set[2])
                 | (result_df[4] > limit_set[3])
                 ])/len(result_df)
        return [all_prob, SM1_prob, SM2_prob, Iso_prob, Bis_prob]


