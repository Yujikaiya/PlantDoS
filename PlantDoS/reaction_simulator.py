import numpy as np
from scipy.integrate import odeint


"""
deterministic model parameters

#fixed parameter
R = 8.314
T_ref = 90 + 273.71

#kinetic parameter(E:activation energy, A:pre-exponential factor)
E_a = 33.3*1000 #2,4-difluoronitrobenzene->ortho-substituted
E_b = 35.3*1000 #2,4-difluoronitrobenzene->para-substituted
E_c = 38.9*1000 #ortho-substituted->bis-substituted
E_d = 44.8*1000 #para-substituted->bis-substituted

A_a = 57.9/100 * np.exp(E_a/(R*T_ref))
A_b = 2.7/100 * np.exp(E_b/(R*T_ref))
A_c = 0.865/100 * np.exp(E_c/(R*T_ref))
A_d = 1.63/100 * np.exp(E_d/(R*T_ref))

Es = [E_a, E_b, E_c, E_d]
As = [A_a, A_b, A_c, A_d]

"""


#Simulater

class reaction_simulator(object):
    """reaction simulator of a nucleophilic aromatic substitution (SnAr) reaction
    
    Notes
    -----
    This simulator is observerd by [Hone]_ et al. The simulator calculated concentrations of all species.
    These concentrations are used to calculate other output parameters. Also, scipy implementation of this
    model was done by [Felton]_ et al.
    
    References
    ----------
    .. [Hone] C. A. Hone et al., React. Chem. Eng., 2017, 2, 103–108. 
       DOI:`10.1039/C6RE00109B <https://doi.org/10.1039/C6RE00109B>`_
    .. [Felton] Kobi Felton et al., Chemistry—Methods, 2021, 1, 116–122. 
       DOI:`10.1002/cmtd.202000051 <https://doi.org/10.1002/cmtd.202000051>`_
    """
    
    # molecular weights (g/mol)
    molecular_weight = [159.09, 71.12, 210.21, 210.21, 261.33]  # molecular weights (g/mol)
    #[2,4-difluoronitrobenzen, pyrrolidine, ortho-substituted, para-substituted, bis-substituted]
    
    def __init__(self):
        self.factors = None
        self.conc_results = None
        self.product_ratio = None
        self.product_yield = None
        self.e_factor = None
        
    #reaction kinetics expressed as ODE
    def rxn(self, Z, t, Temp, As, Es):
        R = 8.314
        k1 = As[0]*np.exp(-Es[0]/(R*Temp))
        k2 = As[1]*np.exp(-Es[1]/(R*Temp))
        k3 = As[2]*np.exp(-Es[2]/(R*Temp))
        k4 = As[3]*np.exp(-Es[3]/(R*Temp))
        
        r1 = k1 * Z[0]*Z[1]             #2,4-difluoronitrobenzene->ortho-substituted
        r2 = k2 * Z[0]*Z[1]             #2,4-difluoronitrobenzene->para-substituted
        r3 = k3 * Z[2]*Z[1]             #ortho-substituted->bis-substituted
        r4 = k4 * Z[3]*Z[1]             #para-substituted->bis-substituted
    
        dSM1dt = - r1 - r2              #change of 2,4-difluoronitrobenzene concentration
        dSM2dt = - r1 - r2 - r3 -r4     #change of pyrrolidine concentration
        dPROdt = r1 - r3                #change of ortho-substituted concentration
        dISOdt = r2 - r4                #change of para-substituted concentration
        dBISdt = r3 + r4                #change of bis-substituted concentration
        
        return [dSM1dt,dSM2dt,dPROdt,dISOdt,dBISdt]
    
    #running simulation
    def run_rxn(self, Molar, SM2_eq, Temp, time, As, Es):
        """
        Molar: initial concentration of 2,4-difluoronitrobenzene (mol/L)
        SM2_eq: equivalent of pyrroridine based on 2,4-difluoronitrobenzene (eq)
        Temp: reaction temperature (K)
        time: reaction time (min)
        """
        SM1 = Molar
        SM2 = Molar * SM2_eq
        Z0 = [SM1, SM2, 0, 0, 0]
        m = 60
        self.conc_results = odeint(self.rxn, Z0, [0,time*m], args=(Temp,As,Es))
        self.factors = [Molar, SM2_eq, Temp-273.71, time]
        
        return self.conc_results
    
    #calculate purity of the reaction mixture(without ethanol)
    def get_product_ratio(self):
        SM1 = self.conc_results[-1,0]
        SM2 = self.conc_results[-1,1]
        PRO = self.conc_results[-1,2]
        ISO = self.conc_results[-1,3]
        BIS = self.conc_results[-1,4]
        SUM = SM1 + SM2 + PRO + ISO + BIS
        SM1 = SM1/SUM
        SM2 = SM2/SUM
        PRO = PRO/SUM
        ISO = ISO/SUM
        BIS = BIS/SUM
        
        self.product_ratio = [SM1, SM2, PRO, ISO, BIS]
        
        return self.product_ratio
    
    #calculate the mol ratio of target product(ortho-substituted) and SM1(2,4-difluoronitorobenzen)
    def calc_product_yield(self):
        init_SM1_conc = self.factors[0]
        self.product_yield = self.conc_results[-1,2]/init_SM1_conc #last PRO conc/initial SM1 conc
        return self.product_yield  
    
    #calculate E-factor
    def clac_Efactor(self):
        rho_eth = 0.789 * 1000  # density of enthanol, g/L (@ 25C)
        term_2 = sum([self.molecular_weight[i] * self.conc_results[-1,i] for i in range(5) if i != 2]) #total mass of side product g/L
        self.e_factor = (rho_eth + term_2) / (self.molecular_weight[2] * self.conc_results[-1,2])
        return self.e_factor



