import torch
from mnl_model import MultipleMNLModel

class MNLEnvironment:
    def __init__(
            self,
            alphas,
            betas,
            el,
            u,
            T,
            ):
        """

        Parameters
        ----------
        T :(int):Time horizon of the experiment
            DESCRIPTION.

        """
        K = alphas.shape[0]
        self.t = 0
        self.T = T
        self.regret = 0
        self.simple_regret = 0
        self.regret_history = []
        self.simple_regret_history = []
        self.price_history = torch.zeros((T,K))
        self.choice_history = torch.zeros(T,dtype = torch.long)
        self.demand_history = torch.zeros((T,K))
        self.model = MultipleMNLModel(
            alphas, 
            betas, 
            el=el,
            u = u,
            )
        
        self.best_price = self.model.max_mean_demand_price()
        self.best_revenue = self.model.mean_profit(self.best_price)
        
    def next_step(self, alg):
        next_price, additional_information = alg.next_price(self)
        next_price = torch.clip(next_price, min=self.model.el, max=self.model.u)
        self.set_price(next_price, additional_information)
        return additional_information
        
    def set_price(self,p,additional_information):
        self.t += 1
        
        self.price_history[self.t-1,:]=p
            
        choice = self.model.sample_choice(p)
        self.choice_history[self.t-1] = choice
            
        mean_demand = self.model.mean_demand(p)
        self.demand_history[self.t-1,:]=mean_demand
        rev = mean_demand.dot(p)
        self.simple_regret = self.best_revenue-rev
        self.regret += self.simple_regret #cumlative regret
        self.regret_history.append(self.regret.item())
        self.simple_regret_history.append(self.simple_regret.item())  
        