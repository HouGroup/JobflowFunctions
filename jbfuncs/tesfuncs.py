from jobflow import job

@job
def count_str(input_str: str):
    return len(input_str)

@job
def sum_numbers(numbers):
    print(sum(numbers))
    return sum(numbers)

from jobflow import Flow, Response, job, Maker
from jobflow import Response
from jobflow.managers.local import run_locally
from dataclasses import dataclass
from skopt import gp_minimize
from skopt.space import Real, Integer
import numpy as np
from typing import Callable, Dict, Any

def cp_updt_dict(old_dict, up_dict):
    if old_dict == None:
        return up_dict
    new_dict = old_dict.copy()
    new_dict.update(up_dict)
    return new_dict

@job
def objective_function(x):
    return (x - 2)**2 + 10*np.sin(x) + np.random.randn() * 0.1  # 添加一些噪声

@dataclass
class GPOptMaker(Maker):
    name: str = 'GP optimizer'
    trials: int = 20
    base_estimator: str = 'GP'
    acq_func: str = 'EI'
    acq_optimizer: str = 'lbfgs'
    random_state: int = 42
    space: Dict[str, Any] = None
    eval_function: Callable = None
    metadata: Dict[str, Any] = None
    
    @job
    def ask(self, previous_results = {'Xi':[], 'yi':[]}, new_x = None, new_y = None):
        if len(previous_results['Xi']) == self.trials:
            return previous_results

        #param space
        sp = []
        for i in self.space.keys():
            here = self.space[i]
            if here['type'] == float:
                sp.append(Real(here['range'][0], here['range'][1], name = i))
            else:
                sp.append(Integer(here['range'][0], here['range'][1], name = i))
        
        #define optimizer
        from skopt import Optimizer
        optimizer = Optimizer(
            dimensions = sp,
            base_estimator = self.base_estimator,
            acq_func = self.acq_func,
            acq_optimizer = self.acq_optimizer,
            random_state = self.random_state
        )

        #tell previous results
        if not new_x == None:
            #tell previous results
            previous_results['Xi'].append(new_x)
            previous_results['yi'].append(new_y)
            print(previous_results)
            optimizer.tell(previous_results['Xi'], previous_results['yi'])
        
        #ask next point
        next_x = optimizer.ask()

        #input params
        params = {list(self.space.keys())[i]:next_x[i] for i in range(len(next_x))}
        
        ##next job
        #metadata
        function_job = self.eval_function(**params)
        function_job.update_metadata(cp_updt_dict(self.metadata, {'id':len(previous_results)+1,'job':'evaluate'}))
        ask_job = self.ask(previous_results, next_x, function_job.output)
        ask_job.update_metadata(cp_updt_dict(self.metadata, {'id':len(previous_results)+1,'job':'ask'}))
        return Response(output = previous_results, addition = Flow([function_job, ask_job]))

    @job
    def make(self):
        ask_job = self.ask()
        ask_job.update_metadata(cp_updt_dict(self.metadata, {'id':1,'job':'ask'}))
        return Response(addition = ask_job)
