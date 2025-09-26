import random
import numpy as np
from datetime import datetime, timedelta
from .initialization import make_sol, NEH_heuristic
from .search_operators import insertion_neighborhood
from .search_strategy import Q_learning, individual, random_selection

class IteratedGreedyAlgorithm(object):

    def __init__(self, instance_processing_times):
        
        self.instance_processing_times = instance_processing_times
        self.current_solution = make_sol(self.instance_processing_times)
        self.new_solution = make_sol(self.instance_processing_times)
        self.best_solution = make_sol(self.instance_processing_times)
        
        self.tau = 0.7 # temparature parameter
        
        ### Settings for QIG, RIG, IIGs, IG_RS, IG_PTL, IG_FF, IG_DPS
        self.tie_breaking_within_NEH = True # True or False
        self.tie_breaking_complete_initial_solution = True # True or False 
        self.tie_breaking_partial_initial = True # True or False 
        self.tie_breaking_destruction_partial_solution = False # True or False
        self.tie_breaking_construction = True # True or False
        self.tie_breaking_main_LS = True # True or False
        
        
        self.until_no_improvement = True
        self.main_local_search = 'insertion_neighborhood'
        self.local_search_destruction_partial_solution = 'insertion_neighborhood'  # 'insertion_neighborhood' or 'nothing' 
        self.local_search_on_complete_initial_solution = 'insertion_neighborhood'  # 'insertion_neighborhood' or 'nothing' 
        self.local_search_within_NEH = 'insertion_neighborhood'  # 'insertion_neighborhood' or 'nothing' 
        
        self.ref_best = False # insertion of jobs based on the order in the best solution
        
        self.operator_list_perturbation = [1,2,3] # actions: value of d (number of jobs to remove)
        self.exe_time = []
        self.best_fitness_list = []
        self.current_fitness_list = []
        
        # Q-learning parameters
        self.episode_size = 6
        self.epsilon_greedy = 0.8
        self.epsilon_greedy_decay = 0.996
        self.learning_rate = 0.8
        self.alpha_learning = 0.6
        
        
    def execute(self, stopping_criterion, runtime_in_miliseconds, max_iteration):
        """
        --> Run the algorithm
        --> stopping criteria = Max Iteration or Max CPU Time
        """
        # 0) Define constant temperature and run time
        if stopping_criterion == 'max_iteration':
            self.max_iteration = max_iteration
            self.runtime = float('inf')
            self.time_limit = datetime.now() + timedelta(milliseconds=runtime_in_miliseconds)
        elif stopping_criterion == 'CPU_time':
            self.max_iteration = float('inf')
            self.runtime = runtime_in_miliseconds
            self.time_limit = datetime.now() + timedelta(milliseconds=runtime_in_miliseconds)
            
        
        self.iterations = 0
        self.temperature = self.calc_temp()
        

        ### Generate initial solution & optional local search on partial sequence ###
        NEH_heuristic(self.current_solution, self.until_no_improvement, self.tie_breaking_within_NEH, self.tie_breaking_partial_initial, self.local_search_within_NEH)
        
        
        self.best_solution.sequence = self.current_solution.sequence.copy()
        self.best_solution.makespan = self.current_solution.makespan
        
        self.best_fitness_list.append(self.best_solution.makespan)
        
        if self.local_search_on_complete_initial_solution == 'insertion_neighborhood':
            insertion_neighborhood(self.current_solution, self.best_solution, self.ref_best, self.until_no_improvement, self.tie_breaking_complete_initial_solution)
        
        else:                
            pass


        self.best_solution.sequence = self.current_solution.sequence.copy()
        self.best_solution.makespan = self.current_solution.makespan
        
        self.best_fitness_list.append(self.best_solution.makespan)
        
        """ Main loop of the algorithm """
        ### To run 'QIG' ###
        Q_learning(self)
        
        ### To run 'IIG' ###
        #search_strategy.individual(self)
        
        ### To run 'RIG' ###
        #search_strategy.random_selection(self)

    def calc_temp(self):
        
        """ Calculate temparature """
        temp = 0
        for i in range(self.current_solution.jobs_number):
            temp += np.sum(self.current_solution.processing_times[i])

        diff = self.current_solution.jobs_number * self.current_solution.machines_number * 10
        return self.tau * (temp/diff)
