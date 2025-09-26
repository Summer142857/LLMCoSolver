import numpy as np
from datetime import datetime
import time
#from exploration import constructive_heuristic
from .search_operators import local_search, perturbation_mechanism
from .acceptance import acceptance_function

def Q_learning(Self):
    Self.Q_matrix = np.zeros((2, len(Self.operator_list_perturbation)))                 
    Self.state = 0
    ind = np.random.randint(len(Self.operator_list_perturbation))
    episode_size = Self.episode_size
    
    Self.epsilon_greedy = Self.epsilon_greedy
    Self.epsilon_greedy_decay = Self.epsilon_greedy_decay
    Self.learning_rate = Self.learning_rate
    Self.alpha_learning = Self.alpha_learning
    
    Self.actions_sequence = []
    Self.states_sequence = []
        
    while datetime.now() < Self.time_limit and Self.iterations < Self.max_iteration:        
        current_fit_during_episodes = [Self.current_solution.makespan]
        best_fit_during_episodes = [Self.best_solution.makespan]
        
        #### Selecting action ####
        if np.random.rand() < Self.epsilon_greedy:
            Self.action = np.random.randint(len(Self.operator_list_perturbation))
        else:
            Self.action = np.argmax(Self.Q_matrix[Self.state])
        Self.actions_sequence.append(Self.action)
        Self.states_sequence.append(Self.state)

        for ep in range(episode_size):
            
        # 1) Exploration of the current solution using the selected method
            perturbation_mechanism(Self, Self.operator_list_perturbation[Self.action])
            
            # 2) Exploitation of the complete explored solution using the given local search method 
            LS_method = Self.main_local_search
            local_search(Self, LS_method)
            
            # 3) Acceptance Criteria
            acceptance_function(Self)
            
            Self.exe_time.append(time.time())
            Self.iterations += 1
            
            Self.best_fitness_list.append(Self.best_solution.makespan)
            Self.current_fitness_list.append(Self.current_solution.makespan)
            
            current_fit_during_episodes.append(Self.current_solution.makespan)
            best_fit_during_episodes.append(Self.best_solution.makespan)
            
        mins_cur = [current_fit_during_episodes[0]]
        improvement_num = 0
        for imp in range(0, episode_size):
            if current_fit_during_episodes[imp+1] < mins_cur[-1]:
                mins_cur.append(current_fit_during_episodes[imp+1])
        
        mins_best = [best_fit_during_episodes[0]]
        for imp in range(0, episode_size):
            if best_fit_during_episodes[imp+1] < mins_best[-1]:
                improvement_num += 1
                mins_best.append(best_fit_during_episodes[imp+1])
        
        Diff_L = current_fit_during_episodes[0] - mins_cur[-1]
        Diff_G = best_fit_during_episodes[0] - mins_best[-1]
        
        DL = Diff_L/current_fit_during_episodes[0]
        DG = Diff_G/Self.best_solution.makespan
        
        # claculate reward
        reward = 0.3*max(DL,0) + 0.7* max(DG,0)
        
        if improvement_num > 0:
            next_state = 1
            Self.Q_matrix[Self.state][Self.action] = Self.Q_matrix[Self.state][Self.action] + Self.alpha_learning * (reward + Self.learning_rate * np.max(Self.Q_matrix[next_state]) - Self.Q_matrix[Self.state][Self.action])
            Self.state = next_state
        else:
            Self.state = 0 
            Self.Q_matrix[Self.state][Self.action] = Self.Q_matrix[Self.state][Self.action] + Self.alpha_learning * (reward - Self.Q_matrix[Self.state][Self.action])
        
        Self.epsilon_greedy *= Self.epsilon_greedy_decay
        Self.learning_rate *= Self.epsilon_greedy_decay

        
    
def individual(Self):
    while datetime.now() < Self.time_limit and Self.iterations < Self.max_iteration:
        
        
        # 1) Exploration of the current solution using the selected method
        perturbation_mechanism(Self, Self.operator_list_perturbation[0])
        
        # 2) Exploitation of the complete explored solution using the given local search method 
        LS_method = Self.main_local_search
        local_search(Self, LS_method)
        
        # 3) Acceptance Criteria
        acceptance_function(Self)
        
        Self.exe_time.append(time.time())
        Self.iterations += 1
        
        Self.best_fitness_list.append(Self.best_solution.makespan)
        Self.current_fitness_list.append(Self.current_solution.makespan)
            
def random_selection(Self):
    
    while datetime.now() < Self.time_limit and Self.iterations < Self.max_iteration:
        
        
        # 1) Exploration of the current solution using the selected method
        indx = int(np.random.randint(0,len(Self.operator_list_perturbation),1))
        perturbation_mechanism(Self, Self.operator_list_perturbation[indx])
        
        # 2) Exploitation of the complete explored solution using the given local search method 
        LS_method = Self.main_local_search
        local_search(Self, LS_method)
        
        # 3) Acceptance Criteria
        acceptance_function(Self)
        
        Self.exe_time.append(time.time())
        Self.iterations += 1
        
        Self.best_fitness_list.append(Self.best_solution.makespan)
        Self.current_fitness_list.append(Self.current_solution.makespan)
        
    
        
    