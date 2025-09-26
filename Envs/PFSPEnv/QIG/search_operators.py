from random import shuffle
import random
import copy

def perturbation_mechanism(Self, action):
    
    # Distruction phase
    Self.num_jobs_remove = action
        
    removed_jobs = random.sample(Self.current_solution.sequence, Self.num_jobs_remove)
    Self.new_solution.sequence = [job for job in Self.current_solution.sequence if job not in removed_jobs]
    
    # Optional: Local search on partial solution
    if Self.local_search_destruction_partial_solution == 'insertion_neighborhood':
        insertion_neighborhood(Self.new_solution, Self.best_solution, False, Self.until_no_improvement, Self.tie_breaking_destruction_partial_solution)
    else:
        pass
            
    # Construction phase
    for job in removed_jobs:
        Self.new_solution.insert_in_best_position(job, Self.tie_breaking_construction)


def local_search(Self, method):
    if method == 'insertion_neighborhood':
        insertion_neighborhood(Self.new_solution, Self.best_solution, Self.ref_best, Self.until_no_improvement, Self.tie_breaking_main_LS)
    else:
        pass

def insertion_neighborhood(solution, best_solution, ref_best=False, until_no_improvement=True, tie_breaking=False):
    
    """ removes jobs one by one randomly and insert them in their best possible position """
    
    current_makespan = solution.makespan
    improve = True
    best_sequence = copy.deepcopy(solution.sequence)
    best_makespan = current_makespan
    
    while improve:
        improve = False
        
        # If jobs are removed based on a reference sequence (e.g., best found solution so far)
        if ref_best:
            not_tested = best_solution.sequence.copy()
        else:
            not_tested = solution.sequence.copy()
            shuffle(not_tested)
        
        for removed_job in not_tested:
            solution.sequence.remove(removed_job)
            
            solution.insert_in_best_position(removed_job, tie_breaking)
            
            if solution.makespan < current_makespan:
                improve = True
                current_makespan = solution.makespan
                best_sequence = copy.deepcopy(solution.sequence)
                best_makespan = current_makespan
                
        if until_no_improvement == False:
            break
    solution.sequence = best_sequence.copy()
    solution.makespan = best_makespan

