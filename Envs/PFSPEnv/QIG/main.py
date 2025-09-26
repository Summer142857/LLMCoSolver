""" 
This is the main body of the code 

This code can be used to execute the proposed QIG and all other IG algorithms in the paper

"""


from initialization import extract_taillard, extract_vrf
from design_IG import IteratedGreedyAlgorithm

def run_algorithm(dataset, inst_number, time_scale):
    
    # Determine which instance to execute
    if dataset =='taillard':
        processing_times = extract_taillard(inst_number)
    elif dataset == 'vrf_small':
        processing_times = extract_vrf(inst_number, 'small')
    elif dataset == 'vrf_large':
        processing_times = extract_vrf(inst_number, 'large')
    
    stopping_criterion = 'CPU_time' # 'max_iteration' or 'CPU_time'
    max_iteration = 90
    
    ig = IteratedGreedyAlgorithm(processing_times)
    
    runtime_in_miliseconds = 0.5 * len(processing_times) * len(processing_times[0]) * time_scale
    
    ig.execute(stopping_criterion, runtime_in_miliseconds, max_iteration)

    print("Best makespan", ig.best_solution.makespan,"iterations:", ig.iterations)
    return ig

if __name__ == "__main__":
    #example1_random_data()
    dataset = 'taillard' # 'taillard' or 'vrf_small', 'vrf_large'
    inst_number = 0 # number of instance to be solved
    time_scale = 60
    IG = run_algorithm(dataset, inst_number, time_scale)
    
    
    
    
    
        
        
        
