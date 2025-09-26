import numpy as np
import QIG.evaluations as evaluations
import random
from .search_operators import insertion_neighborhood

""" This is to design initial solution """
class make_sol(object):

    def __init__(self, processing_times):
        
        self.jobs_number = len(processing_times)
        self.machines_number = len(processing_times[0])
        self.makespan = 0
        self.idle_time = 0

        self.sequence = list()
        self.processing_times = processing_times
        self.idle_times = np.zeros(shape=(self.jobs_number), dtype= 'int32')

    def calc_Cmax(self):
        """Calculate makespan for the sequence."""
        sequence_to_cython = np.array(self.sequence, dtype='int32')
        self.makespan = evaluations.calculate_completion_times(sequence_to_cython,self.processing_times,self.machines_number)
        return self.makespan

    def calculate_idle_times(self):
        """Calculate the idle time wrt each job and saves in self.idle_time."""
        sequence_to_cython = np.array(self.sequence, dtype='int32')
        memory_view_object = evaluations.idle_times_calc(sequence_to_cython,self.processing_times,self.machines_number)
        self.idle_time = np.array(memory_view_object)

    def insert_in_best_position(self, job, tie_breaking=False):
        
        if tie_breaking:
            use_tie_breaking = 1
        else:
            use_tie_breaking = 0

        sequence_to_cython = np.array(self.sequence, dtype='int32')
        
        best_position, self.makespan, best_positions, best_makespans = evaluations.taillard_acceleration(sequence_to_cython,self.processing_times,job,self.machines_number, use_tie_breaking)

        self.sequence.insert(best_position - 1, job)
        return self.makespan


""" This is to generate initial solution """
def NEH_heuristic(sol_init, local_optimum = True, tie_breaking_within_NEH = False, tie_breaking_partial_initial = False, LS_within_NEH = 'nothing'):
    """ 
    --> Generate initial sol_init with NEH heuristic 
    --> reference: Nawaz, Enscore and Hans heuristic (1983)
    """
    # Order jobs
    jobs_sorted = sort_jobs_non_decreasing(sol_init)
    
    sol_init.sequence = [jobs_sorted[0],jobs_sorted[1]]
    makespan_start = sol_init.calc_Cmax()
    sol_init.sequence = [jobs_sorted[1], jobs_sorted[0]]
    if makespan_start < sol_init.calc_Cmax():
        sol_init.sequence = [jobs_sorted[0],jobs_sorted[1]]
        sol_init.makespan = makespan_start

    for j in jobs_sorted[2:]:
        sol_init.insert_in_best_position(j, tie_breaking_within_NEH)
        if LS_within_NEH == 'insertion_neighborhood':
            insertion_neighborhood(sol_init, sol_init, False, False, tie_breaking_partial_initial)
        else:
            pass

def sort_jobs_non_decreasing(sol_init):
    """ sort jobs by non decreasing sum of the processing times"""
    total_processing_times = dict()
    for i in range(1, sol_init.jobs_number + 1):
        total_processing_times[i] = np.sum(sol_init.processing_times[i-1])

    return sorted(total_processing_times, key=total_processing_times.get, reverse=True)


#### Import Data ###

""" This is to import input data """
def extract_taillard(inst_number):
    
    """ 
    --> Import Taillard instances
    --> jobs = 20 to 500; machines = 5 to 20
    --> Reference: Taillard, E.D. (1993). Benchmarks for basic scheduling problems. European Journal of Operational Research, 64 78-285.
    """

    directory = "./instance sets/taillard instances/"
    instances_names = list()

    for i in range(1, 121):
        if i < 10:
            index = "00" + str(i)
        elif i < 100:
            index = "0" + str(i)
        else:
            index = str(i)
        instances_names.append("ta" + index)
    return extract_instance(instances_names, directory)[inst_number]


def extract_vrf(inst_number, size):
    
    if size == 'small':
        """ 
        --> Import VRF small instances
        --> jobs = 10 to 60; machines = 5 to 20
        --> Reference: Vallada, E., Ruiz, R., & Framinan, J. M. (2015). New hard benchmark for flowshop scheduling problems minimising makespan. European Journal of Operational Research, 240(3), 666-677.
        """
        
        directory = "./instance sets/vrf instances/Small/"
        instances_names = list()
    
        # Generate instance names
        for i in [10,20,30,40,50,60]:
            for j in [5,10,15,20]:
                for k in range(1,11):
                    name = "VFR"+str(i)+"_"+str(j)+"_"+str(k)
                    instances_names.append(name)
        return extract_instance(instances_names, directory, vrf= True)[inst_number]
    
    elif size == 'large':
        """ 
    --> Import VRF small instances
    --> jobs = 100 to 800; machines = 20 to 60
    --> Reference: Vallada, E., Ruiz, R., & Framinan, J. M. (2015). New hard benchmark for flowshop scheduling problems minimising makespan. European Journal of Operational Research, 240(3), 666-677.
    """
    
    directory = "./instance sets/vrf instances/Large/"
    instances_names = list()

    # Generate instance names
    for i in [100,200,300,400,500,600,700,800]:
        for j in [20,40,60]:
            for k in range(1, 11):
                name = "VFR" + str(i) + "_" + str(j) + "_" + str(k)
                instances_names.append(name)
    return extract_instance(instances_names, directory, vrf= True)[inst_number]


def extract_instance(instances_names, directory, vrf=False):
    
    instances = list()
    instance = list()

    # Read files - each file is an instance
    for _, instance_name in enumerate(instances_names):
        if vrf:
            file_name_extension = "_Gap.txt"
        else:
            file_name_extension = ""
        # Open file and jump first line
        f = open(directory + instance_name + file_name_extension, "r")
        f.readline()
        # Reset variables for each instance
        instance.clear()
        job_num = 0
        i = 0
        for line in f.readlines():
            instance.append(list())

            for item in line.split():
                if item != " " and item != "\n":
                    i += 1
                    if i % 2 == 0:
                        instance[job_num].append(int(item))
            job_num += 1
        # Create numpy 2d array for each instance and append to instances list
        instances.append(np.array(instance.copy(), dtype='int32'))
    return instances






