"""
This file contains the baseline algorithms for the PFSP problem.
"""
import numpy as np
import json
from typing import Tuple, List
import random
import pandas as pd
def calculate_makespan(job_sequence: List[int], processing_times: np.ndarray) -> int:
    """
    Calculate the makespan for a given job sequence.
    
    Args:
        job_sequence: List of job indices in the sequence
        processing_times: 2D numpy array of processing times where 
                         processing_times[i,j] is the processing time of job j on machine i
    
    Returns:
        The makespan (maximum completion time)
    """
    processing_times = processing_times.T
    num_machines = processing_times.shape[0]
    num_jobs = len(job_sequence)
    
    # Initialize completion times
    completion_times = np.zeros((num_machines, num_jobs), dtype=np.int32)
    
    # Calculate completion time for the first job
    completion_times[0, 0] = processing_times[0, job_sequence[0]-1]
    for i in range(1, num_machines):
        completion_times[i, 0] = completion_times[i-1, 0] + processing_times[i, job_sequence[0]-1]
    
    # Calculate completion times for the rest of the jobs
    for j in range(1, num_jobs):
        completion_times[0, j] = completion_times[0, j-1] + processing_times[0, job_sequence[j]-1]
        
        for i in range(1, num_machines):
            # Job j on machine i can start after:
            # 1. Job j on machine i-1 is completed
            # 2. Job j-1 on machine i is completed
            completion_times[i, j] = max(completion_times[i-1, j], completion_times[i, j-1]) + processing_times[i, job_sequence[j]-1]
    
    # The makespan is the completion time of the last job on the last machine
    return int(completion_times[num_machines-1, num_jobs-1])


def insert_best_position(instance, job, sequence):
    """ Insert the given job in the position that minimize makespan.
    Parameters:
        instance: instance of the problem (np array)
        job: job to be inserted (int).
        sequence: sequence to be inserted (np array, default: [], meaning self.sequence)
    Returns:
        new_sequence: sequence after insertion (np array)
        makespan: makespan after inserting the job (int)
    """
    length = len(sequence)
    _, m = instance.shape
    e = np.zeros((length, m))  # earliest completion time
    q = np.zeros((length + 1, m))  # tail
    f = np.zeros((length + 1, m))  # earliest relative completion time
    e[0][0] = instance[sequence[0] - 1][0]
    for i in range(1, length):
        e[i][0] = e[i - 1][0] + instance[sequence[i] - 1][0]
    for j in range(1, m):
        e[0][j] = e[0][j - 1] + instance[sequence[0] - 1][j]
    for i in range(1, length):
        for j in range(1, m):
            e[i][j] = max(e[i - 1][j], e[i][j - 1]) + instance[sequence[i] - 1][j]
    for j in range(m):
        q[length][j] = 0
    for i in range(length - 1, -1, -1):
        q[i][m - 1] = q[i + 1][m - 1] + instance[sequence[i] - 1][m - 1]
    for i in range(length - 1, -1, -1):
        for j in range(m - 2, -1, -1):
            q[i][j] = max(q[i + 1][j], q[i][j + 1]) + instance[sequence[i] - 1][j]
    f[0][0] = instance[job - 1][0]
    for i in range(1, length + 1):
        f[i][0] = e[i - 1][0] + instance[job - 1][0]
    for j in range(1, m):
        f[0][j] = f[0][j - 1] + instance[job - 1][j]
    for i in range(1, length + 1):
        for j in range(1, m):
            f[i][j] = max(f[i][j - 1], e[i - 1][j]) + instance[job - 1][j]
    makespans = np.amax(f + q, axis=1)
    best_makespan = np.amin(makespans)
    best_position_tem = np.where(makespans == best_makespan)
    best_position = best_position_tem[0][0]
    new_sequence = np.insert(sequence, best_position, job)
    return new_sequence, best_makespan


def NEH(instance, order_jobs='SD', given_order=[]):
    """get a solution with NEH heuristic
    Parameters:
        tie_breaking: use tie breaking mechanism (0 or 1, default: 0).
        order_jobs: priority order of jobs, possible values are:
                SD: non-decreasing sum of processing times (default);
                RD: random order.
        given_order: if other values are assigned to order_jobs, given_order will be selected as priority order. (array-like)
    Returns:
        sequence: NEH sequence (np array)
        makespan: makespan of NEH sequence (int)
    """
    # set job order
    if order_jobs == 'SD':
        total_processing_times = dict()
        for i in range(1, len(instance)+1):
            total_processing_times[i] = np.sum(instance[i-1])
        sorted_jobs = sorted(total_processing_times, key=total_processing_times.get, reverse=True)
    elif order_jobs == 'RD':
        sorted_jobs = list(range(1, len(instance)+1))
        random.shuffle(sorted_jobs)
    else:
        sorted_jobs = given_order
    # take jobs in order_jobs and insert them in turn in the place which minimize partial makespan
    sequence = np.array([sorted_jobs[0], sorted_jobs[1]], dtype = 'int32')
    makespan_tmp = calculate_makespan(sequence, instance)
    sequence = np.array([sorted_jobs[1], sorted_jobs[0]], dtype = 'int32')
    if makespan_tmp < calculate_makespan(sequence, instance):
        sequence = np.array([sorted_jobs[0], sorted_jobs[1]], dtype = 'int32')
        makespan = makespan_tmp
    for job in sorted_jobs[2:]:
        sequence, makespan = insert_best_position(instance, job, sequence)
    return sequence, makespan


def neh_variants(instance, method):
    '''variants of NEH heuristics.
    
    Parameters:
        method: method to generate priority index (str)
    Returns:
        sequence: sequence obtained (array-like)
        value: fitness value of sequence (int)
        priority_index: priority order used in NEH variants (array-like)
    '''
    _, m = instance.shape
    if method == 'kk1':
        # NEHKK1
        c = dict()
        for i in range(1, len(instance)+1):
            a, b = 0, 0
            for j in range(m):
                a = a + ((m-1) * (m-2) / 2 + m - j) * instance[i-1,j]
                b = b + ((m-1) * (m-2) / 2 + j - 1) * instance[i-1,j]
            c[i] = a if a <= b else b
        priority_index = sorted(c, key=c.get, reverse=True)
    elif method == 'kk2':
        # NEHKK2
        s, t = int(np.floor(m / 2)), int(np.ceil(m / 2))
        c = dict()
        for i in range(1, len(instance)+1):
            u = 0
            for h in range(s):
                u += (h-3/4) / (s-3/4) * (instance[i-1, s+1-h] - instance[i-1,t+h])
            a = np.sum(instance[i-1]) + u
            b = np.sum(instance[i-1]) - u
            c[i] = a if a <= b else b
        priority_index = sorted(c, key=c.get, reverse=True)
    sequence, makespan = NEH(instance, order_jobs='', given_order=priority_index)
    return sequence, makespan, priority_index


def heuristic(instance, method):
    '''heuristic for the PFSP problem.
    
    Parameters:
        instance: instance of the problem (np array)
        method: heuristic to solve the PFSP problem (str)
    Returns:
        sequence: sequence obtained (array-like)
        value: fitness value of sequence (int)
    '''
    if method == 'NEH':
        sequence, makespan = NEH(instance, order_jobs='SD')
    elif method == 'NEH_KK1':
        sequence, makespan, _ = neh_variants(instance, method='kk1')
    elif method == 'NEH_KK2':
        sequence, makespan, _ = neh_variants(instance, method='kk2')
    return sequence, makespan

def parse_processing_times(input_str: str) -> np.ndarray:
    """
    Parse the processing times from the input string.
    
    Args:
        input_str: The input string containing processing times
        
    Returns:
        A 2D numpy array of processing times where times[i][j] is the processing time 
        of job j on machine i
    """
    # Initialize list to store machine processing times
    instance = []
    lines = input_str.split(';')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Extract the processing times for this machine
        parts = line.split(':')
        if len(parts) < 2:
            continue
            
        times_part = parts[1].strip()
        # Extract the list of processing times
        time_str = times_part.split(']')[0].strip('[')
        times = [int(t.strip()) for t in time_str.split(',')]
        instance.append(times)
    
    # Convert to numpy array with int32 dtype
    return np.array(instance, dtype=np.int32)


def read_instance_files(file_paths: list[str]) -> list[np.ndarray]:
    """
    Read the instance files and return a list of numpy arrays.
    """
    instances = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            with open(file_path, 'r') as file:
                data = file.read()
                # Parse the JSON array
                instance_data = json.loads(data)
                # Add each instance from the array to our list
                for instance in instance_data:
                    instances.append(instance)
    return instances


def run_on_instances(instance_paths: list[str], methods: list[str], save_path: str = None):
    """
    Run the heuristic on the instances and print the results.

    Parameters:
        instance_paths: list of paths to the instance files
        methods: list of methods to run on the instances
        save_path: path to save the results
    Returns:
        df: dataframe with the results
    """
    instances_data = read_instance_files(instance_paths)
    results = []
    for i, instance_data in enumerate(instances_data):
        instance = parse_processing_times(instance_data["input"])
        for method in methods:
            sequence, makespan = heuristic(instance, method)
            results.append({
                'instance': i,
                'method': method,
                'sequence': sequence,
                'makespan': makespan
            })
    df = pd.DataFrame(results)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df

