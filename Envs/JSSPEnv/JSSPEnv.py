"""This script contains the JSSP environment for the JSSP problem. Here JSSPs are defined as: n jobs need to be processed on m machines. Each job consists of m operations, each operation needs to be processed on a specific machine. The processing time of each operation is given in the instance. The goal is to find a schedule that minimizes the makespan (the time when the last operation is completed)."""


import json
import random
import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool
from ortools.sat.python import cp_model
import math


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def calculate_k_operators_with_lowest_processing_time(instance: np.array, k: int) -> list[list[tuple[int, int, float]]]:
    """
    Calculates the k operators with the lowest processing time for each machine.

    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    k : int
        Number of operators to be calculated.

    Returns
    -------
    list[list[tuple[int, int, float]]]
        A list of k operators with the lowest processing time for each job. Each tuple contains operator index, machine index and processing time.
    """
    num_jobs, num_columns = instance.shape
    num_machines = num_columns // 2
    job_operators = {job: [] for job in range(num_jobs)}

    # Iterate through each job and operator
    for job_idx in range(num_jobs):
        for op_idx in range(num_machines):
            machine_idx, processing_time = instance[job_idx, op_idx * 2], instance[job_idx, op_idx * 2 + 1]
            job_operators[job_idx].append((op_idx, machine_idx, processing_time))

    # Sort operators for each job by processing time and select the k operators with the lowest processing time
    result = []
    for job in range(num_jobs):
        sorted_operators = sorted(job_operators[job], key=lambda x: x[2])  # Sort by processing time
        result.append(sorted_operators[:k])  # Take the k operators with the lowest processing time

    return result


def get_makespan(instance: np.array, schedule: list[list[int]]) -> float:
    """
    Decodes a machine-based job shop scheduling solution.
    
    Parameters:
        instance (np.array): 2D array with shape (n, 2*m) where each row represents a job.
                             For each job the operations are represented as consecutive pairs:
                             (machine, processing_time). There are m operations per job.
        schedule (list of lists): A list of m lists, each being a permutation of job indices.
                                        The k-th list gives the order in which machine k processes 
                                        its designated operations.
                                        
    Returns:
        The makespan (a numerical value) if a feasible schedule is constructed, or
        the string "infeasible" if the machine-based representation does not lead to a
        feasible schedule.
    """
    n, two_m = instance.shape
    m = two_m // 2  # each job has m operations
    
    # job_next[j] tracks the next operation index (0-indexed) to be scheduled for job j.
    job_next = [0] * n
    # job_finish[j] records the finish time of the last scheduled operation for job j.
    job_finish = [0] * n
    # machine_finish[k] records the finish time of the last operation scheduled on machine k.
    machine_finish = [0] * m
    # For each machine, machine_ptr[k] indicates how far we have advanced in its permutation.
    machine_ptr = [0] * m
    
    scheduled_ops = 0
    total_ops = n * m
    
    # Iterative scheduling: while there are operations not yet scheduled.
    while scheduled_ops < total_ops:
        available = []  # list to collect available operations as (start_time, machine, job)
        
        # For each machine, if there is still an operation assigned in its permutation ...
        for machine in range(m):
            if machine_ptr[machine] < n:
                job = schedule[machine][machine_ptr[machine]]
                op_idx = job_next[job]
                # Check that the next unscheduled operation for this job is indeed designated for this machine.
                if op_idx < m and instance[job, 2 * op_idx] == machine:
                    # The operation can start only after the job's previous operation and the machine are free.
                    start_time = max(job_finish[job], machine_finish[machine])
                    available.append((start_time, machine, job))
        
        # If no machine has a ready operation, then the representation is infeasible.
        if not available:
            return "infeasible"
        
        # Choose the operation with the earliest possible start time.
        available.sort(key=lambda x: x[0])
        start_time, machine, job = available[0]
        op_idx = job_next[job]
        proc_time = instance[job, 2 * op_idx + 1]
        finish_time = start_time + proc_time
        
        # Update the finish times and pointers.
        job_finish[job] = finish_time
        machine_finish[machine] = finish_time
        job_next[job] += 1
        machine_ptr[machine] += 1
        scheduled_ops += 1

    # All operations scheduled successfully.
    makespan = max(job_finish)
    return makespan


def solve_jssp(instance: np.array):
    """
    Solve a JSSP instance using or-tools.
    
    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    Returns
    -------
        A tuple (makespan, schedule) if a feasible solution is found, where:
          - makespan: the overall completion time (int)
          - schedule: a dictionary mapping each machine (int) to a list of tuples,
                      each tuple is (job, op, start_time, end_time) for that operation,
                      sorted by start time.
        If no feasible solution is found, returns "infeasible".
    """
    num_jobs = instance.shape[0]
    num_ops = instance.shape[1] // 2  # number of operations per job
    
    # Compute an upper bound on the makespan (sum of all processing times)
    horizon = int(sum(instance[j, 2 * op + 1] for j in range(num_jobs) for op in range(num_ops)))
    
    model = cp_model.CpModel()
    
    # Create task variables: tasks[(j, op)] = (start, end, processing_time, interval)
    tasks = {}
    machine_to_intervals = {}  # Map machine -> list of (job, op, interval)
    
    for j in range(num_jobs):
        for op in range(num_ops):
            machine = int(instance[j, 2 * op])
            processing_time = int(instance[j, 2 * op + 1])
            suffix = f'_j{j}_o{op}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, processing_time, end_var, 'interval' + suffix)
            tasks[(j, op)] = (start_var, end_var, processing_time, interval_var)
            
            if machine not in machine_to_intervals:
                machine_to_intervals[machine] = []
            machine_to_intervals[machine].append((j, op, interval_var))
    
    # Add precedence constraints for each job: operation op must finish before op+1 starts.
    for j in range(num_jobs):
        for op in range(num_ops - 1):
            model.Add(tasks[(j, op)][1] <= tasks[(j, op + 1)][0])
    
    # Add no-overlap constraints for operations assigned to the same machine.
    for machine, intervals in machine_to_intervals.items():
        model.AddNoOverlap([interval for (_, _, interval) in intervals])
    
    # Define makespan: maximum end time over all jobs.
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_ends = [tasks[(j, num_ops - 1)][1] for j in range(num_jobs)]
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)
    
    # Solve the model.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 300
    status = solver.Solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "infeasible"
    
    # Build the schedule: for each machine, list the operations (job, op, start, end)
    schedule = []
    for m in range(num_ops):
        machine_ops = []
        if m in machine_to_intervals:
            # Collect each operation's job and its start time on machine m.
            for (j, op, interval) in machine_to_intervals[m]:
                start_time = solver.Value(tasks[(j, op)][0])
                machine_ops.append((j, start_time))
            # Sort operations by their start time.
            machine_ops.sort(key=lambda x: x[1])
            # Extract only the job ids in processing order.
            schedule.append([j for j, start in machine_ops])
        else:
            schedule.append([])
    
    return solver.Value(makespan), schedule


def tag_prompt_and_transform_to_json(instance, k=2):
    """
    Combines tagging and JSON transformation for a JSSP instance.

    Parameters
    ----------
    instance : np.array
        A JSSP instance.
    k : int
        Number of tags to be used.
    
    Returns
    -------
    dict
        A dictionary with the tagged JSSP instance.
    """
    n_jobs, n_columns = instance.shape
    n_machines = n_columns // 2
    n_ops = n_columns // 2
    instruction = (
        f"Solve the Job Shop Scheduling Problem (JSSP) with {n_jobs} jobs and {n_machines} machines. "
        f"Each job consists of {n_ops} operations which need to be sequentially processed on specific machines. Each machine can process only one job at a time and each job can be processed by only one machine at a time. "
        "Identify the schedule that minimizes the maximum completion time (makespan). "
        "The input includes the information of operations for each job, including their specific machine and processing time, as well as the operators with lowest processing time and their respective machines and processing times. "
        "Provide the solution in the following format:\n\n"
        "1. Schedule: List the order that jobs are processed on each machine.\n"
        "2. Makespan: The makespan of the schedule.\n"        
    )

    instance_reformated = []
    for job_idx in range(n_jobs):
        job_info = instance[job_idx].tolist()
        job_info_reformated = []
        for op_idx in range(n_ops):
            job_info_reformated.append((job_info[2 * op_idx], job_info[2 * op_idx + 1]))
        instance_reformated.append(job_info_reformated)

    k_operations_with_lowest_processing_time = calculate_k_operators_with_lowest_processing_time(instance, k)
    job_descriptions = []
    for job_idx, k_operations in enumerate(k_operations_with_lowest_processing_time):
        lowest_str = [f"{op_info[0]}: ({op_info[1]}, {op_info[2]})" for op_info in k_operations]
        job_desc = (
            f"Job {job_idx}, machines and processing times for operations: {instance_reformated[job_idx]}, "
            f"operators with lowest processing time: {lowest_str}; "
        ).replace("\'", "")
        job_descriptions.append(job_desc)

    # solve the JSSP
    makespan, schedule = solve_jssp(instance)
    output = f"Schedule: {schedule}, Makespan: {makespan}"

    # create the JSON-ready dictionary
    jssp_json = {
        "n": str(n_jobs),
        "m": str(n_machines),
        "instruction": instruction,
        "output": output,
        "input": ''.join(job_descriptions),
    }
    jssp_json['input'] = ".".join(jssp_json['input'].rsplit(";", 1))

    return jssp_json


class JSSPEnv:
    """
    A JSSP Environment class to generate JSSP instances (processing time matrices) using random integers and save them in JSON format.
    """
    def __init__(
            self, 
            n_job_range: list[int], 
            n_mac_range: list[int], 
            seed: int | None = None
    ) -> None:
        """
        Initialize the JSSP environment.

        Parameters
        ----------
        n_job_range : list[int]
            [min_job, max_job] range for number of jobs.
        n_mac_range : list[int]
            [min_mac, max_mac] range of number of machines.
        seed : int | None
            Seed for random number generation.
        """
        self.n_job_range = n_job_range
        self.n_mac_range = n_mac_range
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


    def generate_jssp(self, num_jobs: int, num_machines: int, dataset_size: int) -> list[np.array]:
        """
        Generate a list of JSSP instances.

        Parameters
        ----------
        num_jobs : int
            Number of jobs.
        num_machines : int
            Number of machines.
        dataset_size : int
            Number of instances to generate.

        Returns
        -------
        list[np.array]
            A list of JSSP instances as 2-d np-arrays.
        """
        instances = []
        for _ in range(dataset_size):
            processing_time = np.random.randint(1, 100, size=(num_jobs, num_machines), dtype=np.int32)
            machine_assignment = np.zeros((num_jobs, num_machines))
            for job_idx in range(num_jobs):
                machines = list(range(num_machines))
                random.shuffle(machines)
                machine_assignment[job_idx] = machines
            instance = []
            for job_idx in range(num_jobs):
                job_info = []
                for op_idx in range(num_machines):
                    job_info += [machine_assignment[job_idx, op_idx], processing_time[job_idx, op_idx]]
                instance.append(job_info)
            instances.append(np.array(instance, dtype=np.int32))
        return instances

    
    def generate_instances(self, n_instance: int) -> list[np.array]:
        """
        Generate a list of JSSP instances.

        Parameters
        ----------
        n_instance : int
            Number of instances to generate.

        Returns
        -------
        list[np.array]
            A list of JSSP instances as 2-d np-arrays.
        """
        instances = []
        for _ in range(n_instance):
            num_jobs = np.random.randint(self.n_job_range[0], self.n_job_range[1] + 1)
            num_machines = np.random.randint(self.n_mac_range[0], self.n_mac_range[1] + 1)
            instances += self.generate_jssp(num_jobs, num_machines, 1)
        return instances
    

    def save_dataset(self, dataset, filename, disable_print=False):
        filedir = os.path.split(filename)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(check_extension(filename), 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        if not disable_print:
            print(">> Save dataset to {}".format(filename))


    def generate_instances_and_save(self, n_instance: int, file_name: str, save_pkl: bool, k: int, pickle_path: str=None) -> None:
        """
        Generate JSSP instances and save them to a JSON file.

        Parameters
        ----------
        n_instance : int
            Number of JSSP instances to generate.
        file_name : str
            Path where the resulting JSON file should be saved.
        save_pkl : bool
            Whether to save the dataset as a pickle file.
        k : int
            Number of tags to be used.
        pickle_path : str
            Path where the resulting pickle file should be saved.
        """
        instances = self.generate_instances(n_instance)
        if save_pkl and pickle_path is not None:
            self.save_dataset(instances, pickle_path)
        jssp_data = []
        for instance in tqdm(instances, desc="Generating JSSP instances"):
            jssp_data.append(tag_prompt_and_transform_to_json(instance, k))

        with open(file_name, 'w') as f:
            json.dump(jssp_data, f, indent=4)
        
       
def parse_args():
    """
    Parse command line arguments for the PFSP environment.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Job Shop Scheduling Problem (JSSP) Instance Generator')
    
    # Job and machine range parameters
    parser.add_argument('--min_jobs', type=int, default=5, help='Minimum number of jobs')
    parser.add_argument('--max_jobs', type=int, default=20, help='Maximum number of jobs')
    parser.add_argument('--min_machines', type=int, default=5, help='Minimum number of machines')
    parser.add_argument('--max_machines', type=int, default=20, help='Maximum number of machines')
    
    # Instance generation parameters
    parser.add_argument('--n_instances', type=int, help='Number of PFSP instances to generate')
    parser.add_argument('--output_file', type=str, default='./output/train_jssp.json', help='Path to save the JSON output file')
    parser.add_argument('--save_pkl', action='store_true', default=False, help='Whether to save the dataset as a pickle file')
    parser.add_argument('--pickle_path', type=str, default=None, help='Path to save the pickle file')
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # JSON transformation parameters
    parser.add_argument('--k', type=int, default=2, help='Number of tags to include in the description')
    parser.add_argument('--n_processes', type=int, default=8, help='Number of processes to use for parallel generation')
    
    return parser.parse_args()


def generate_subset(process_index, seed, args, instances_per_process, n_processes):
    """
    Generate a subset of JSSP instances in a single process.
    
    Parameters
    ----------
    process_index : int
        Index of the current process.
    seed : int
        Random seed for this process.
    args : argparse.Namespace
        Command line arguments.
    instances_per_process : int
        Number of instances to generate per process.
    n_processes : int
        Total number of processes.
        
    Returns
    -------
    str
        Path to the file containing the generated instances.
    """
    # Create output file name with process index
    base_name, ext = os.path.splitext(args.output_file)
    process_file_name = f"{base_name}_part{process_index}{ext}"
    pickle_path = None
    if args.pickle_path is not None:
        pickle_path = f"{base_name}_part{process_index}.pkl"
    
    # Create JSSP environment with derived seed
    jssp_env = JSSPEnv(
        n_job_range=[args.min_jobs, args.max_jobs],
        n_mac_range=[args.min_machines, args.max_machines],
        seed=seed
    )

    # Calculate number of instances for this process
    if process_index == n_processes - 1:
        # Last process handles the remainder
        n_instances = args.n_instances - (process_index * instances_per_process)
    else:
        n_instances = instances_per_process

    # Generate instances and save to separate file
    print(f"Process {process_index}: Generating {n_instances} instances with seed {seed}")
    jssp_env.generate_instances_and_save(
        n_instance=n_instances,
        file_name=process_file_name,
        save_pkl=args.save_pkl,
        pickle_path=pickle_path,
        k=args.k
    )

    return process_file_name


def generate_instances_parallel(args, n_processes, combine_files=True):
    """
    Generate JSSP instances in parallel using multiple processes.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    n_processes : int
        Number of parallel processes to use.
    combine_files : bool
        Whether to combine individual JSON files into one.

    Returns
    -------
    list or list of str
        If combine_files is True, returns the combined data from all processes.
        If combine_files is False, returns the list of file paths to individual JSON files.
    """
    # Calculate instances per process
    instances_per_process = math.ceil(args.n_instances / n_processes)

    # Generate derived seeds from initial seed
    np.random.seed(args.seed)
    derived_seeds = np.random.randint(0, 100000, size=n_processes).tolist()

    # Create and start the process pool
    with Pool(processes=n_processes) as pool:
        result_files = pool.starmap(
            generate_subset, 
            [(i, derived_seeds[i], args, instances_per_process, n_processes) for i in range(n_processes)]
        )

    # Combine files if requested
    if not combine_files:
        print(f"Generated {args.n_instances} JSSP instances in {n_processes} separate files:")
        for file_name in result_files:
            print(f"  - {file_name}")
        return result_files
    
    # Combine the individual JSON files into one
    print("Combining results from all processes...")
    combined_data = []
    for file_name in result_files:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
            # Remove the temporary file
            os.remove(file_name)
    # Save the combined data
    with open(args.output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Successfully generated {len(combined_data)} JSSP instances and saved to {args.output_file}")
    return combined_data


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Generate instances in parallel
    generate_instances_parallel(args, n_processes=args.n_processes, combine_files=True)
    
    # generate 500,000 instances: 
    # python JSSPEnv.py --n_instances 500000 --output_file ./training_instance/train_jssp.json --k 2 --n_processes 32

    # generate 300 instances:
    # python JSSPEnv.py --n_instances 300 --output_file ./testing/test_jssp.json --k 2 --n_processes 1 --save_pkl --pickle_path ./testing/test_jssp.pkl