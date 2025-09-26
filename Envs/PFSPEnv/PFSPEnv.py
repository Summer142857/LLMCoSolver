import json
import random
import pickle
import numpy as np
# import torch
# from scipy.spatial import cKDTree, distance
# from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import argparse
from multiprocessing import Pool
import math

from QIG.design_IG import IteratedGreedyAlgorithm


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


'''  
### these functions are not used in the current version of the PFSPEnv
def generate_weights(start: int, end: int) -> list[int]:
    """
    Generate weights for weighted random sampling (1 to n).

    Parameters
    ----------
    start : int
        Start of the range (inclusive).
    end : int
        End of the range (inclusive).

    Returns
    -------
    list[int]
        List of weights corresponding to the range [start, end].
    """
    return [i for i in range(start, end + 1)]


def weighted_random_choice(start: int, end: int) -> int:
    """
    Choose a random number from start to end (inclusive) with weighted probability.

    Parameters
    ----------
    start : int
        Start of the range (inclusive).
    end : int
        End of the range (inclusive).

    Returns
    -------
    int
        A randomly chosen integer in [start, end].
    """
    numbers = list(range(start, end + 1))
    weights = generate_weights(start, end)
    return random.choices(numbers, weights=weights, k=1)[0]
'''


def calculate_k_jobs_with_lowest_processing_time(instance: np.array, k: int) -> list[list[tuple[int, float]]]:
    """
    For each machine, calculate the k jobs with the lowest processing time.

    Parameters
    ----------
    instance : np.array
        Processing time matrix of the PFSP problem. The element at position (i, j) represents the processing time of job
        i on machine j.
    k : int
        Number of jobs to select.

    Returns
    -------
    list[list[tuple[int, float]]]
        A list of length m, where m is the number of machines. Each element is a list of k tuples, where each tuple
        contains the job index and its processing time.
    """
    n_jobs, n_machines = instance.shape
    result = []
    
    for machine_idx in range(n_machines):
        # Get processing times for all jobs on this specific machine
        processing_times = [(job_idx, instance[job_idx, machine_idx]) for job_idx in range(n_jobs)]
        
        # Sort by processing time (ascending)
        processing_times.sort(key=lambda x: x[1])
        
        # Take the k jobs with lowest processing times
        k_lowest = processing_times[:k]
        
        result.append(k_lowest)
    
    return result


def tag_prompt_and_transform_to_json(instance, k=2, running_time_scale=30):
    """
    Combines tagging and JSON transformation for a PFSP instance.

    Parameters
    ----------
    instance : np.array
        Processing time matrix of the PFSP problem. The element at position (i, j) represents the processing time of job
        i on machine j.
    k : int
        Number of jobs with lowest processing time to include in the description.

    Returns
    -------
        dict: JSON-ready dictionary containing the PFSP instance description with all numerical results as text.
    """
    n_jobs, n_machines = instance.shape
    instruction = (
        f"Solve the Permutation Flowshop Scheduling Problem (PFSP) with {n_jobs} jobs and {n_machines} machines. "
        "Each machine can process only one job at a time and each job can be processed by only one machine at a time. Jobs must be processed on each machine in the same order. "
        "Identify the job order that minimizes the maximum completing time. "
        f"The input includes the processing times of each machine on every jobs, the jobs with lowest processing time for each machines, and their respective processing times. "
        "Provide the solution in the following format:\n\n"
        "1. Order: List the order that jobs are processed on each machine.\n"
        "2. Objective: The objective value (maximum completing time)."
    )

    k_lowest_processing_times = calculate_k_jobs_with_lowest_processing_time(instance, k)
    machine_descriptions = []
    for machine_idx, k_lowest in enumerate(k_lowest_processing_times):
        lowest_str = [f"{job_info[0]}: {job_info[1]}" for job_info in k_lowest]
        machine_desc = (
            f"Machine {machine_idx}, processing times: {instance[:, machine_idx].tolist()}, "
            f"jobs with lowest processing time: {lowest_str}; "
        ).replace("\'", "")
        machine_descriptions.append(machine_desc)

    # solve the PFSP
    qig = IteratedGreedyAlgorithm(instance)
    running_time = 0.5 * n_jobs * n_machines * running_time_scale
    qig.execute("CPU_time", running_time, 90)
    order = qig.best_solution.sequence
    obj_value = qig.best_solution.makespan
    output = f"Order: {order}, Objective: {obj_value}"

    # create the JSON-ready dictionary
    pfsp_json = {
        "n": str(n_jobs),
        "m": str(n_machines),
        "instruction": instruction,
        "output": output,
        "input": ''.join(machine_descriptions),
    }
    pfsp_json['input'] = ".".join(pfsp_json['input'].rsplit(";", 1))

    return pfsp_json


class PFSPEnv:
    """
    A PFSP Environment class to generate PFSP instances (processing time matrices) using random integers and save them in JSON format.

    The PFSP instances are generated following two well-known benchmarks: 
        - Taillard benchmark: Taillard, E. (1993). Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2), 278-285. 
        - VRF benchmark: Vallada, E., Ruiz, R., & Framinan, J. M. (2015). New hard benchmark for flowshop scheduling problems minimising makespan. European Journal of Operational Research, 240(3), 666-677.
    """

    def __init__(
            self,
            n_job_range: list[int],
            n_mac_range: list[int],
            seed: int | None = None
    ) -> None:
        """
        Initialize the PFSP environment.

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

    def generate_instances(self, n_instance: int) -> list[np.array]:
        """
        Generate a list PFSP instances as 2-d np-arrays.

        Parameters
        ----------
        n_instance : int
            Number of instances to generate.

        Returns
        -------
        list[np.array]
            A list of processing time matrix for PFSP. The elements are 2-d np-arrays of shape (n,m), where n is randomly chosen from n_job_range and m is randomly chosen from n_mac_range.
        """
        instances = []
        for _ in range(n_instance):
            n_jobs = np.random.randint(self.n_job_range[0], self.n_job_range[1]+1)
            n_machines = np.random.randint(self.n_mac_range[0], self.n_mac_range[1]+1)
            
            # generate instances
            instance = (
                self.generate_pfsp(1, n_jobs, n_machines)
            )
            instances += instance
        return instances
    
    def generate_pfsp(self, dataset_size: int, num_jobs: int, num_machines: int) -> list[np.ndarray]:
        """
        Generate PFSP instances with Gaussian mixture distribution.

        Parameters
        ----------
        dataset_size : int
            Number of PFSP instances to generate.
        num_jobs : int
            Number of jobs.
        num_machines : int
            Number of machines.

        Returns
        -------
        list[np.ndarray]
            A list of 2-d np-arrays of shape (num_jobs, num_machines) representing the processing times of the PFSP instances.
        """
        result = []
        for _ in range(dataset_size):
            result.append(
                np.random.randint(1, 100, size=(num_jobs, num_machines), dtype=np.int32)
            )
        return result
    


    def save_dataset(self, dataset, filename, disable_print=False):
        filedir = os.path.split(filename)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(check_extension(filename), 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        if not disable_print:
            print(">> Save dataset to {}".format(filename))

    
    def generate_instances_and_save(self, n_instance: int, file_name: str, save_pkl: bool, k: int, running_time_scale: int, pickle_path: str=None) -> None:
        """
        Generate PFSP instances and save them to a JSON file.

        Parameters
        ----------
        n_instance : int
            Number of PFSP instances to generate.
        file_name : str
            Path where the resulting JSON file should be saved.
        save_pkl : bool
            Whether to save the dataset as a pickle file.
        k : int
            Number of jobs with lowest processing time to include in the description.
        running_time_scale : int
            Scale factor for running time calculation.
        pickle_path : str
            Path where the resulting pickle file should be saved.
        """
        instances = self.generate_instances(n_instance)
        if save_pkl and pickle_path is not None:
            self.save_dataset(instances, pickle_path)
        pfsp_data = []
        for instance in tqdm(instances, desc="Generating PFSP instances"):
            pfsp_data.append(tag_prompt_and_transform_to_json(instance, k, running_time_scale))

        with open(file_name, 'w') as f:
            json.dump(pfsp_data, f, indent=4)


def parse_args():
    """
    Parse command line arguments for the PFSP environment.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Permutation Flowshop Scheduling Problem (PFSP) Instance Generator')
    
    # Job and machine range parameters
    parser.add_argument('--min_jobs', type=int, default=5, help='Minimum number of jobs')
    parser.add_argument('--max_jobs', type=int, default=20, help='Maximum number of jobs')
    parser.add_argument('--min_machines', type=int, default=5, help='Minimum number of machines')
    parser.add_argument('--max_machines', type=int, default=20, help='Maximum number of machines')
    
    # Instance generation parameters
    parser.add_argument('--n_instances', type=int, help='Number of PFSP instances to generate')
    parser.add_argument('--output_file', type=str, default='./output/train_pfsp.json', help='Path to save the JSON output file')
    parser.add_argument('--save_pkl', action='store_true', default=False, help='Whether to save the dataset as a pickle file')
    parser.add_argument('--pickle_path', type=str, default=None, help='Path to save the pickle file')
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # JSON transformation parameters
    parser.add_argument('--k', type=int, default=2, help='Number of jobs with lowest processing time to include in the description')
    parser.add_argument('--running_time_scale', type=int, default=30, help='Scale factor for running time calculation')
    parser.add_argument('--n_processes', type=int, default=8, help='Number of processes to use for parallel generation')
    
    return parser.parse_args()


def generate_instances_parallel(args, n_processes, combine_files=True):
    """
    Generate PFSP instances in parallel using multiple processes.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    n_processes : int
        Number of parallel processes to use.
    combine_files : bool
        Whether to combine individual JSON files into one. If False, keep individual files.
        
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
    
    if not combine_files:
        print(f"Generated {args.n_instances} PFSP instances in {n_processes} separate files:")
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
    
    print(f"Successfully generated {len(combined_data)} PFSP instances and saved to {args.output_file}")
    return combined_data


def generate_subset(process_index, seed, args, instances_per_process, n_processes):
    """
    Generate a subset of PFSP instances in a single process.
    
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
    
    # Create PFSP environment with derived seed
    pfsp_env = PFSPEnv(
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
    pfsp_env.generate_instances_and_save(
        n_instance=n_instances,
        file_name=process_file_name,
        save_pkl=args.save_pkl,
        pickle_path=pickle_path,
        k=args.k,
        running_time_scale=args.running_time_scale
    )
    
    return process_file_name


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Generate instances in parallel
    generate_instances_parallel(args, n_processes=args.n_processes, combine_files=False)

    # generate 500,000 instances: 
    # python PFSPEnv.py --n_instances 500000 --output_file ./output/train_pfsp.json --k 2 --running_time_scale 30 --n_processes 8

    # generate 300 instances:
    # python PFSPEnv.py --n_instances 300 --output_file ./testing/test_pfsp.json --k 2 --running_time_scale 30 --n_processes 1 --save_pkl --pickle_path ./testing/test_pfsp.pkl
