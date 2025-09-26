import os, pickle
import numpy as np
import re
import json
from datasets import Dataset
import ast
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

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

def calculate_pfsp_makespan(job_order, processing_times, m_machines):
    """Calculate the makespan for a given job order in PFSP."""
    # Convert 1-indexed job order to 0-indexed
    zero_indexed_order = [job - 1 for job in job_order]
    n_jobs = len(zero_indexed_order)

    # Initialize completion times
    completion_times = [[0 for _ in range(n_jobs)] for _ in range(m_machines)]

    # Calculate completion time for first job on all machines
    first_job = zero_indexed_order[0]
    completion_times[0][0] = processing_times[0][first_job]
    for m in range(1, m_machines):
        completion_times[m][0] = completion_times[m-1][0] + processing_times[m][first_job]

    # Calculate completion times for remaining jobs
    for j in range(1, n_jobs):
        current_job = zero_indexed_order[j]

        # First machine
        completion_times[0][j] = completion_times[0][j-1] + processing_times[0][current_job]

        # Remaining machines
        for m in range(1, m_machines):
            completion_times[m][j] = max(completion_times[m][j-1], completion_times[m-1][j]) + processing_times[m][current_job]

    # Return the makespan (completion time of the last job on the last machine)
    return completion_times[m_machines-1][n_jobs-1]


def extract_objectives(json_file_path):
    """
    Extract all objective values from a TSP JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing TSP data
    
    Returns:
        List of extracted objective values
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file")
        return []
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return []
    
    objectives = []
    
    # Check if data is a list of objects or a single object
    if isinstance(data, list):
        items = data
    else:
        items = [data]
    
    for item in items:
        if 'output' in item:
            # Extract the objective value using regex
            match = re.search(r'Objective: (\d+\.?\d*)', item['output'])
            if match:
                objectives.append(float(match.group(1)))
            else:
                match = re.search(r'Makespan: (\d+\.?\d*)', item['output'])
                if match:
                    objectives.append(float(match.group(1)))
                else:
                    print(f"Warning: Could not extract objective value from {item.get('num_nodes', 'unknown')}-node problem")
    
    return objectives

def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ])))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus

    
def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def load_pkl_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data

def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def compute_euclidean_distance_matrix(locations):
    if isinstance(locations, list):
        locations = np.array(locations)
    num_nodes = locations.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # Calculate the Euclidean distance between each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return distance_matrix


def calculate_total_distance(B, A):
    """
    B: tour
    A: distance matrix
    """
    total_distance = 0
    for i in range(len(B) - 1):
        from_node = B[i]
        to_node = B[i + 1]
        total_distance += A[from_node][to_node]

    # Add the distance from the last node back to the starting node to complete the tour
    total_distance += A[B[-1]][B[0]]

    return total_distance

def extract_predicted_solution(text: str) -> str:
    """
    Given a full decoded text which contains:
      ### Instruction (Problem): ...
      ### Input (Instance Data): ...
      ### Response (Proposed Solution): ...
    Return only the substring after '### Response (Proposed Solution):'
    so that we can parse route, objective, etc., from that substring.
    """
    # Split at "### Response (Proposed Solution):"
    # (Adjust the exact marker to match your prompt formatting.)
    parts = text.split("### Response:")
    if len(parts) > 1:
        # The actual solution text is everything after the marker
        return parts[1].strip()
    else:
        # If for some reason the marker is missing, just return the raw text
        return text

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

def calculate_pfsp_makespan(job_order, processing_times, m_machines):
    """Calculate the makespan for a given job order in PFSP."""
    # Convert 1-indexed job order to 0-indexed
    zero_indexed_order = [job - 1 for job in job_order]
    n_jobs = len(zero_indexed_order)
    
    # Initialize completion times
    completion_times = [[0 for _ in range(n_jobs)] for _ in range(m_machines)]
    
    # Calculate completion time for first job on all machines
    first_job = zero_indexed_order[0]
    completion_times[0][0] = processing_times[0][first_job]
    for m in range(1, m_machines):
        completion_times[m][0] = completion_times[m-1][0] + processing_times[m][first_job]
    
    # Calculate completion times for remaining jobs
    for j in range(1, n_jobs):
        current_job = zero_indexed_order[j]
        
        # First machine
        completion_times[0][j] = completion_times[0][j-1] + processing_times[0][current_job]
        
        # Remaining machines
        for m in range(1, m_machines):
            completion_times[m][j] = max(completion_times[m][j-1], completion_times[m-1][j]) + processing_times[m][current_job]
    
    # Return the makespan (completion time of the last job on the last machine)
    return completion_times[m_machines-1][n_jobs-1]


def compute_metric_cop(predictions, labels, instances, problem):
    """
    Compute custom feasibility and optimality gap metrics for TSP, OP, CVRP, MVC, MIS, or JSSP.

    :param predictions: A list of full decoded strings (including prompt + response).
    :param labels: A list of reference/label strings (including prompt + gold solution).
    :param instances: The raw instances data for each example.
    :param problem: "tsp", "op", "cvrp", "mvc", "mis", or "jssp".
    :return: (feasibility_rate, mean_optimality_gap).
    """
    gaps = []
    infeasibility = 0
    predicted_objectives = []
    optimal_objectives = []

    for i, prediction_full_text in enumerate(predictions):
        # Extract just the solution part from predicted text
        prediction_solution = extract_predicted_solution(prediction_full_text)

        # Extract the solution part from the gold/label text
        label_solution = extract_predicted_solution(labels[i])
        

        if problem == "tsp":
            # Parse route from the label
            label_match = re.search(r"Route:\s*\[([^\]]+)\]", label_solution)
            if not label_match:
                # If we can't parse the label route for some reason, skip
                infeasibility += 1
                continue

            tour_label_str = label_match.group(1)
            tour_label = list(map(int, tour_label_str.split(", ")))

            # Parse the route from the prediction
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No route found in predicted text
                infeasibility += 1
                continue

            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))

            # Check feasibility
            if len(set(tour_list)) != len(set(tour_label)):
                infeasibility += 1
                continue
            if tour_list[-1] != tour_list[0]:
                infeasibility += 1
                continue

            # Parse the objective from the gold (label_solution)
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            solution_distance = float(label_obj_match.group(1))

            # Calculate the LLM's objective
            llm_distance = calculate_total_distance(
                tour_list,
                compute_euclidean_distance_matrix(instances[i])
            )

            gap = (llm_distance - solution_distance) / solution_distance
            gaps.append(gap)
            predicted_objectives.append(llm_distance)
            optimal_objectives.append(solution_distance)

        elif problem == "op":
            # instance structure is [locs, prizes, max_length]
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                infeasibility += 1
                continue

            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))
            
            try:
                total_distance = calculate_total_distance(tour_list, compute_euclidean_distance_matrix(instances[i][0]))
            except:
                infeasibility += 1
                continue
            
            # Check if tour starts from depot
            if tour_list[0] != 0:
                infeasibility += 1
                continue

            # Check if each node is visited at most once
            unique_nodes = set(tour_list)
            if len(tour_list) != len(unique_nodes):
                infeasibility += 1
                continue

            if total_distance > instances[i][2]:
                # The route is too long
                infeasibility += 1
                continue

            # Parse objective from the gold label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                continue
            solution_prize = float(label_obj_match.group(1))

            llm_prize = sum(instances[i][1][j] for j in tour_list)
            gap = (solution_prize - llm_prize) / solution_prize
            gaps.append(gap)
            predicted_objectives.append(llm_prize)
            optimal_objectives.append(solution_prize)

        elif problem == "cvrp":
            # instance structure is [locs, demands, capacity]
            locs = instances[i][0]       # list of (x, y) coordinates
            demands = instances[i][1]    # list of demands for each node
            capacity = instances[i][2]   # vehicle capacity
            # locs = np.array([instances[i][0]] + instances[i][1])*1000
            # demands = [0] + instances[i][2]
            # capacity = instances[i][3]

            # Attempt to parse predicted routes from the text
            # Example: "Routes: [[0,1,2,0], [0,3,4,0]]"
            pred_match = re.search(r"Routes:\s*\[\s*(.*)\]", prediction_solution, re.DOTALL)
            if not pred_match:
                # No match: we cannot parse the solution
                infeasibility += 1
                continue

            routes_str = pred_match.group(1).strip()
            # Safely parse the string as a Python list of lists
            try:
                # Wrap in brackets to ensure it's recognized as a single Python list:
                # "[[0,1,2,0], [0,3,4,0]]" => We just do literal_eval directly.
                predicted_routes = ast.literal_eval(f'[{routes_str}]')
            except (SyntaxError, ValueError):
                infeasibility += 1
                continue

            # Check that we got a list of lists
            if not all(isinstance(r, list) for r in predicted_routes):
                infeasibility += 1
                continue
            

            # 1) Capacity check & route start/end check
            any_route_infeasible = False
            distance_matrix = compute_euclidean_distance_matrix(locs)
            
            try:
                for route in predicted_routes:
                    # Must start/end at depot 0
                    if not route or route[0] != 0 or route[-1] != 0:
                        any_route_infeasible = True
                        break

                    # Sum the demands
                    total_demand = sum(demands[node] for node in route if node != 0)
                    if total_demand > capacity:
                        any_route_infeasible = True
                        break

                if any_route_infeasible:
                    infeasibility += 1
                    continue
            except:
                infeasibility += 1
                continue

            # 2) Check that each customer (1..N-1) is visited exactly once.
            #    (Assuming the problem requires visiting *all* customers.)
            n_customers = len(demands)
            required_customers = set(range(1, n_customers))  # Usually 0 is the depot
            visited_customers = set()
            for route in predicted_routes:
                # Exclude the depot from visited customers
                # (first and last node are typically 0).
                visited_customers.update(route[1:-1])

            if visited_customers != required_customers:
                infeasibility += 1
                continue
            
            # 3) Parse reference objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            solution_cost = float(label_obj_match.group(1))

            # 4) Compute predicted cost
            pred_cost = 0.0
            for route in predicted_routes:
                pred_cost += calculate_total_distance(route, distance_matrix)

            # 5) Compute gap
            gap = (pred_cost - solution_cost) / solution_cost
            gaps.append(gap)
            predicted_objectives.append(pred_cost)
            optimal_objectives.append(solution_cost)

        elif problem == "mvc":

            # 1) Parse predicted cover from "Set: [ ... ]"
            # prediction_solution = '[' + prediction_solution
            pred_match = re.search(r"\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No cover set found
                pred_match = re.search(r"Response: \s*\[([^\]]+)\]", prediction_solution)
                if not pred_match:
                    infeasibility += 1
                    continue
            

            cover_str = pred_match.group(1).strip()
            # Attempt to parse the string as a list of ints
            try:
                predicted_cover = list(map(int, cover_str.split(",")))
            except ValueError:
                # Could not parse the set properly
                infeasibility += 1
                continue

            # 2) Check feasibility: every edge must be covered
            edges_mvc = instances[i][1]  # edges is the second element: (num_nodes, edges, ...)
            cover_set = set(predicted_cover)

            # If any edge is not covered, solution is infeasible
            not_covered = False
            for (u, v) in edges_mvc:
                if u not in cover_set and v not in cover_set:
                    not_covered = True
                    break

            if not_covered:
                infeasibility += 1
                continue

            # 3) Parse the reference (gold) objective from label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                # If we can't parse the gold objective, skip
                infeasibility += 1
                continue

            optimal_cover_size = float(label_obj_match.group(1))

            # 4) Compute gap = (pred_cover_size - optimal_cover_size) / optimal_cover_size
            pred_cover_size = len(cover_set)
            gap = (pred_cover_size - optimal_cover_size) / (optimal_cover_size if optimal_cover_size != 0 else 1e-9)
            gaps.append(gap)
            predicted_objectives.append(pred_cover_size)
            optimal_objectives.append(optimal_cover_size)
            
        elif problem == "mis":
            # 1) Parse predicted independent set from "Set: [ ... ]"
            pred_match = re.search(r"\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No cover set found
                pred_match = re.search(r"Response: \s*\[([^\]]+)\]", prediction_solution)
                if not pred_match:
                    infeasibility += 1
                    continue

            indset_str = pred_match.group(1).strip()
            # Attempt to parse the string as a list of ints
            try:
                predicted_indset = list(map(int, indset_str.split(",")))
            except ValueError:
                # Could not parse the set properly
                infeasibility += 1
                continue
            
            # 2) Check feasibility: no two vertices in the set should be adjacent
            # instance structure is expected to be (num_nodes, edges, ...)
            edges_mis = instances[i][1]  # edges is the second element
            indset = set(predicted_indset)
            
            # Check if any pair of vertices in the independent set are adjacent
            is_independent = True
            for (u, v) in edges_mis:
                if u in indset and v in indset:
                    is_independent = False
                    break
                    
            if not is_independent:
                print("not independent")
                infeasibility += 1
                continue

            # 3) Parse the reference (gold) objective from label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                # If we can't parse the gold objective, skip
                infeasibility += 1
                continue

            optimal_indset_size = float(label_obj_match.group(1))

            # 4) Compute gap = (optimal_indset_size - pred_indset_size) / optimal_indset_size
            # Note: For MIS, we maximize the set size, so the gap is reversed compared to MVC
            pred_indset_size = len(indset)
            gap = (optimal_indset_size - pred_indset_size) / (optimal_indset_size if optimal_indset_size != 0 else 1e-9)
            gaps.append(max(0.0, gap))
            predicted_objectives.append(pred_indset_size)
            optimal_objectives.append(optimal_indset_size)

        elif problem == "pfsp":
            # Parse the predicted job order
            pred_match = re.search(r"Order:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                infeasibility += 1
                continue

            # Extract and parse the job order
            job_order_str = pred_match.group(1)
            try:
                job_order = list(map(int, job_order_str.split(", ")))
            except ValueError:
                infeasibility += 1
                continue

            # Extract instance data
            n_jobs = instances[i].shape[0]
            m_machines = instances[i].shape[1]
            
            # Parse processing times
            processing_times = instances[i].T
            
            
            # Check if the job order contains all jobs exactly once
            expected_jobs = set(range(1, n_jobs + 1))  # Jobs are 1-indexed
            if set(job_order) != expected_jobs or len(job_order) != n_jobs:
                infeasibility += 1
                continue

            # Parse the objective from the gold solution
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            optimal_makespan = float(label_obj_match.group(1))
            
            # Calculate the makespan for the predicted job order
            predicted_makespan = calculate_pfsp_makespan(job_order, processing_times, m_machines)
            
            # Calculate optimality gap
            gap = (predicted_makespan - optimal_makespan) / optimal_makespan
            gaps.append(gap)
            predicted_objectives.append(predicted_makespan)
            optimal_objectives.append(optimal_makespan)
            
        elif problem == "jssp":
            # Parse predicted schedule from the text
            schedule_match = re.search(r"Schedule:\s*(\[\[.*?\]\])", prediction_solution, re.DOTALL)
            if not schedule_match:
                # No schedule found
                infeasibility += 1
                continue
                
            try:
                # Parse the schedule
                schedule_str = schedule_match.group(1).strip()
                schedule = ast.literal_eval(schedule_str)
                
                # Get instance data
                instance = instances[i]
                
                # Basic validation: Check if the prediction has the right structure
                n_jobs = instance.shape[0]
                n_machines = instance.shape[1] // 2
                
                # Check if the number of machines in the schedule matches the instance
                if len(schedule) != n_machines:
                    infeasibility += 1
                    continue
                
                # Check if each machine processes each job exactly once
                valid_structure = True
                for machine_schedule in schedule:
                    if len(machine_schedule) != n_jobs or set(machine_schedule) != set(range(n_jobs)):
                        valid_structure = False
                        break
                
                if not valid_structure:
                    infeasibility += 1
                    continue
                
                # Use the provided get_makespan function to calculate the makespan and check feasibility
                predicted_makespan = get_makespan(instance, schedule)
                
                # If the schedule is infeasible, mark it and continue
                if predicted_makespan == "infeasible":
                    infeasibility += 1
                    continue
                
                # Parse the objective from the gold solution
                label_obj_match = re.search(r"Makespan: (\d+)", label_solution)
                if not label_obj_match:
                    infeasibility += 1
                    continue
                
                optimal_makespan = float(label_obj_match.group(1))
                
                # Calculate optimality gap
                gap = (predicted_makespan - optimal_makespan) / optimal_makespan
                gaps.append(gap)
                predicted_objectives.append(predicted_makespan)
                optimal_objectives.append(optimal_makespan)
                
            except Exception as e:
                # Error in parsing or evaluating the schedule
                infeasibility += 1
                continue

        else:
            raise NotImplementedError(f"Problem {problem} is not implemented!")

    # Calculate gap percentiles statistics
    # set negative gaps to 0
    gaps = [max(0.0, g) for g in gaps]
    
    if len(gaps) > 0:
        below_1pct = sum(1 for g in gaps if g < 0.01)
        below_5pct = sum(1 for g in gaps if g < 0.05)
        below_10pct = sum(1 for g in gaps if g < 0.10)
        
        print("\nGap distribution:")
        print("Below 1%: {} instances ({:.1f}%)".format(
            below_1pct, below_1pct/len(gaps)*100))
        print("Below 5%: {} instances ({:.1f}%)".format(
            below_5pct, below_5pct/len(gaps)*100))
        print("Below 10%: {} instances ({:.1f}%)".format(
            below_10pct, below_10pct/len(gaps)*100))
        
        # Print average objective values
        if predicted_objectives and optimal_objectives:
            avg_predicted = sum(predicted_objectives) / len(predicted_objectives)
            avg_optimal = sum(optimal_objectives) / len(optimal_objectives)
            print("\nObjective values:")
            print("Average predicted: {:.2f}".format(avg_predicted))
            print("Average optimal: {:.2f}".format(avg_optimal))

    # After looping over all predictions, compute feasibility and gap
    # If all are infeasible, handle corner case in gap average
    feasibility_rate = 1 - (infeasibility / len(predictions))
    if len(gaps) == 0:
        mean_gap = float('inf')  # or some fallback
    else:
        mean_gap = sum(gaps) / len(gaps)
        
    std_gap = np.std(gaps) if len(gaps) > 1 else 0.0

    return feasibility_rate, mean_gap, std_gap

def transform_data_op(d):
    """
    Flatten the 'instance' field: [2D_list, 1D_list, float]
    into 3 separate columns.
    """
    # If "instance" is the nested list
    inst_2d, inst_1d, inst_float = d["instance"]
    d["instance_coords"] = inst_2d      # e.g. 2D list of [x,y]
    d["instance_prizes"] = inst_1d      # e.g. 1D list
    d["instance_max_dist"] = inst_float # float
    # remove the original field to avoid Arrow conflict
    del d["instance"]
    return d


def transform_data_cvrp(d):
    """
    Transform CVRP data into the required format.
    Expected input format:
    - Node information in input string
    - vehicle_capacity as a float
    """
    inst_2d, inst_1d, inst_float = d["instance"]
    # Store the processed data
    d["instance_coords"] = inst_2d       # 2D list of [x,y] coordinates
    d["instance_demands"] = inst_1d     # 1D list of demands
    d["instance_capacity"] = inst_float   # float vehicle capacity
    del d["instance"]
    
    return d


def load_single_dict_dataset(json_path, problem, num_samples=None):
    """
    Load and transform dataset from JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        problem (str): Problem type ('op', 'tsp', 'mvc', or 'cvrp')
        num_samples (int, optional): Number of samples to load
    """
    import json
    from datasets import Dataset
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If your top-level JSON is a single dict -> wrap in list of length 1
    # If it's already a list -> we can process each dict in that list.
    if isinstance(data, dict):
        data = [data]
    
    # Limit samples if specified
    if num_samples is not None:
        data = data[:num_samples]
    
    # Transform data based on problem type
    if problem == 'op':
        transformed_data = [transform_data_op(d) for d in data]
    elif problem == 'cvrp':
        transformed_data = [transform_data_cvrp(d) for d in data]
    else:
        # For other problems, no transformation needed yet
        transformed_data = data

    return Dataset.from_list(transformed_data)


def load_train_dict_dataset(json_path, problem):
    """
    Load and transform training dataset from JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        problem (str): Problem type ('op', 'tsp', 'mvc', or 'cvrp')
    """
    import json
    from datasets import Dataset
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If the top-level JSON is a single dict -> wrap in list of length 1
    # If it's already a list -> we can process each dict in that list.
    if isinstance(data, dict):
        # Flatten the single dict
        if problem == 'op':
            data = [transform_data_op(data)]
        elif problem == 'tsp':
            pass
        elif problem == 'cvrp':
            data = [transform_data_cvrp(data)]
    elif isinstance(data, list):
        # Flatten each dict in the list
        if problem == 'op':
            data = [transform_data_op(d) for d in data]
        elif problem == 'tsp':
            pass
        elif problem == 'cvrp':
            data = [transform_data_cvrp(d) for d in data]
    else:
        raise ValueError("JSON must be either a dict or a list of dicts.")
    return Dataset.from_list(data)


def get_dataset(problem, tokenizer=None, num_samples=None, train=True):
    """
    Get formatted dataset for RL training.
    
    Args:
        problem (str): Problem type ('op', 'tsp', 'mvc', or 'cvrp')
        tokenizer: Tokenizer for formatting prompts
        num_samples (int, optional): Number of samples to load
        train (bool): Whether to load training dataset
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    # Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance. 
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective. You can reason about how to achieve that in the <reasoning></reasoning> section.

    ### Instruction:{}

    ### Input:{}

    ### Response:"""
    EOS_TOKEN = tokenizer.eos_token if tokenizer else ""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        prompts = []
        completions = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            prompt = alpaca_prompt.format(instruction, input_text)
            prompts.append(prompt)
            completions.append(output + EOS_TOKEN)
        return {
            "prompt": prompts,
            "ground_truth": completions,
            # You could also copy instance_coords, instance_prizes, etc.
        }
    
    if not train:
        eval_json_path = f"./data_rl/{problem}/eval/test.json"
        # eval_json_path = f'./data_compare_with_heu/{problem}/test_small.json'
        # eval_json_path = f'./data_compare_lncs/{problem}/data.json'
        eval_dataset = load_single_dict_dataset(eval_json_path, problem, num_samples=num_samples)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
        
        # Limit number of samples if specified
        if num_samples is not None:
            eval_dataset = eval_dataset.select(range(num_samples))
            
        return None, eval_dataset
    else:
        train_json_path = f"./data_rl/{problem}/train/train_rl.json"
        eval_json_path = f"./data_rl/{problem}/eval/test.json"

        # Create a dataset from flattened JSON
        train_dataset = load_train_dict_dataset(train_json_path, problem)
        eval_dataset = load_single_dict_dataset(eval_json_path, problem, num_samples=num_samples)

        # Now map your formatting function
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

        return train_dataset, eval_dataset