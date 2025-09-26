import re
import ast
from utils import calculate_total_distance, compute_euclidean_distance_matrix, load_pkl_dataset, extract_predicted_solution, get_makespan
import numpy as np

def parse_solution_jssp(response):
    """Parse the JSSP solution from the response."""
    schedule_match = re.search(r"Schedule:\s*(\[\[.+?\]\])", response, re.DOTALL)
    if not schedule_match:
        return None
    
    makespan_match = re.search(r"Makespan:\s*(\d+)", response)
    if not makespan_match:
        return None
    
    schedule_str = schedule_match.group(1)
    makespan_str = makespan_match.group(1)
    
    try:
        # Convert string representation of schedule to a list of lists
        schedule = ast.literal_eval(schedule_str)
        makespan = int(makespan_str)
        return {"schedule": schedule, "makespan": makespan}
    except (SyntaxError, ValueError):
        return None

def feasibility_reward_func_tsp(completions, instance, **kwargs) -> list[float]:
    scores = []
    responses = completions
    for i, response in enumerate(responses):
        pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
        if pred_match:
            tour_str = pred_match.group(1)
            try:
                tour_list = list(map(int, tour_str.split(", ")))
            except:
                # Error in parsing the tour list
                scores.append(0.0)
                continue
            # Check feasibility
            if len(set(tour_list)) != len(instance[i]):
                scores.append(0.0)
                continue
            if tour_list[-1] != tour_list[0]:
                scores.append(0.0)
                continue
            scores.append(2.0)
        else:
            # No solution
            scores.append(0.0)
            continue
    return scores


def optimality_reward_func_tsp(completions, ground_truth, instance, **kwargs) -> list[float]:
    scores = []
    responses = completions
    feasible_rewards = feasibility_reward_func_tsp(completions, instance)
    for i, response in enumerate(responses):
        if feasible_rewards[i] != 2.0:
            scores.append(0.0)
            continue
        pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
        tour_str = pred_match.group(1)
        tour_list = list(map(int, tour_str.split(", ")))
        llm_distance = calculate_total_distance(tour_list, compute_euclidean_distance_matrix(np.array(instance[i])))
        label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
        solution_distance = float(label_obj_match.group(1))
        gap = (llm_distance - solution_distance) / solution_distance
        scores.append(max(0.0, 1.0 - gap))
    return scores



def feasibility_reward_func_op(completions, coords, max_dist, **kwargs) -> list[float]:
    """
    Calculate the feasibility reward for the Orienteering Problem. The infeasibility possibilities are:
    1. The route is not given
    2. The route does not start from the depot
    3. The same node is visited more than once
    4. The total distance exceeds the maximum route length
    """
    scores = []
    # responses = [generation[0]['generated_text'] for generation in completions]
    responses = completions

    for i, response in enumerate(responses):
        pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
        if pred_match:
            tour_str = pred_match.group(1)
            try:
                tour_list = list(map(int, tour_str.split(", ")))
            except:
                # Error in parsing the tour list
                scores.append(0.0)
                continue

            if tour_list[0] != 0:
                # the tour does not start from the depot, infeasible!
                scores.append(0.0)
                continue

            # each node is visited only once
            if len(tour_list) != len(set(tour_list)):
                scores.append(0.0)
                continue

            try:
                total_distance = calculate_total_distance(tour_list,
                                                          compute_euclidean_distance_matrix(np.array(coords[i])))
            except:
                # Error in calculating the total distance
                scores.append(0.0)
                continue

            if total_distance > max_dist[i]:
                # print(f"Total distance {total_distance} exceeds max distance {max_dist[i]}")
                # A list of solution but not feasible due to the total distance constraint
                scores.append(1.0)
            else:
                scores.append(2.0)
            continue
        else:
            # No solution
            scores.append(0.0)
            continue
    return scores



def optimality_reward_func_op(completions, ground_truth, coords, max_dist, prizes, **kwargs) -> list[float]:
    """
    Calculate the optimality reward for the Orienteering Problem. The optimality is measured by the objective value.
    """
    scores = []
    # responses = [generation[0]['generated_text'] for generation in completions]
    responses = completions
    feasible_rewards = feasibility_reward_func_op(completions, coords, max_dist)

    for i, response in enumerate(responses):
        if feasible_rewards[i] != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue
        else:
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))
            llm_prize = sum(prizes[i][j] for j in tour_list)
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            solution_prize = float(label_obj_match.group(1))
            gap = (solution_prize - llm_prize) / solution_prize
            scores.append(max(0.0, 1.0 - gap))
    return scores


def feasibility_reward_func_cvrp(completions, instance_coords, instance_demands, instance_capacity, **kwargs) -> list[float]:
    """
    Calculate the feasibility reward for the CVRP. The infeasibility possibilities are:
    1. Routes are not properly formatted
    2. Routes don't start/end at depot (0)
    3. Vehicle capacity is exceeded
    4. Not all customers are visited exactly once
    """
    scores = []
    responses = completions

    for i, response in enumerate(responses):
        # Parse predicted routes from the text (e.g., "Routes: [[0,1,2,0], [0,3,4,0]]")
        pred_match = re.search(r"Routes:\s*\[\s*(.*)\]", response, re.DOTALL)
        if not pred_match:
            # No match: cannot parse the solution
            scores.append(0.0)
            continue

        routes_str = pred_match.group(1).strip()
        try:
            # Parse the string as a Python list of lists
            predicted_routes = ast.literal_eval(f'[{routes_str}]')
            if not all(isinstance(r, list) for r in predicted_routes):
                scores.append(0.0)
                continue
        except (SyntaxError, ValueError):
            scores.append(0.0)
            continue

        # Get instance data
        # demands = instance[i][1]    # list of demands for each node
        # capacity = instance[i][2]   # vehicle capacity
        demands = instance_demands[i]
        capacity = instance_capacity[i]


        # Check feasibility conditions
        is_feasible = True

        try:
            # 1. Check each route starts/ends at depot and respects capacity
            for route in predicted_routes:
                # Must start/end at depot 0
                if not route or route[0] != 0 or route[-1] != 0:
                    is_feasible = False
                    break

                # Check capacity constraint
                total_demand = sum(demands[node] for node in route if node != 0)
                if total_demand > capacity:
                    is_feasible = False
                    break

            if not is_feasible:
                scores.append(0.0)
                continue

            # 2. Check that each customer is visited exactly once
            n_customers = len(demands)
            required_customers = set(range(1, n_customers))  # 0 is the depot
            visited_customers = set()

            for route in predicted_routes:
                # Exclude the depot from visited customers
                visited_customers.update(route[1:-1])

            if visited_customers != required_customers:
                scores.append(0.0)
                continue
        except:
            scores.append(0.0)
            continue

        # All feasibility checks passed
        scores.append(2.0)

    return scores

def optimality_reward_func_cvrp(completions, ground_truth, instance_coords, instance_demands, instance_capacity,
                                **kwargs) -> list[float]:
    """
    Calculate the optimality reward for the CVRP. The optimality is measured by the total route length.
    """
    scores = []
    responses = completions
    feasible_rewards = feasibility_reward_func_cvrp(completions, instance_coords, instance_demands, instance_capacity)

    for i, (response, is_feasible) in enumerate(zip(responses, feasible_rewards)):
        if is_feasible != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue

        # Parse predicted routes
        pred_match = re.search(r"Routes:\s*\[\s*(.*)\]", response, re.DOTALL)
        routes_str = pred_match.group(1).strip()
        predicted_routes = ast.literal_eval(f'[{routes_str}]')

        # Calculate total distance for predicted solution
        distance_matrix = compute_euclidean_distance_matrix(np.array(instance_coords[i]))
        pred_cost = 0.0
        for route in predicted_routes:
            pred_cost += calculate_total_distance(route, distance_matrix)

        # Parse the reference (gold) objective
        label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
        if not label_obj_match:
            scores.append(0.0)
            continue

        solution_cost = float(label_obj_match.group(1))

        # Compute gap = (pred_cost - solution_cost) / solution_cost
        # Convert to a score between 0 and 1
        gap = (pred_cost - solution_cost) / solution_cost
        scores.append(max(0.0, 1.0 - gap))

    return scores


def feasibility_reward_func_mvc(completions, instance, **kwargs) -> list[float]:
    scores = []
    responses = completions

    for i, response in enumerate(responses):
        # Parse predicted cover from "Set: [ ... ]"
        response = extract_predicted_solution(response)
        pred_match = re.search(r"\s*\[([^\]]+)\]", response)
        if not pred_match:
            # No cover set found
            pred_match = re.search(r"Response: \s*\[([^\]]+)\]", response)
            if not pred_match:
                # No cover set found
                scores.append(0.0)
                continue

        # Parse the vertex cover set
        cover_str = pred_match.group(1).strip()
        try:
            predicted_cover = list(map(int, cover_str.split(",")))
            cover_set = set(predicted_cover)
        except ValueError:
            # Could not parse the set properly
            scores.append(0.0)
            continue

        # Check feasibility: every edge must be covered
        edges_mvc = instance[i]['edges']  # edges is the second element

        # If any edge is not covered, solution is infeasible
        is_feasible = True
        for (u, v) in edges_mvc:
            if u not in cover_set and v not in cover_set:
                is_feasible = False
                break

        scores.append(2.0 if is_feasible else 0.0)

    return scores


def optimality_reward_func_mvc(completions, ground_truth, instance, **kwargs) -> list[float]:
    scores = []
    feasible_rewards = feasibility_reward_func_mvc(completions, instance)

    for i, (response, is_feasible) in enumerate(zip(completions, feasible_rewards)):
        if is_feasible != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue

        # Parse predicted cover
        response = extract_predicted_solution(response)
        pred_match = re.search(r"\s*\[([^\]]+)\]", response)
        cover_str = pred_match.group(1).strip()
        predicted_cover = list(map(int, cover_str.split(",")))
        pred_cover_size = len(set(predicted_cover))

        # Parse the reference (gold) objective
        label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
        if not label_obj_match:
            scores.append(0.0)
            continue

        optimal_cover_size = float(label_obj_match.group(1))

        # Compute gap = (pred_cover_size - optimal_cover_size) / optimal_cover_size
        # Convert to a score between 0 and 1
        gap = (pred_cover_size - optimal_cover_size) / (optimal_cover_size if optimal_cover_size != 0 else 1e-9)
        scores.append(max(0.0, 1.0 - gap))

    return scores


def feasibility_reward_func_mis(completions, instance, **kwargs) -> list[float]:
    scores = []
    responses = completions

    for i, response in enumerate(responses):
        pred_match = re.search(r"Response: \s*\[([^\]]+)\]", response)
        if not pred_match:
            # No independent set found
            scores.append(0.0)
            continue

        # Parse the independent set
        indset_str = pred_match.group(1).strip()
        try:
            predicted_indset = list(map(int, indset_str.split(",")))
            indset = set(predicted_indset)
        except ValueError:
            # Could not parse the set properly
            scores.append(0.0)
            continue
        
        # Check feasibility: no two vertices in the set should be adjacent
        edges_mis = instance[i]['edges']

        # If any edge connects two vertices in the independent set, solution is infeasible
        is_independent = True
        for (u, v) in edges_mis:
            if u in indset and v in indset:
                # print(f"Edge ({u}, {v}) connects two vertices in the independent set.")
                is_independent = False
                break

        scores.append(2.0 if is_independent else 0.0)

    return scores


def optimality_reward_func_mis(completions, ground_truth, instance, **kwargs) -> list[float]:
    scores = []
    feasible_rewards = feasibility_reward_func_mis(completions, instance)

    for i, (response, is_feasible) in enumerate(zip(completions, feasible_rewards)):
        if is_feasible != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue

        # Parse predicted independent set
        pred_match = re.search(r"Response: \s*\[([^\]]+)\]", response)
        indset_str = pred_match.group(1).strip()
        predicted_indset = list(map(int, indset_str.split(",")))
        pred_indset_size = len(set(predicted_indset))

        # Parse the reference (gold) objective
        label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
        if not label_obj_match:
            scores.append(0.0)
            continue

        optimal_indset_size = float(label_obj_match.group(1))

        # Compute gap = (optimal_indset_size - pred_indset_size) / optimal_indset_size
        # For MIS, larger independent sets are better, so gap is reversed compared to MVC
        # Convert to a score between 0 and 1
        gap = (optimal_indset_size - pred_indset_size) / (optimal_indset_size if optimal_indset_size != 0 else 1e-9)
        scores.append(max(0.0, 1.0 - gap))

    return scores


def feasibility_reward_func_pctsp(completions, coords, prizes, max_distance, depot=0, **kwargs) -> list[float]:
    """
    Calculate the feasibility reward for the Prize-Collecting TSP (PCTSP). The infeasibility possibilities are:
    1. The route is not given or cannot be parsed
    2. The route does not start and end at the depot
    3. The same node is visited more than once (except depot at start/end)
    4. The total distance exceeds the maximum distance constraint
    """
    scores = []
    responses = completions

    for i, response in enumerate(responses):
        pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
        if pred_match:
            tour_str = pred_match.group(1)
            try:
                tour_list = list(map(int, tour_str.split(", ")))
            except:
                # Error in parsing the tour list
                scores.append(0.0)
                continue

            # Check if route starts and ends at depot
            if len(tour_list) < 2 or tour_list[0] != depot:
                scores.append(0.0)
                continue

            # Check for repeated nodes (except depot at start/end)
            route_without_endpoints = tour_list[1:-1]
            if len(route_without_endpoints) != len(set(route_without_endpoints)):
                scores.append(0.0)
                continue

            # Check if depot appears in the middle (not allowed)
            if depot in route_without_endpoints:
                scores.append(0.0)
                continue

            try:
                total_distance = calculate_total_distance(tour_list,
                                                          compute_euclidean_distance_matrix(np.array(coords[i])))
            except:
                # Error in calculating the total distance
                scores.append(0.0)
                continue

            if total_distance > max_distance[i]:
                # Route violates distance constraint
                scores.append(1.0)  # Partial score for valid format but infeasible
            else:
                scores.append(2.0)  # Fully feasible
            continue
        else:
            # No solution
            scores.append(0.0)
            continue
    return scores


def optimality_reward_func_pctsp(completions, ground_truth, coords, prizes, max_distance, depot=0, **kwargs) -> list[float]:
    """
    Calculate the optimality reward for the Prize-Collecting TSP (PCTSP).
    The objective is to minimize total distance + penalties for unvisited nodes (penalty = 1000 per unvisited node).
    """
    scores = []
    responses = completions
    feasible_rewards = feasibility_reward_func_pctsp(completions, coords, prizes, max_distance, depot)

    for i, response in enumerate(responses):
        if feasible_rewards[i] != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue
        else:
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))
            
            # Calculate travel distance
            try:
                total_distance = calculate_total_distance(tour_list,
                                                          compute_euclidean_distance_matrix(np.array(coords[i])))
            except:
                scores.append(0.0)
                continue
            
            # Calculate number of visited nodes (excluding depot)
            visited_nodes = len(tour_list) - 2  # Exclude depot at start and end
            total_nodes = len(coords[i]) - 1  # Total nodes excluding depot
            unvisited_nodes = total_nodes - visited_nodes
            
            # Calculate objective (distance + penalties for unvisited nodes)
            penalty_per_node = 1000
            llm_objective = total_distance + (unvisited_nodes * penalty_per_node)
            
            # Parse ground truth objective
            label_obj_match = re.search(r"Objective:\s*([\d.-]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue
            
            solution_objective = float(label_obj_match.group(1))
            
            # For PCTSP, lower objective is better (minimization problem)
            if solution_objective <= 0:
                # Handle edge case for very small or negative objectives
                if llm_objective <= solution_objective:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            else:
                gap = (llm_objective - solution_objective) / solution_objective
                scores.append(min(1.0, max(0.0, 1.0 - gap)))  # Clamp to [0, 1]
            
    return scores


def feasibility_reward_func_jssp(completions, instance, **kwargs):
    """
    Calculate the feasibility reward for the Job Shop Scheduling Problem (JSSP).
    
    For JSSP, the solution must satisfy:
    1. Each job must have all its operations scheduled
    2. Operations of a job must be processed in order
    3. Each machine can process only one job at a time
    
    Returns a score between 0 and 1 based on how close the solution is to feasibility.
    """
    scores = []
    
    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,         # Solution can be parsed correctly
            "job_coverage": 0.2,  # All jobs are scheduled
            "machine_validity": 0.2,  # No machine conflicts
            "precedence": 0.4,    # Operations of a job processed in order
        }
        
        score = 0.0
        parsed_solution = parse_solution_jssp(response)
        
        if parsed_solution is None:
            scores.append(0.0)
            continue
            
        # Solution can be parsed
        score += weights["parse"]
        
        schedule = parsed_solution["schedule"]
        
        # Get the instance data for this example
        instance_arr = np.array(instance[i])
        
        # Check job coverage: all jobs should appear exactly once in each machine's schedule
        try:
            n_jobs = int(instance_arr['n'])
            n_machines = int(instance_arr['m'])
        except:
            # For the case where instance is numpy array
            n_jobs = instance_arr.shape[0]
            n_machines = instance_arr.shape[1] // 2
        
        # Check if the number of machines in the solution matches the instance
        if len(schedule) != n_machines:
            scores.append(score)  # Only get points for parsing
            continue
        
        # Check if all jobs are scheduled on all machines
        all_jobs_scheduled = True
        for machine_schedule in schedule:
            if len(machine_schedule) != n_jobs or set(machine_schedule) != set(range(n_jobs)):
                all_jobs_scheduled = False
                break
                
        if all_jobs_scheduled:
            score += weights["job_coverage"]
        
        # Check machine validity - no overlapping operations on same machine
        # This is inherently satisfied by the schedule format, as each machine 
        # processes jobs sequentially. We're primarily checking if the schedule format is valid.
        valid_machine_scheduling = all(len(machine_schedule) == n_jobs for machine_schedule in schedule)
        
        if valid_machine_scheduling:
            score += weights["machine_validity"]
            
        # Check precedence constraints using get_makespan function from utils
        try:
            # Get the real makespan which automatically checks precedence constraints
            real_makespan = get_makespan(instance_arr, schedule)
            
            # If get_makespan returns a number (not "infeasible"), the schedule respects precedence constraints
            if real_makespan != "infeasible":
                score += weights["precedence"]
        except Exception as e:
            # Error in get_makespan likely means precedence constraints are violated
            pass  # No additional points for precedence
        
        scores.append(score)
    
    return scores

def optimality_reward_func_jssp(completions, ground_truth, instance, **kwargs):
    """
    Calculate the optimality reward for the Job Shop Scheduling Problem (JSSP).
    
    The optimality is measured by the makespan compared to the optimal solution.
    For JSSP, shorter makespan is better.
    """
    scores = []
    feasibility_scores = feasibility_reward_func_jssp(completions, instance)
    
    for i, (response, feasibility_score) in enumerate(zip(completions, feasibility_scores)):
        # If solution is not feasible, give no optimality reward
        if feasibility_score < 0.99:  # JSSP requires high feasibility to be meaningful
            scores.append(0.0)
            continue
        
        parsed_solution = parse_solution_jssp(response)
        if parsed_solution is None:
            scores.append(0.0)
            continue
            
        try:
            # Get the instance data for this example
            instance_arr = np.array(instance[i])
            
            # Get the schedule from parsed solution
            schedule = parsed_solution["schedule"]
            
            # Calculate the real makespan using the get_makespan function from utils
            real_makespan = get_makespan(instance_arr, schedule)
            
            # Check if the schedule is feasible
            if real_makespan == "infeasible":
                scores.append(0.0)
                continue
                
            # Parse the reference (optimal) makespan
            label_makespan_match = re.search(r"Makespan:\s*(\d+)", ground_truth[i])
            if not label_makespan_match:
                scores.append(0.0)
                continue
                
            optimal_makespan = float(label_makespan_match.group(1))
            
            # For JSSP, shorter makespan is better, so calculate inverse ratio
            if real_makespan < optimal_makespan:
                # If prediction is better than ground truth (rare but possible), give full score
                score = 1.0
            else:
                # Calculate gap-based score with a smooth function
                gap = (real_makespan - optimal_makespan) / (optimal_makespan if optimal_makespan != 0 else 1e-9)
            scores.append(max(0.0, 1.0 - gap))

        except Exception as e:
            # Error in calculating makespan
            scores.append(0.0)
    
    return scores
