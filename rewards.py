import re
import ast
from utils import calculate_total_distance, compute_euclidean_distance_matrix, load_pkl_dataset, get_makespan
import numpy as np


def parse_solution_mis(response):
    """Parse the MIS solution from the response."""
    pred_match = re.search(r"Response: \s*\[([^\]]+)\]", response)
    if not pred_match:
        pred_match = re.search(r"\s*\[([^\]]+)\]", response)
        if not pred_match:
            return None
    
    indset_str = pred_match.group(1).strip()
    try:
        predicted_indset = [int(x.strip()) for x in indset_str.split(",")]
        return predicted_indset
    except ValueError:
        return None


def parse_solution_vrp(response):
    """Parse the solution from the response."""
    pred_match = re.search(r"Routes:\s*\[\s*(.*)\]", response, re.DOTALL)
    if not pred_match:
        return None

    routes_str = pred_match.group(1).strip()
    try:
        predicted_routes = ast.literal_eval(f'[{routes_str}]')
        if not all(isinstance(r, list) for r in predicted_routes):
            return None
        return predicted_routes
    except (SyntaxError, ValueError):
        return None


def parse_solution_op(response):
    """Parse the Orienteering Problem solution from the response."""
    pred_match = re.search(r"Route:\s*\[([^\]]+)\]", response)
    if not pred_match:
        return None

    tour_str = pred_match.group(1)
    try:
        tour_list = list(map(int, tour_str.split(", ")))
        return tour_list
    except (ValueError, SyntaxError):
        return None


def feasibility_reward_func_op(completions, instance_coords, instance_max_dist, **kwargs):
    """
    Calculate the feasibility reward for the Orienteering Problem with more granular feedback.

    Returns a score between 0 and 1 based on how close the solution is to feasibility.

    The infeasibility possibilities are:
    1. The route is not given or cannot be parsed
    2. The route does not start from the depot (node 0)
    3. The same node is visited more than once
    4. The total distance exceeds the maximum route length
    """
    scores = []

    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,  # Solution can be parsed correctly
            "start_depot": 0.1,  # Tour starts from depot
            "unique_visits": 0.2,  # Each node visited at most once
            "distance_constraint": 0.5  # Total distance within limit
        }

        score = 0.0
        tour_list = parse_solution_op(response)

        if tour_list is None:
            scores.append(0.0)
            continue

        # Solution can be parsed
        score += weights["parse"]

        # Check if tour starts from depot
        if tour_list and tour_list[0] == 0:
            score += weights["start_depot"]

        # Check if each node is visited at most once
        unique_nodes = set(tour_list)
        if len(tour_list) == len(unique_nodes):
            score += weights["unique_visits"]


        # Check distance constraint
        try:
            distance_matrix = compute_euclidean_distance_matrix(np.array(instance_coords[i]))
            total_distance = calculate_total_distance(tour_list, distance_matrix)

            if total_distance <= instance_max_dist[i]:
                score += weights["distance_constraint"]
            else:
                score += 0.0  # No credit for exceeding distance
        except:
            # Error in calculating distance
            pass

        scores.append(score)

    return scores


def optimality_reward_func_op(completions, ground_truth, instance_coords, instance_max_dist, instance_prizes, **kwargs):
    """
    Calculate the optimality reward for the Orienteering Problem with improved gradient.

    The optimality is measured by the total prize collected compared to the optimal solution.
    """
    scores = []
    feasible_scores = feasibility_reward_func_op(completions, instance_coords, instance_max_dist)

    for i, (response, feasibility_score) in enumerate(zip(completions, feasible_scores)):
        # If solution has very low feasibility score, give no optimality reward
        if feasibility_score < 1:  # Only reward solutions that are mostly feasible
            scores.append(0.0)
            continue

        tour_list = parse_solution_op(response)
        if tour_list is None:
            scores.append(0.0)
            continue

        # Calculate total prize collected
        try:
            llm_prize = sum(instance_prizes[i][j] for j in tour_list)

            # Parse the reference (optimal) objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue

            solution_prize = float(label_obj_match.group(1))

            # Compute prize ratio: LLM_prize / optimal_prize
            prize_ratio = 2*llm_prize / max(0.1, solution_prize)  # Avoid division by zero

            # Use a smooth function to map prize ratio to [0, 1]
            # This is better than the linear scaling as it provides a smoother gradient

            scores.append(prize_ratio)

        except Exception as e:
            # Error in calculating prizes
            scores.append(0.0)

    return scores





def feasibility_reward_func_cvrp(completions, instance_coords, instance_demands, instance_capacity, **kwargs):
    """
    Calculate the feasibility reward for the CVRP with more granular feedback.

    Returns a score between 0 and 1 based on how close the solution is to feasibility.
    """
    scores = []

    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,  # Solution can be parsed
            "depot_constraint": 0.1,  # Routes start/end at depot
            "capacity": 0.6,  # Capacity constraints satisfied
            "coverage": 0.1  # All customers visited exactly once
        }

        score = 0.0
        predicted_routes = parse_solution_vrp(response)

        if predicted_routes is None:
            scores.append(0.0)
            continue

        # Solution can be parsed
        score += weights["parse"]

        # Get instance data
        demands = instance_demands[i]
        capacity = instance_capacity[i]

        # Check depot constraint
        depot_ok = True
        for route in predicted_routes:
            if not route or route[0] != 0 or route[-1] != 0:
                depot_ok = False
                break

        if depot_ok:
            score += weights["depot_constraint"]

        # Check capacity constraint
        try:
            capacity_ok = True
            for route in predicted_routes:
                total_demand = sum(demands[node] for node in route if node != 0)
                if total_demand > capacity:
                    capacity_ok = False
                    break

            if capacity_ok:
                score += weights["capacity"]
        except:
            # Error in checking capacity constraint
            pass

        # Check coverage constraint
        try:
            n_customers = len(demands)
            required_customers = set(range(1, n_customers))
            visited_customers = set()

            for route in predicted_routes:
                visited_customers.update(route[1:-1])

            if visited_customers == required_customers:
                score += weights["coverage"]
            else:
                # Partial credit for coverage based on how many customers are correctly visited
                coverage_ratio = len(visited_customers.intersection(required_customers)) / len(required_customers)
                score += weights["coverage"] * coverage_ratio
        except:
            # Error in checking coverage constraint
            pass

        scores.append(score)

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
        if is_feasible < 0.99:
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
        # scores.append(max(0.0, 1.0 - gap))
        scores.append(max(0.0, 1.0 / (1.0 + gap)))

    return scores


def feasibility_reward_func_mvc(completions, instance, **kwargs) -> list[float]:
    """
    Calculate the feasibility reward for the Minimum Vertex Cover (MVC) problem with more granular feedback.
    
    Returns a score between 0 and 1 based on how close the solution is to feasibility.
    
    For MVC, feasibility requires every edge to be covered by at least one vertex in the cover set.
    """
    scores = []
    
    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,               # Solution can be parsed correctly
            "edge_coverage": 0.8,       # All edges are covered
        }
        score = 0.0
        predicted_cover = parse_solution_mis(response)
        
        if predicted_cover is None:
            scores.append(0.0)
            continue
            
        # Solution can be parsed
        score += weights["parse"]
        
        # Convert to set for faster lookups
        cover_set = set(predicted_cover)
        
        # Check edge coverage: every edge must be covered by at least one vertex
        edges_mvc = instance[i]['edges']
        
        # Count uncovered edges
        uncovered = 0
        for u, v in edges_mvc:
            if u not in cover_set and v not in cover_set:
                uncovered += 1
        
        # Calculate edge coverage score
        if uncovered == 0:
            # Perfect coverage
            score += weights["edge_coverage"]
        else:
            # Partial credit based on proportion of covered edges
            # coverage_ratio = 1.0 - (uncovered / max(1, len(edges_mvc)))
            # score += weights["edge_coverage"] * coverage_ratio
            score = 0.0  # No credit for uncovered edges
            
        scores.append(score)
    
    return scores


def optimality_reward_func_mvc(completions, ground_truth, instance, **kwargs) -> list[float]:
    """
    Calculate the optimality reward for the Minimum Vertex Cover (MVC) problem with improved gradient.
    
    The optimality is measured by the size of the vertex cover compared to the optimal solution.
    For MVC, smaller covers are better.
    """
    scores = []
    feasibility_scores = feasibility_reward_func_mvc(completions, instance)
    
    for i, (response, feasibility_score) in enumerate(zip(completions, feasibility_scores)):
        # If solution has very low feasibility score, give minimal optimality reward
        if feasibility_score < 0.9:  # MVC requires high feasibility to be meaningful
            scores.append(0)
            continue
        
        predicted_cover = parse_solution_mis(response)
        if predicted_cover is None:
            scores.append(0.0)
            continue
        
        try:
            # Size of the predicted cover
            pred_cover_size = len(set(predicted_cover))
            
            # Parse the reference (optimal) objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue
                
            optimal_cover_size = float(label_obj_match.group(1))
            
            # For MVC, smaller is better, so calculate the inverse ratio with smoothing
            if pred_cover_size < optimal_cover_size:
                # If prediction is better than ground truth (rare but possible), give full score
                score = 1.0
            else:
                # Calculate gap-based score with a smooth function
                gap = (pred_cover_size - optimal_cover_size) / max(1.0, optimal_cover_size)
                score = 1.0 / (1.0 + gap)
            
            scores.append(score)
            
        except Exception as e:
            # Error in calculating sizes
            scores.append(0.0)
    
    return scores



def feasibility_reward_func_tsp(completions, instance, **kwargs):
    """
    Calculate the feasibility reward for the TSP with more granular feedback.
    
    Returns a score between 0 and 1 based on how close the solution is to feasibility.
    
    The infeasibility possibilities are:
    1. The route is not given or cannot be parsed
    2. The route does not visit all nodes exactly once
    3. The route is not a complete tour (doesn't return to start)
    """
    scores = []
    
    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,               # Solution can be parsed correctly
            "visit_all_nodes": 0.5,     # All nodes visited exactly once
            "complete_tour": 0.3        # Tour returns to starting point
        }
        
        score = 0.0
        tour_list = parse_solution_op(response)
        
        if tour_list is None:
            scores.append(0.0)
            continue
            
        # Solution can be parsed
        score += weights["parse"]
        
        # Check if all nodes are visited exactly once
        n_nodes = len(instance[i])
        unique_nodes = set(tour_list)
        
        # Remove the duplicate start/end node for proper counting
        if len(tour_list) > 0 and tour_list[0] == tour_list[-1]:
            nodes_in_tour = len(tour_list) - 1
        else:
            nodes_in_tour = len(tour_list)
        
        # Check uniqueness - each node should appear exactly once except start/end
        if len(unique_nodes) == nodes_in_tour and nodes_in_tour == n_nodes:
            score += weights["visit_all_nodes"]
        else:
            # Partial credit based on coverage ratio
            coverage_ratio = min(1.0, len(unique_nodes) / n_nodes)
            score += weights["visit_all_nodes"] * coverage_ratio
        
        # Check if tour returns to starting point
        if len(tour_list) >= 2 and tour_list[0] == tour_list[-1]:
            score += weights["complete_tour"]
        
        scores.append(score)
    
    return scores

def optimality_reward_func_tsp(completions, ground_truth, instance, **kwargs):
    """
    Calculate the optimality reward for the TSP with improved gradient.
    
    The optimality is measured by the total tour distance compared to the optimal solution.
    """
    scores = []
    feasibility_scores = feasibility_reward_func_tsp(completions, instance)
    
    for i, (response, feasibility_score) in enumerate(zip(completions, feasibility_scores)):
        # If solution has very low feasibility score, give minimal optimality reward
        if feasibility_score < 0.9:  # TSP requires high feasibility to be meaningful
            # Give a very small proportional reward to guide learning
            scores.append(0.1 * feasibility_score)
            continue
        
        tour_list = parse_solution_op(response)
        if tour_list is None:
            scores.append(0.0)
            continue
        
        try:
            # Calculate tour distance
            distance_matrix = compute_euclidean_distance_matrix(np.array(instance[i]))
            llm_distance = calculate_total_distance(tour_list, distance_matrix)
            
            # Parse the reference (optimal) objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue
                
            solution_distance = float(label_obj_match.group(1))
            
            # Compute gap ratio
            gap = (llm_distance - solution_distance) / solution_distance
            
            # Use a smoother function to map gap to [0, 1]
            # This provides more gradient for improvement
            score = max(0.0, 1.0 / (1.0 + gap))
            
            scores.append(score)
            
        except Exception as e:
            # Error in calculating distances
            scores.append(0.0)
    
    return scores

def feasibility_reward_func_mis(completions, instance, **kwargs):
    """
    Calculate the feasibility reward for the Maximum Independent Set (MIS) problem with more granular feedback.
    
    Returns a score between 0 and 1 based on how close the solution is to feasibility.
    
    For MIS, feasibility requires no two vertices in the set to be adjacent.
    """
    scores = []
    
    for i, response in enumerate(completions):
        # Assign weights to different feasibility aspects
        weights = {
            "parse": 0.2,               # Solution can be parsed correctly
            "independence": 0.8,        # No adjacent vertices in the set
        }
        
        score = 0.0
        predicted_indset = parse_solution_mis(response)
        
        if predicted_indset is None:
            scores.append(0.0)
            continue
            
        # Solution can be parsed
        score += weights["parse"]
        
        # Convert to set for faster lookups
        indset = set(predicted_indset)
        
        # Check independence: no two vertices should be adjacent
        edges_mis = instance[i]['edges']
        
        # Count violations of independence
        violations = 0
        for u, v in edges_mis:
            if u in indset and v in indset:
                violations += 1
        
        # Calculate independence score
        if violations == 0:
            # Perfectly independent
            score += weights["independence"]
        else:
            # Partial credit based on proportion of violations
            # Calculate maximum possible violations
            total_edges = len(edges_mis)
            max_violations = min(total_edges, len(indset) * (len(indset) - 1) // 2)
            
            # If there are potential violations, calculate a proportion
            if max_violations > 0:
                independence_ratio = max(0, 1 - (violations / max_violations))
                score += weights["independence"] * independence_ratio
            
        scores.append(score)
    
    return scores

def optimality_reward_func_mis(completions, ground_truth, instance, **kwargs):
    """
    Calculate the optimality reward for the Maximum Independent Set (MIS) problem with improved gradient.
    
    The optimality is measured by the size of the independent set compared to the optimal solution.
    """
    scores = []
    feasibility_scores = feasibility_reward_func_mis(completions, instance)
    
    for i, (response, feasibility_score) in enumerate(zip(completions, feasibility_scores)):
        # If solution has very low feasibility score, give minimal optimality reward
        if feasibility_score < 0.9:  # MIS requires high feasibility to be meaningful
            # Give a very small proportional reward to guide learning
            # scores.append(0.1 * feasibility_score)
            scores.append(0.0)
            continue
        
        predicted_indset = parse_solution_mis(response)
        if predicted_indset is None:
            scores.append(0.0)
            continue
        
        try:
            # Size of the predicted independent set
            pred_indset_size = len(set(predicted_indset))
            
            # Parse the reference (optimal) objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue
                
            optimal_indset_size = float(label_obj_match.group(1))
            
            # For MIS, larger is better, so the ratio is the score
            ratio = pred_indset_size / max(1.0, optimal_indset_size)
            
            scores.append(ratio)
            
        except Exception as e:
            # Error in calculating sizes
            scores.append(0.0)
    
    return scores

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
                gap = (real_makespan - optimal_makespan) / max(1.0, optimal_makespan)
                score = 3*1.0 / (1.0 + gap)
                
            scores.append(score)
            
        except Exception as e:
            # Error in calculating makespan
            scores.append(0.0)
    
    return scores

# if __name__ == "__main__":
#     alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.
#     Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.
#
#     ### Instruction:{}
#
#     ### Input:{}
#
#     ### Response:"""
#
#
#     eval_data = [
#     {
#         "num_nodes": "27",
#         "max_route_length": "2710.4",
#         "start_node": "0",
#         "instruction": "Solve the Orienteering Problem with 27 nodes. Each node has (x, y) coordinates and a prize for visiting it. You must plan a route that starts at depot 0, collecting the maximum total prize possible, subject to a maximum route length T = 2710.4. You may visit a subset of nodes, but the total distance traveled must not exceed T.\n\nProvide the solution in the following format:\n1. Route: The ordered list of visited nodes.\n2. Objective: The objective value (summation of the collecting prizes).",
#         "output": "Route: [0, 1, 4, 6, 2, 19, 22, 14, 11, 18, 13, 15, 16, 25, 26, 21, 24, 23, 3, 5, 7], Objective: 107.00",
#         "input": "Node 0, coordinates: [0, 213], prize: 0, neighbors: [7: 45.3, 5: 121.2];Node 1, coordinates: [118, 0], prize: 9, neighbors: [4: 65.4, 6: 209.3];Node 2, coordinates: [193, 267], prize: 5, neighbors: [6: 78.6, 3: 150.2];Node 3, coordinates: [72, 356], prize: 1, neighbors: [5: 59.6, 7: 116.8];Node 4, coordinates: [169, 41], prize: 2, neighbors: [1: 65.4, 6: 169.3];Node 5, coordinates: [17, 333], prize: 10, neighbors: [3: 59.6, 7: 87.0];Node 6, coordinates: [141, 208], prize: 9, neighbors: [2: 78.6, 7: 117.7];Node 7, coordinates: [30, 247], prize: 3, neighbors: [0: 45.3, 5: 87.0];Node 8, coordinates: [765, 999], prize: 4, neighbors: [20: 120.8, 12: 127.6];Node 9, coordinates: [935, 888], prize: 2, neighbors: [17: 77.4, 12: 99.1];Node 10, coordinates: [935, 682], prize: 3, neighbors: [11: 97.5, 18: 99.7];Node 11, coordinates: [848, 638], prize: 8, neighbors: [18: 10.0, 14: 50.6];Node 12, coordinates: [836, 893], prize: 3, neighbors: [16: 89.6, 9: 99.1];Node 13, coordinates: [800, 732], prize: 7, neighbors: [15: 66.0, 14: 78.0];Node 14, coordinates: [800, 654], prize: 1, neighbors: [18: 42.8, 11: 50.6];Node 15, coordinates: [801, 798], prize: 10, neighbors: [16: 57.4, 13: 66.0];Node 16, coordinates: [763, 841], prize: 3, neighbors: [15: 57.4, 12: 89.6];Node 17, coordinates: [1000, 930], prize: 7, neighbors: [9: 77.4, 12: 168.1];Node 18, coordinates: [842, 646], prize: 7, neighbors: [11: 10.0, 14: 42.8];Node 19, coordinates: [639, 483], prize: 3, neighbors: [22: 54.2, 23: 113.6];Node 20, coordinates: [651, 959], prize: 8, neighbors: [8: 120.8, 16: 162.7];Node 21, coordinates: [645, 701], prize: 8, neighbors: [24: 28.4, 26: 49.0];Node 22, coordinates: [660, 533], prize: 1, neighbors: [19: 54.2, 23: 123.0];Node 23, coordinates: [537, 533], prize: 7, neighbors: [19: 113.6, 22: 123.0];Node 24, coordinates: [617, 696], prize: 6, neighbors: [21: 28.4, 26: 66.8];Node 25, coordinates: [706, 756], prize: 2, neighbors: [26: 47.7, 21: 82.1];Node 26, coordinates: [659, 748], prize: 5, neighbors: [25: 47.7, 21: 49.0]."
#     },
#     {
#         "num_nodes": "18",
#         "max_route_length": "2548.3",
#         "start_node": "0",
#         "instruction": "Solve the Orienteering Problem with 18 nodes. Each node has (x, y) coordinates and a prize for visiting it. You must plan a route that starts at depot 0, collecting the maximum total prize possible, subject to a maximum route length T = 2548.3. You may visit a subset of nodes, but the total distance traveled must not exceed T.\n\nProvide the solution in the following format:\n1. Route: The ordered list of visited nodes.\n2. Objective: The objective value (summation of the collecting prizes).",
#         "output": "Route: [0, 2, 11, 16, 9, 6, 7, 5, 17, 10, 13, 14, 1], Objective: 53.00",
#         "input": "Node 0, coordinates: [696, 908], prize: 0, neighbors: [2: 433.4, 14: 458.3];Node 1, coordinates: [850, 307], prize: 5, neighbors: [3: 178.1, 2: 301.4];Node 2, coordinates: [606, 484], prize: 4, neighbors: [11: 205.5, 1: 301.4];Node 3, coordinates: [1000, 403], prize: 4, neighbors: [1: 178.1, 2: 402.2];Node 4, coordinates: [317, 0], prize: 7, neighbors: [12: 229.2, 8: 282.5];Node 5, coordinates: [100, 470], prize: 8, neighbors: [7: 56.8, 17: 107.8];Node 6, coordinates: [167, 382], prize: 4, neighbors: [7: 75.0, 9: 101.4];Node 7, coordinates: [155, 456], prize: 6, neighbors: [5: 56.8, 6: 75.0];Node 8, coordinates: [153, 230], prize: 4, neighbors: [6: 152.6, 15: 166.6];Node 9, coordinates: [267, 399], prize: 3, neighbors: [16: 82.3, 6: 101.4];Node 10, coordinates: [5, 909], prize: 5, neighbors: [13: 137.4, 14: 233.7];Node 11, coordinates: [495, 311], prize: 5, neighbors: [12: 145.5, 2: 205.5];Node 12, coordinates: [474, 167], prize: 1, neighbors: [11: 145.5, 4: 229.2];Node 13, coordinates: [108, 1000], prize: 4, neighbors: [10: 137.4, 14: 169.6];Node 14, coordinates: [238, 891], prize: 4, neighbors: [13: 169.6, 10: 233.7];Node 15, coordinates: [0, 164], prize: 9, neighbors: [8: 166.6, 6: 274.6];Node 16, coordinates: [244, 478], prize: 4, neighbors: [9: 82.3, 7: 91.7];Node 17, coordinates: [170, 552], prize: 6, neighbors: [7: 97.2, 16: 104.7]."
#     },
#     {
#         "num_nodes": "25",
#         "max_route_length": "2587.1",
#         "start_node": "0",
#         "instruction": "Solve the Orienteering Problem with 25 nodes. Each node has (x, y) coordinates and a prize for visiting it. You must plan a route that starts at depot 0, collecting the maximum total prize possible, subject to a maximum route length T = 2587.1. You may visit a subset of nodes, but the total distance traveled must not exceed T.\n\nProvide the solution in the following format:\n1. Route: The ordered list of visited nodes.\n2. Objective: The objective value (summation of the collecting prizes).",
#         "output": "Route: [0, 5, 3, 2, 4, 8, 11, 22, 17, 14, 16, 15, 24, 10, 21, 19, 9], Objective: 99.00",
#         "input": "Node 0, coordinates: [1000, 365], prize: 0, neighbors: [12: 226.6, 19: 293.3];Node 1, coordinates: [171, 740], prize: 6, neighbors: [6: 153.0, 4: 202.5];Node 2, coordinates: [537, 861], prize: 10, neighbors: [7: 153.4, 3: 235.4];Node 3, coordinates: [651, 655], prize: 10, neighbors: [2: 235.4, 5: 238.3];Node 4, coordinates: [368, 693], prize: 10, neighbors: [8: 168.4, 1: 202.5];Node 5, coordinates: [858, 773], prize: 9, neighbors: [3: 238.3, 2: 332.8];Node 6, coordinates: [170, 893], prize: 7, neighbors: [1: 153.0, 4: 281.4];Node 7, coordinates: [602, 1000], prize: 5, neighbors: [2: 153.4, 5: 342.1];Node 8, coordinates: [279, 550], prize: 3, neighbors: [4: 168.4, 1: 218.5];Node 9, coordinates: [693, 404], prize: 10, neighbors: [19: 125.7, 10: 159.1];Node 10, coordinates: [640, 254], prize: 5, neighbors: [21: 23.3, 19: 83.5];Node 11, coordinates: [406, 344], prize: 1, neighbors: [22: 34.2, 17: 36.1];Node 12, coordinates: [806, 248], prize: 1, neighbors: [19: 93.0, 21: 150.4];Node 13, coordinates: [798, 38], prize: 4, neighbors: [23: 108.8, 12: 210.2];Node 14, coordinates: [433, 233], prize: 5, neighbors: [16: 18.1, 17: 87.0];Node 15, coordinates: [517, 204], prize: 10, neighbors: [24: 59.7, 16: 72.9];Node 16, coordinates: [451, 235], prize: 4, neighbors: [14: 18.1, 15: 72.9];Node 17, coordinates: [433, 320], prize: 10, neighbors: [22: 33.5, 11: 36.1];Node 18, coordinates: [210, 269], prize: 2, neighbors: [11: 209.9, 20: 216.6];Node 19, coordinates: [719, 281], prize: 3, neighbors: [21: 76.8, 10: 83.5];Node 20, coordinates: [0, 322], prize: 6, neighbors: [18: 216.6, 8: 360.3];Node 21, coordinates: [656, 237], prize: 5, neighbors: [10: 23.3, 19: 76.8];Node 22, coordinates: [439, 353], prize: 1, neighbors: [17: 33.5, 11: 34.2];Node 23, coordinates: [696, 0], prize: 9, neighbors: [13: 108.8, 21: 240.4];Node 24, coordinates: [526, 263], prize: 3, neighbors: [15: 59.7, 16: 80.1]."
#     }
#     ]
#
#     instances = load_pkl_dataset("data/op/instances.pkl")[0:3]
#
#     def formatting_prompts_func(example):
#         instruction = example["instruction"]
#         input_text = example["input"]
#         text = alpaca_prompt.format(instruction, input_text)
#         return text
#
#     prompts = [formatting_prompts_func(example) for example in eval_data]
#     completions = [example["output"] for example in eval_data]
#
#     rewards_1 = feasibility_reward_func_op(prompts, completions, instances)
#     rewards_2 = optimality_reward_func_op(prompts, completions, [example["output"] for example in eval_data], instances)
#     print(rewards_1)
#     print(rewards_2)
