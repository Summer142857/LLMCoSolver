import os
from pyCombinatorial.algorithm import farthest_insertion, nearest_neighbour, nearest_insertion
import pickle
from tqdm import tqdm
import numpy as np
import time
from copy import copy
from scipy.spatial import distance_matrix
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def select_next_node_AEL(current_node, destination_node, unvisited_nodes, distance_matrix, threshold=0.7):
    """Algorithm Evolution Using Large Language Model"""
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        score = 0.4 * distance_matrix[current_node][node] - 0.3 * average_distance_to_unvisited + 0.2 * std_dev_distance_to_unvisited - 0.1 * distance_matrix[destination_node][node]
        scores[node] = score
    if min(scores.values()) > threshold:
        next_node = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
    else:
        next_node = min(scores, key=scores.get)
    return next_node

def select_next_node_ReEvo(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit from the unvisited nodes."""
    weights = {'distance_to_current': 0.4,
               'average_distance_to_unvisited': 0.25,
               'std_dev_distance_to_unvisited': 0.25,
               'distance_to_destination': 0.1}
    scores = {}
    for node in unvisited_nodes:
        future_distances = [distance_matrix[node, i] for i in unvisited_nodes if i != node]
        if future_distances:
            average_distance_to_unvisited = sum(future_distances) / len(future_distances)
            std_dev_distance_to_unvisited = (sum((x - average_distance_to_unvisited) ** 2 for x in future_distances) / len(future_distances)) ** 0.5
        else:
            average_distance_to_unvisited = std_dev_distance_to_unvisited = 0
        score = (weights['distance_to_current'] * distance_matrix[current_node, node] -
                 weights['average_distance_to_unvisited'] * average_distance_to_unvisited +
                 weights['std_dev_distance_to_unvisited'] * std_dev_distance_to_unvisited -
                 weights['distance_to_destination'] * distance_matrix[destination_node, node])
        scores[node] = score
    next_node = min(scores, key=scores.get)
    return next_node

def select_next_node_mcts_ahd(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return None

    next_node = None
    min_score = float('inf')
    total_unvisited_distance = sum(distance_matrix[current_node][node] for node in unvisited_nodes)

    # Improved decay factor inspired by No.1 algorithm
    decay_factor = 0.5 - (0.1 / max(1, len(unvisited_nodes)))

    for node in unvisited_nodes:
        local_distance = distance_matrix[current_node][node]
        global_contribution = total_unvisited_distance / (1 + sum(distance_matrix[node][j] for j in unvisited_nodes))

        # Score calculation emphasizing local distance with the new decay factor
        score = 0.6 * local_distance + 0.4 * decay_factor * global_contribution

        # Selecting the node with the minimum score
        if score < min_score:
            min_score = score
            next_node = node

    return next_node

def eval_heuristic(node_positions: np.ndarray, method='AEL') -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.

    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    method : str
        The method to use for next node selection ('AEL', 'ReEvo', or 'mcts').

    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        if method == 'AEL':
            next_node = select_next_node_AEL(
                current_node=solution[-1],
                destination_node=start_node,
                unvisited_nodes=copy(unvisited),
                distance_matrix=dist_mat.copy(),
            )
        elif method == 'ReEvo':
            next_node = select_next_node_ReEvo(
                current_node=solution[-1],
                destination_node=start_node,
                unvisited_nodes=copy(unvisited),
                distance_matrix=dist_mat.copy(),
            )
        elif method == 'mcts':
            next_node = select_next_node_mcts_ahd(
                current_node=solution[-1],
                destination_node=start_node,
                unvisited_nodes=copy(unvisited),
                distance_matrix=dist_mat.copy(),
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")

    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj

def compute_euclidean_distance_matrix(locations):
    num_nodes = locations.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    # Calculate the Euclidean distance between each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return distance_matrix

def solve_per_instance(problem, method="AEL"):
    coords = problem
    
    # Scale data for MCTS method
    # if method == 'mcts':
    #     # Convert tensor to numpy if needed
    #     if torch.is_tensor(coords):
    #         data = coords.cpu().numpy()
    #     else:
    #         data = np.array(coords)
            
    #     # Apply scaling for MCTS
    #     scale = max(np.max(data, axis=0) - np.min(data, axis=0))
    #     data = (data - np.min(data, axis=0)) / scale
        
    #     # Convert back to tensor if the original was a tensor
    #     if torch.is_tensor(coords):
    #         coords = torch.tensor(data, dtype=coords.dtype, device=coords.device)
    #     else:
    #         coords = data
    
    distances = compute_euclidean_distance_matrix(coords)
    if method == "farthest_insertion":
        _, cost = farthest_insertion(distances, initial_location=1, verbose=False)
    elif method == "nearest_neighbour":
        _, cost = nearest_neighbour(distances, initial_location=1, local_search=False, verbose=False)
    elif method == 'ReEvo':
        cost = eval_heuristic(coords.cpu().numpy(), method='ReEvo')
    elif method == 'AEL':
        cost = eval_heuristic(coords.cpu().numpy(), method='AEL')
    elif method == 'mcts':
        cost = eval_heuristic(coords.cpu().numpy(), method='mcts')
    else:
        assert 0, 'The heuristic is not supported!'
    return cost

def solve_all_heuristic(method="AEL"):
    results = []
    problems = load_pkl_dataset("data_benchmark/tsp_transformed/tsplib_all.pkl")
    for i, instance in tqdm(enumerate(problems)):
        result = solve_per_instance(instance, method=method)
        results.append(result)
    return results

if __name__ == "__main__":
    # Optimal solutions provided for each instance
    optimal_solutions = [7542, 426, 538, 108159, 1211, 675]
    
    # Choose the method to evaluate: "farthest_insertion", "nearest_neighbour", "ReEvo", "AEL", or "mcts"
    method = "AEL"
    
    start_time = time.time()
    heuristic_costs = solve_all_heuristic(method=method)
    solve_time = time.time() - start_time
    
    # Calculate gaps
    gaps = []
    for i, (heuristic, optimal) in enumerate(zip(heuristic_costs, optimal_solutions)):
        gap = (heuristic - optimal) / optimal * 100
        gaps.append(gap)
        print(f"Instance {i}: Heuristic ({method}) = {heuristic:.2f}, Optimal = {optimal}, Gap = {gap:.2f}%")
    
    print("\nSummary:")
    print(f"Method: {method}")
    print(f">> Solving within {solve_time:.2f}s using heuristics")
    print(f"Average cost: {np.mean(heuristic_costs):.2f} ± {2 * np.std(heuristic_costs) / np.sqrt(len(heuristic_costs)):.2f}")
    print(f"Average gap: {np.mean(gaps):.2f}% ± {2 * np.std(gaps) / np.sqrt(len(gaps)):.2f}%")
    print(f"Min gap: {np.min(gaps):.2f}%, Max gap: {np.max(gaps):.2f}%")

    # NN: 1.09h