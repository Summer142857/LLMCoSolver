import json
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os, pickle
from utils import save_dataset
from lkh_solve import get_lkh_solutions

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def transform_route(lst):
    result = []
    current = [0]  # Start each sublist with 0
    for x in lst:
        current.append(x)
        if x == 0:
            result.append(current)
            current = [0]  # Start a new sublist with a 0
    # If there are any leftover numbers in the current group,
    # append a trailing 0 and add it to the result.
    if len(current) > 1:
        current.append(0)
        result.append(current)
    return result

# ----------------------------
#  Distance computations
# ----------------------------

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)

def compute_euclidean_distance_matrix(locations: np.ndarray) -> np.ndarray:
    num_nodes = locations.shape[0]
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return dist_matrix

def calculate_top_k_nearest_nodes(nodes: np.ndarray, k: int = 2) -> list[list[tuple[int, float]]]:
    kdtree = cKDTree(nodes)
    top_k_nearest_nodes = []
    for node in nodes:
        distances, indices = kdtree.query(node, k + 1)  # k+1 to include the node itself
        distances, indices = distances[1:], indices[1:]  # exclude the node itself
        neighbors = [(idx, dist) for idx, dist in zip(indices, distances)]
        top_k_nearest_nodes.append(neighbors)
    return top_k_nearest_nodes


def solve_euclidian_cvrp(depot, loc, demands, vehicle_capacity):
    routes, objective_value = get_lkh_solutions(depot, loc, demands, vehicle_capacity)
    routes = transform_route(routes)
    return objective_value, routes

# ----------------------------
#  Main JSON transformation function
# ----------------------------

def tag_prompt_and_transform_to_json_cvrp(
    instance: torch.Tensor,
    demands: np.ndarray,
    vehicle_capacity: float,
    k_nn: int = 2
) -> dict:
    """
    Create a JSON-ready dictionary describing a CVRP instance:
    - Node coordinates (assume instance[0] is the depot)
    - Node demands (demands[0] = 0 for the depot)
    - Vehicle capacity
    - K nearest neighbors for each node
    - Instruction prompt for the LLM
    - Output: route(s) and objective
    """

    p_size = instance.shape[0]  # number of total locations (depot + customers)

    instruction = (
        f"Solve the Capacitated Vehicle Routing Problem (CVRP) with {p_size - 1} customers "
        "and 1 depot (node 0). Each customer node has a demand. "
        f"All vehicles have the same capacity {vehicle_capacity}. You must assign each customer to exactly one route "
        "and ensure that the sum of demands on each route does not exceed the vehicle capacity. "
        "Minimize the total distance traveled.\n\n"
        f"The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. "
        "Provide the solution in the following format:\n"
        "1. A list of routes, each route as an ordered list of visited nodes (start/end at the depot).\n"
        "2. Objective: The total distance of all routes."
    )

    # Calculate k-nearest neighbors
    node_array = instance.cpu().numpy() if isinstance(instance, torch.Tensor) else instance
    nns = calculate_top_k_nearest_nodes(node_array, k_nn)

    # Build node descriptions
    nodes_description = []
    for i in range(p_size):
        neighbor_str = [f"{n[0]}: {n[1]:.1f}" for n in nns[i]]
        # demands[i], coordinates: instance[i].tolist()
        node_desc = (
            f"Node {i}, coordinates: {instance[i].tolist()}, "
            f"demand: {int(demands[i])}, "
            f"neighbors: {neighbor_str};"
        ).replace("'", "")
        nodes_description.append(node_desc)

    # Attempt to solve the instance
    try:
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        obj, route_list = solve_euclidian_cvrp(
            depot=instance[0],
            loc=node_array[1:],
            demands=demands[1:],
            vehicle_capacity=vehicle_capacity
        )
    except:
        route_list = None

    # Format the solution
    if route_list:
        # e.g. route_list might be [[0, 1, 2, 0], [0, 3, 4, 0]] etc.
        output_str = f"Routes: {route_list}, Objective: {obj:.2f}"
    else:
        return None

    cvrp_json = {
        "num_nodes": str(p_size),
        "vehicle_capacity": f"{vehicle_capacity:.1f}",
        "instruction": instruction,
        "output": output_str,
        "input": "".join(nodes_description),
    }
    # Just a small text fix like your orienteering code:
    cvrp_json['input'] = ".".join(cvrp_json['input'].rsplit(";", 1))

    return cvrp_json

# ----------------------------
#  CVRP Environment Class
# ----------------------------

class CVRPEnv:
    """
    An Environment class to generate random CVRP instances
    with nodes, demands, and a vehicle capacity.
    The first node (index 0) is the depot.
    """

    def __init__(
        self,
        n_node_range: list[int],
        distributions: list[str],
        demand_range: tuple[int, int] = (1, 10),
        seed: int | None = None,
        n_c: int = 3,
        std_cluster: float = 0.07
    ) -> None:
        """
        Parameters
        ----------
        n_node_range : list[int]
            [min_nodes, max_nodes] range for random instance size (excluding the depot).
            So total points = 1 depot + [min_nodes..max_nodes] customers.
        distributions : list[str]
            A list of distribution names to sample from. 
            E.g. ['uniform', 'gaussian_mixture_2_5', 'clustered', 'mixed'].
        demand_range : tuple[int, int]
            (low, high) range for random demand generation.
        seed : int | None
            Random seed for reproducibility.
        n_c : int
            Number of cluster centers for clustered and mixed distributions.
        std_cluster : float
            Standard deviation for normal distribution of city clusters.
        """
        self.n_node_range = n_node_range
        self.distributions = distributions
        self.demand_range = demand_range
        self.seed = seed
        self.n_c = n_c
        self.std_cluster = std_cluster

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate_clustered_nodes(self, n_nodes: int, mixed: bool = False, max_xy: float = 1.0) -> np.ndarray:
        """
        Generate node coordinates using clustered or mixed distribution.
        
        Parameters
        ----------
        n_nodes : int
            Total number of nodes to generate.
        mixed : bool
            If True, use mixed distribution (uniform + clustered). If False, use pure clustered.
        max_xy : float
            Maximum coordinate value for the space.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_nodes, 2) with node coordinates.
        """
        uniform_frac = 0.5 if mixed else 0.0
        n_uniform = int(n_nodes * uniform_frac)
        n_clustered = n_nodes - n_uniform
        
        # Generate uniform nodes for mixed distribution
        uniform_locs = np.random.uniform(0, max_xy, size=(n_uniform, 2)) if n_uniform > 0 else np.empty((0, 2))
        
        # Generate cluster centers
        assert self.n_c < n_nodes, f"Number of clusters ({self.n_c}) must be less than number of nodes ({n_nodes})"
        centers = np.random.uniform(0.2, max_xy - 0.2, size=(self.n_c, 2))
        
        # Generate clustered nodes around centers
        n_clustered_samples = 0
        all_clustered_locs = []
        while n_clustered_samples < n_clustered:
            # Sample random centers for each point
            center_locs = centers[np.random.randint(len(centers), size=2 * (n_clustered - n_clustered_samples))]
            # Generate points around centers using normal distribution
            cluster_locs = np.random.normal(center_locs, self.std_cluster)
            # Keep only points within bounds
            cluster_locs = cluster_locs[(cluster_locs >= 0).all(axis=1) & (cluster_locs < max_xy).all(axis=1)]
            all_clustered_locs.append(cluster_locs)
            n_clustered_samples += len(cluster_locs)
        
        # Combine all clustered locations and trim to exact number needed
        cluster_locs = np.concatenate(all_clustered_locs)[:n_clustered] if all_clustered_locs else np.empty((0, 2))
        
        # Combine uniform and clustered locations
        if n_uniform > 0 and n_clustered > 0:
            xys = np.vstack((uniform_locs, cluster_locs))
        elif n_uniform > 0:
            xys = uniform_locs
        else:
            xys = cluster_locs
            
        return xys

    def generate_gaussian_mixture_points(
        self,
        dataset_size: int,
        graph_size: int,
        num_modes: int = 0,
        cdist_: float = 0
    ) -> np.ndarray:
        """
        Generate node coordinates with a Gaussian mixture distribution or uniform distribution.
        """

        def single_gaussian_mixture(graph_size=100, modes=0, cdist_val=1):
            nums = np.random.multinomial(graph_size, np.ones(modes) / modes)
            xy_list = []
            for num in nums:
                center = np.random.uniform(0, cdist_val, size=(1, 2))
                points = np.random.multivariate_normal(
                    mean=center.squeeze(),
                    cov=np.eye(2, 2),
                    size=(num,)
                )
                xy_list.extend(points)
            xy_arr = np.array(xy_list)
            xy_arr = MinMaxScaler().fit_transform(xy_arr)
            return xy_arr

        if num_modes == 0:
            # (0, 0) => uniform
            return np.random.uniform(0, 1, [dataset_size, graph_size, 2])
        else:
            result = []
            for _ in range(dataset_size):
                result.append(
                    single_gaussian_mixture(
                        graph_size=graph_size,
                        modes=num_modes,
                        cdist_val=cdist_
                    )
                )
            return np.array(result)

    def uniform_random_choice(self, start: int, end: int) -> int:
        return random.randint(start, end)

    def generate_instances(self, n_instance: int) -> list[tuple[torch.Tensor, np.ndarray, float]]:
        """
        Generate a list of random CVRP instances:
        - (coords_tensor, demands_array, vehicle_capacity).

        coords_tensor shape: [N+1, 2], with index 0 as the depot
        demands_array shape: [N+1], demands[0] = 0
        vehicle_capacity is a single float or int
        """
        from_code_to_modes = {
            'uniform': (0, 0),
            'gaussian_mixture_2_5': (2, 5),
            'gaussian_mixture_3_10': (3, 10)
        }

        low_demand, high_demand = self.demand_range
        instances = []

        for _ in range(n_instance):
            # 1) Randomly choose number of customers
            size_i = self.uniform_random_choice(self.n_node_range[0], self.n_node_range[1])
            total_points = size_i + 1  # +1 for depot
            # 2) Randomly choose distribution
            distribution_i = random.choice(self.distributions)
            
            # 3) Generate coords for all points (including depot) based on distribution type
            if distribution_i == 'clustered':
                coords = self.generate_clustered_nodes(total_points, mixed=False, max_xy=1.0)
            elif distribution_i == 'mixed':
                coords = self.generate_clustered_nodes(total_points, mixed=True, max_xy=1.0)
            elif distribution_i in from_code_to_modes:
                num_modes, cdist_ = from_code_to_modes[distribution_i]
                coords = self.generate_gaussian_mixture_points(1, total_points, num_modes, cdist_)[0]
            else:
                raise NotImplementedError(f"Distribution '{distribution_i}' is not defined.")

            # Scale by 1000 and convert to tensor
            coords = coords * 1000
            coords_tensor = torch.tensor(coords).int()

            # 4) Generate demands, with demands[0] = 0 for the depot
            demands = np.random.randint(low=low_demand, high=high_demand + 1, size=total_points)
            demands[0] = 0  # depot

            # 5) Generate a random vehicle capacity
            # Huang et al. 2025: Rethinking Light Decoder-based Solvers for VRPs
            r = np.random.triangular(3, 6, 25)
            avg_demand = np.mean(demands[1:])
            vehicle_capacity = int(r*avg_demand)

            instances.append((coords_tensor, demands, vehicle_capacity))

        return instances

    def generate_instances_and_save(
            self,
            n_instance: int,
            file_name: str,
            save_pkl: bool,
            rl_data: bool = False,
            k_nn: int = 2
    ) -> None:
        """
        Generate CVRP instances and save them in both .pkl and .json formats.
        We use a while loop to ensure that any instance that fails or leads to a solver error
        doesn't cause mismatch between .pkl and .json.

        Parameters
        ----------
        n_instance : int
            Number of valid instances to generate.
        file_name : str
            Output JSON file path.
        save_pkl : bool
            Whether to save instances as a pickle file.
        rl_data : bool
            Whether to include the instance data in the JSON file.
        k_nn : int
            Number of nearest neighbors to include in the instance description.
        """

        valid_json_data = []
        valid_instances_pkl = []

        count_valid = 0
        pbar = tqdm(total=n_instance, desc="Generating CVRP instances")

        while count_valid < n_instance:
            coords_tensor, demands, capacity = self.generate_instances(1)[0]
            try:
                cvrp_json = tag_prompt_and_transform_to_json_cvrp(
                    coords_tensor,
                    demands,
                    vehicle_capacity=capacity,
                    k_nn=k_nn
                )
                if cvrp_json is not None:
                    valid_json_data.append(cvrp_json)
                    if save_pkl:
                        valid_instances_pkl.append((coords_tensor, demands, capacity))
                    if rl_data:
                        cvrp_json['instance'] = [coords_tensor.tolist(), demands.tolist(), capacity]
                    count_valid += 1
                    pbar.update(1)
            except Exception:
                pass

        pbar.close()

        if save_pkl:
            save_dataset(valid_instances_pkl, "./instance_mixed.pkl")

        with open(file_name, 'w') as f:
            print(f"Generated {len(valid_json_data)} valid CVRP instances.")
            json.dump(valid_json_data, f, indent=4)
            

    def read_and_transform_pkl(self, pkl_file: str, output_file: str, rl_data: bool = False, k_nn: int = 2) -> None:
        """
        Read CVRP instances from a pickle file and transform them to textual format.

        Parameters
        ----------
        pkl_file : str
            Path to the pickle file containing CVRP instances.
        output_file : str
            Path where the resulting JSON file should be saved.
        rl_data : bool
            Whether to include the original instance data in the output.
        k_nn : int
            Number of nearest neighbors to include in the instance description.
        """
        # Read the pickle file
        with open(pkl_file, 'rb') as f:
            instances = pickle.load(f)
        
        # Transform instances to text format
        cvrp_data = []
        for instance in tqdm(instances, desc="Transforming CVRP instances"):
            depot, coords_tensor, demands, capacity = instance
            coords_tensor = np.array([depot] + coords_tensor) * 1000
            coords_tensor = coords_tensor.astype(int)
            demands = np.array([0] + demands)
            try:
                json_data = tag_prompt_and_transform_to_json_cvrp(
                    coords_tensor,
                    demands,
                    vehicle_capacity=int(capacity),
                    k_nn=k_nn
                )
                if json_data is not None:
                    if rl_data:
                        json_data["instance"] = [coords_tensor.tolist(), demands.tolist(), capacity]
                    cvrp_data.append(json_data)
            except Exception as e:
                print(f"Error processing instance: {e}")
                continue

        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(cvrp_data, f, indent=4)
        print(f">> Saved {len(cvrp_data)} transformed instances to {output_file}")



# ----------------------------
#  Example usage
# ----------------------------
if __name__ == "__main__":
    # Example usage with all distribution types including clustered and mixed
    cvrp_env = CVRPEnv(
        n_node_range=[10, 100],
        distributions=['mixed'],
        demand_range=(1, 10),
        seed=42,
        n_c=7,  # Number of cluster centers for clustered/mixed distributions
        std_cluster=0.1  # Standard deviation for cluster spread
    )

    # Generate 200 example instances, store as cvrp_train.json
    cvrp_env.generate_instances_and_save(
        n_instance=100,
        file_name='test_mixed.json',
        save_pkl=True,  # If True, also save a separate .pkl with raw data
        rl_data=True,
        k_nn=2           # 2 nearest neighbors in the instance description
    )

    # from utils import concat_json_files
    # concat_json_files('train_new.json', 'cvrp_1.json', 'train.json')