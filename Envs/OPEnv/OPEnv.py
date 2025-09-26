import json
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os, pickle
# N
from pyCombinatorial.algorithm import nearest_neighbour
from pyCombinatorial.utils import graphs, util
from compass_solve import solve_compass

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))


def solve_euclidian_op(depot, loc, prize, max_length):
    objective_value, cycle = solve_compass("./op-solver/build/src/op-solver", depot, loc, prize, max_length)
    return objective_value, cycle



def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(point1 - point2)


def compute_euclidean_distance_matrix(locations: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for a set of points.
    """
    num_nodes = locations.shape[0]
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return dist_matrix


def calculate_top_k_nearest_nodes(nodes: np.ndarray, k: int = 2) -> list[list[tuple[int, float]]]:
    """
    For each node, calculate its top k nearest neighbors using a k-d tree.

    Returns
    -------
    list of lists, where the ith entry is a list of (neighbor_index, distance)
    for the k nearest neighbors of node i.
    """
    kdtree = cKDTree(nodes)
    top_k_nearest_nodes = []
    for node in nodes:
        distances, indices = kdtree.query(node, k + 1)  # k+1 to include the node itself
        # Exclude the node itself (first index)
        distances, indices = distances[1:], indices[1:]
        neighbors = [(idx, dist) for idx, dist in zip(indices, distances)]
        top_k_nearest_nodes.append(neighbors)
    return top_k_nearest_nodes


def generate_random_prizes(num_nodes: int, low: int = 1, high: int = 100) -> np.ndarray:
    """
    Generate an array of random prizes (integers) for each node.
    """
    return np.random.randint(low=low, high=high + 1, size=num_nodes)


def tag_prompt_and_transform_to_json_orienteering(
    instance: torch.Tensor,
    prizes: np.ndarray,
    max_route_length: float,
    k_nn: int = 2,
    start_node: int = 0
) -> dict:
    """
    Create a JSON-ready dictionary describing an Orienteering instance:
    - Node coordinates
    - Node prizes
    - Maximum route length T
    - (Optionally) a designated start and end node
    - K nearest neighbors for each node

    Parameters
    ----------
    instance : torch.Tensor
        Coordinates of the nodes. Shape: (N, 2).
    prizes : np.ndarray
        1D array of length N with the prize for each node.
    max_route_length : float
        The maximum route length T for the orienteering constraint.
    k_nn : int
        Number of nearest neighbors to list for each node. Default is 2.
    start_node : int
        The designated start node index (often 0).

    Returns
    -------
    dict
        Dictionary containing an orienteering instance description.
    """
    p_size = instance.shape[0]

    instruction = (
        f"Solve the Orienteering Problem with {p_size} nodes. "
        "Each node has (x, y) coordinates and a prize for visiting it. "
        f"You must plan a route that starts at depot {start_node}, "
        f"collecting the maximum total prize possible, subject to a maximum route length T = {max_route_length:.1f}. "
        "You may visit a subset of nodes, but the total distance traveled must not exceed T.\n\n"
        f"The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. "
        "Provide the solution in the following format:\n"
        "1. Route: The ordered list of visited nodes.\n"
        "2. Objective: The objective value (summation of the collecting prizes)."
    )

    # Calculate nearest neighbors
    node_array = instance.cpu().numpy() if isinstance(instance, torch.Tensor) else instance
    nns = calculate_top_k_nearest_nodes(node_array, k_nn)

    # Build node descriptions
    nodes_description = []
    for i in range(p_size):
        neighbor_str = [f"{n[0]}: {n[1]:.1f}" for n in nns[i]]
        node_desc = (
            f"Node {i}, coordinates: {instance[i].tolist()}, "
            f"prize: {int(prizes[i])}, "
            f"neighbors: {neighbor_str};"
        ).replace("\'", "")
        nodes_description.append(node_desc)

    # ----------------------------
    # Solve the instance with compass
    # ----------------------------
    try:
        obj, route = solve_euclidian_op(
            depot=instance[start_node].cpu().numpy(),
            loc=instance.cpu().numpy(),
            prize=prizes,
            max_length=max_route_length
        )
    except:
        route = None

    # Format the solution output
    if route:
        output_str = (
            f"Route: {route}, "
            f"Objective: {obj:.2f}"
        )
    else:
        # No feasible solution (should be rare if T is large enough)
        # output_str = "No feasible solution found under the given constraints."
        return None

    # Final JSON structure
    orienteering_json = {
        "num_nodes": str(p_size),
        "max_route_length": f"{max_route_length:.1f}",
        "start_node": str(start_node),
        "instruction": instruction,
        "output": output_str,
        "input": "".join(nodes_description),
    }

    orienteering_json['input'] = ".".join(orienteering_json['input'].rsplit(";", 1))

    return orienteering_json


class OrienteeringEnv:
    """
    An Environment class to generate random orienteering instances
    with nodes, prizes, and a maximum route length T.
    """

    def __init__(
        self,
        n_node_range: list[int],
        distributions: list[str],
        prize_range: tuple[int, int] = (1, 100),
        seed: int | None = None,
        n_c: int = 3,
        std_cluster: float = 0.07
    ) -> None:
        """
        Parameters
        ----------
        n_node_range : list[int]
            [min_nodes, max_nodes] range for random instance size.
        distributions : list[str]
            A list of distribution names to sample from.
            E.g. ['uniform', 'gaussian_mixture_2_5', 'clustered', 'mixed'].
        prize_range : tuple[int, int]
            (low, high) range for random prize generation.
        seed : int | None
            Random seed for reproducibility.
        n_c : int
            Number of cluster centers for clustered and mixed distributions.
        std_cluster : float
            Standard deviation for normal distribution of city clusters.
        """
        self.n_node_range = n_node_range
        self.distributions = distributions
        self.prize_range = prize_range
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
            # from sklearn.preprocessing import MinMaxScaler
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

    def weighted_random_choice(self, start: int, end: int) -> int:
        """
        Choose a random number from start to end (inclusive) with weighted probability.
        """
        numbers = list(range(start, end + 1))
        weights = [i for i in range(start, end + 1)]
        return random.choices(numbers, weights=weights, k=1)[0]

    def uniform_random_choice(self, start: int, end: int) -> int:
        """
        Choose a random number from start to end (inclusive) with uniform probability.
        """
        # Option 1: Using random.randint which is efficient and concise.
        return random.randint(start, end)

    def generate_instances(self, n_instance: int) -> list[tuple[torch.Tensor, np.ndarray, float]]:
        """
        Generate a list of random orienteering instances: (coords_tensor, prize_array, T).

        Returns
        -------
        list of tuples (coords_tensor, prizes, max_route_length).
        """
        from_code_to_modes = {
            'uniform': (0, 0),
            'gaussian_mixture_2_5': (2, 5),
            'gaussian_mixture_3_10': (3, 10)
        }

        low_prize, high_prize = self.prize_range

        instances = []
        for _ in range(n_instance):
            # 1) Randomly choose number of nodes
            size_i = self.uniform_random_choice(self.n_node_range[0], self.n_node_range[1])
            # 2) Randomly choose distribution
            distribution_i = random.choice(self.distributions)
            
            # 3) Generate coords based on distribution type
            if distribution_i == 'clustered':
                coords = self.generate_clustered_nodes(size_i, mixed=False, max_xy=1.0)
            elif distribution_i == 'mixed':
                coords = self.generate_clustered_nodes(size_i, mixed=True, max_xy=1.0)
            elif distribution_i in from_code_to_modes:
                num_modes, cdist_ = from_code_to_modes[distribution_i]
                coords = self.generate_gaussian_mixture_points(1, size_i, num_modes, cdist_)[0]
            else:
                raise NotImplementedError(f"Distribution '{distribution_i}' is not defined.")

            # Scale by 1000 and convert to tensor
            coords = coords * 1000
            coords_tensor = torch.tensor(coords).int()
            
            # 4) Generate prizes
            prizes = generate_random_prizes(size_i, low=low_prize, high=high_prize)
            prizes[0] = 0  # Depot prize is 0
            
            # 5) Generate random T based on approximate TSP distance
            distance_matrix = util.build_distance_matrix(coords)
            parameters = {
                'initial_location': 0,  # -1 =  Try All Locations.
                'local_search': False,
                'verbose': False
            }
            _, distance_tsp = nearest_neighbour(distance_matrix, **parameters)

            route_limit_T = np.random.uniform(0.5 * distance_tsp, 0.7 * distance_tsp)

            instances.append((coords_tensor, prizes, route_limit_T))

        return instances

    def generate_instances_and_save(
            self,
            n_instance: int,
            file_name: str,
            save_pkl: bool,
            rl_data: bool = False,
            k_nn: int = 2,
            start_node: int = 0,
    ) -> None:
        """
        Generate orienteering instances and save them in both .pkl and .json formats.
        We use a while loop to ensure that failed processing doesn't cause a mismatch
        between the number of .pkl and .json records.

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
        start_node : int
            Designated start node index.
        """

        valid_json_data = []
        valid_instances_pkl = []

        count_valid = 0

        pbar = tqdm(total=n_instance, desc="Generating Orienteering instances")

        # Keep generating until we reach n_instance valid instances
        while count_valid < n_instance:
            # Generate a single instance (adjust to your actual generate method)
            # If your `generate_instances(1)` returns a list of length 1, we can grab [0].
            coords_tensor, prizes, T_val = self.generate_instances(1)[0]

            try:
                # Process the instance
                op_json = tag_prompt_and_transform_to_json_orienteering(
                    coords_tensor,
                    prizes,
                    max_route_length=T_val,
                    k_nn=k_nn,
                    start_node=start_node
                )

                # If processing succeeds and returns a valid JSON, store both in memory
                if rl_data:
                    if op_json is not None:
                        op_json["instance"] = [coords_tensor.tolist(), prizes.tolist(), T_val]
                        valid_json_data.append(op_json)
                        valid_instances_pkl.append((coords_tensor, prizes, T_val))
                        count_valid += 1
                        pbar.update(1)
            except Exception as e:
                pass

        pbar.close()

        if save_pkl:
            # Save the valid instances as a pickle file
            save_dataset(valid_instances_pkl, "./tmp.pkl")

        with open(file_name, 'w') as f:
            print(f"Generated {len(valid_json_data)} valid Orienteering instances.")
            json.dump(valid_json_data, f, indent=4)


if __name__ == "__main__":
    # Example usage with all distribution types including clustered and mixed
    orienteering_env = OrienteeringEnv(
        n_node_range=[10, 100],
        distributions=['uniform', 'gaussian_mixture_2_5', 'gaussian_mixture_3_10'],
        prize_range=(1, 10),
        seed=42,
        n_c=7,  # Number of cluster centers for clustered/mixed distributions
        std_cluster=0.1  # Standard deviation for cluster spread
    )

    orienteering_env.generate_instances_and_save(
        n_instance=100,
        file_name='tmp.json',
        save_pkl=True,
        rl_data=True,
        k_nn=2,        # 2 nearest neighbors for demonstration
        start_node=0,  # Start at node 0
    )

    # from utils import concat_json_files
    # concat_json_files('train_new.json', 'train_op1.json', 'train_op.json')

    # import json
    # input_file = "E:\\program\\fm4co\\Envs\\MISEnv\\test.json"
    #
    # output_file = "E:\\program\\fm4co\\Envs\\MISEnv\\eval.json"
    #
    # with open(input_file, "r", encoding="utf-8") as f_in:
    #     data = json.load(f_in)
    #
    # # 'data' should be a list of dictionaries.
    # # Remove 'instance' if it exists in each record.
    # for item in data:
    #     if "instance" in item:
    #         del item["instance"]
    #
    # # Write the modified data to a new JSON file.
    # with open(output_file, "w", encoding="utf-8") as f_out:
    #     json.dump(data, f_out, ensure_ascii=False, indent=2)