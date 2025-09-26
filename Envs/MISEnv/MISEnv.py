import random
import numpy as np
import torch
import pickle
import json
import os
from tqdm import tqdm

# For the solver
from gurobipy import Model, GRB
import networkx as nx

###############################################################################
# Utility Functions
###############################################################################

def check_extension(filename):
    """
    Ensure the filename ends with '.pkl'.
    """
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename, disable_print=False):
    """
    Save 'dataset' to a pickle file named 'filename'.
    """
    filedir = os.path.split(filename)[0]
    if filedir != "" and not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(f">> Save dataset to {filename}")

###############################################################################
# Gurobi MIS Solver
###############################################################################

def solve_mis_gurobi(vertices, edges):
    """
    Solve the Maximum Independent Set exactly using Gurobi.

    Parameters
    ----------
    vertices : list[int]
        List of vertex IDs (e.g. [0, 1, 2, ...]).
    edges : list[tuple[int, int]]
        List of edges (u, v).

    Returns
    -------
    independent_set : list[int]
        The set of vertices in an optimal maximum independent set (sorted).
    """
    m = Model("MIS")
    m.setParam("OutputFlag", 0)  # to silence Gurobi logs if desired

    # 1) Create binary decision variable x[v] for each vertex
    x = {}
    for v in vertices:
        # Objective is to maximize sum of x[v]
        x[v] = m.addVar(vtype=GRB.BINARY, obj=1.0, name=f"x_{v}")

    m.update()

    # 2) Add constraints: x[u] + x[v] <= 1 for every edge (u, v)
    #    (cannot pick both endpoints of any edge in an independent set)
    for (u, v) in edges:
        m.addConstr(x[u] + x[v] <= 1, name=f"e_{u}_{v}")

    m.update()

    # 3) Solve (maximize total x[v])
    m.modelSense = GRB.MAXIMIZE
    m.optimize()

    # 4) Collect vertices in the independent set
    independent_set = []
    for v in vertices:
        if x[v].X > 0.5:
            independent_set.append(v)
    independent_set.sort()

    return independent_set

def compute_top_k_neighbors_by_degree(num_nodes, edges, k=3):
    """
    Compute for each node up to k neighbors sorted by descending neighbor-degree.

    Returns
    -------
    top_k_for_each: dict[int, list[tuple[int,int]]]
        A dictionary keyed by node i, whose value is a list of (neighbor, neighbor_degree).
    """
    # 1) Build adjacency list
    adjacency_list = {i: [] for i in range(num_nodes)}
    for (u, v) in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # 2) Compute degrees
    degrees = [len(adjacency_list[i]) for i in range(num_nodes)]

    # 3) For each node, sort neighbors by descending degree and take top-k
    top_k_for_each = {}
    for i in range(num_nodes):
        # List of (nbr, degree_of_nbr)
        neighbor_degs = [(nbr, degrees[nbr]) for nbr in adjacency_list[i]]
        neighbor_degs.sort(key=lambda x: x[1], reverse=True)
        top_k_for_each[i] = neighbor_degs[:k]
    return top_k_for_each

###############################################################################
# Tagging Function for JSON (MIS)
###############################################################################

def tag_prompt_and_transform_to_json_mis_with_neighbors(num_nodes, edge_list, independ_set, k_neighbors=2):
    """
    Create a JSON-like dict describing:
      - The instruction to solve MIS
      - The 'input' description, which includes top-k adjacency info in format N0:[4,2,#7,#4]
      - The 'output' (independent set and objective)

    We embed a small "heuristic" clue by listing each node's top k neighbors
    sorted by that neighbor's degree.
    """
    # Compute top-k neighbor info
    top_k_info = compute_top_k_neighbors_by_degree(num_nodes, edge_list, k_neighbors)

    instruction = (
        f"Given an undirected graph with {num_nodes} nodes (0..{num_nodes - 1}) "
        f"and edges specified below. For each node, we also provide up to {k_neighbors} neighbors connected to it. "
        "Find a maximum independent set: the largest set of vertices "
        "where no two vertices share an edge.\n\n"
        f"The input includes the edges of the graph and the top-{k_neighbors} neighbors for each node in format N[a,b,#c,#d], "
        f"where a and b are the top-{k_neighbors} neighbors, "
        "#c is degree of a, and #d is degree of b. Format:\n"
        "1. Set: The list of vertices in the maximum independent set.\n"
        "2. Objective: The size of that set."
    )

    output_str = (
        f"Set: {independ_set}, "
        f"Objective: {len(independ_set)}"
    )

    # Remove spaces from edge list
    compact_edge_list = str(edge_list).replace(' ', '')

    # Build neighbor information in N0:[4,2,#7,#4] format
    neighbor_entries = []
    for node in range(num_nodes):
        neighbor_data = top_k_info[node]
        if k_neighbors == 2 and len(neighbor_data) == 2:
            # Format for k=2: N0:[4,2,#7,#4]
            n1, deg1 = neighbor_data[0]
            n2, deg2 = neighbor_data[1]
            neighbor_entries.append(f"N{node}:[{n1},{n2},#{deg1},#{deg2}]")
        else:
            # Handle different k values or incomplete neighbor data
            neighbor_pairs = []
            for n, deg in neighbor_data:
                neighbor_pairs.extend([n, f"#{deg}"])
            neighbor_entries.append(f"N{node}:[{','.join(map(str, neighbor_pairs))}]")

    # Combine edges + neighbor info
    input_str = (
        f"Edges: {compact_edge_list}\n"
        f"{';'.join(neighbor_entries)}"
    )

    mis_json = {
        "num_nodes": str(num_nodes),
        "instruction": instruction,
        "output": output_str,
        "input": input_str
    }
    return mis_json
###############################################################################
# Environment Class for Generating Random MIS Instances
# with random selection of either Erdős–Rényi or Barabási–Albert
###############################################################################

class MISEnv:
    """
    Environment class to generate random graph instances for the Maximum Independent Set (MIS) problem.
    Randomly picks between:
      - an Erdős–Rényi model, or
      - a Barabási–Albert model
    for each instance.
    """
    def __init__(self,
                 n_node_range: list[int],
                 distributions: list[str],
                 p_edge_range: tuple[float, float] = (0.1, 0.5),
                 attach_links_range: list[int] = [1, 3],
                 seed: int | None = None):
        """
        Parameters
        ----------
        n_node_range : list[int]
            [min_nodes, max_nodes] range for random instance size.
        distributions : list[str]
            A list that may include "er" or "ba" to denote which distributions
            the environment randomly picks from for each instance.
            e.g. ["er", "ba"] to use both.
        p_edge_range : tuple[float, float]
            For Erdős–Rényi: (low, high) range for random edge probability.
        attach_links_range : list[int]
            For Barabási–Albert: [min_m, max_m] for 'm'.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_node_range = n_node_range
        self.distributions = distributions
        self.p_edge_range = p_edge_range
        self.attach_links_range = attach_links_range
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def uniform_random_choice(self, start: int, end: int) -> int:
        """
        Choose a random integer from [start, end] uniformly.
        """
        return random.randint(start, end)

    def generate_erdos_renyi_graph(self, num_nodes: int, p_edge: float):
        """
        Generate an Erdős–Rényi random graph with 'num_nodes' and edge probability 'p_edge'.
        Returns a list of edges.
        """
        edges = []
        for u in range(num_nodes):
            for v in range(u+1, num_nodes):
                if random.random() < p_edge:
                    edges.append((u, v))
        return edges

    def generate_barabasi_albert_graph(self, num_nodes: int, attach_links: int):
        """
        Generate a Barabási–Albert graph with 'num_nodes' and 'attach_links' edges
        for each new node.
        Returns a list of edges.
        """
        # If the number of nodes is less than or equal to attach_links,
        # degrade to smaller attach_links
        if num_nodes <= attach_links:
            attach_links = max(1, num_nodes - 1)

        G = nx.barabasi_albert_graph(num_nodes, attach_links)
        edges = list(G.edges())
        return edges

    def generate_one_instance(self):
        """
        Randomly pick a distribution from self.distributions,
        generate either an ER or BA graph, then return (num_nodes, edges).
        """
        size_i = self.uniform_random_choice(self.n_node_range[0], self.n_node_range[1])
        dist_choice = random.choice(self.distributions)  # e.g. "er" or "ba"

        edges = None
        if dist_choice == "er":
            # Erdős–Rényi
            p = np.random.uniform(self.p_edge_range[0], self.p_edge_range[1])
            edges = self.generate_erdos_renyi_graph(size_i, p)
        elif dist_choice == "ba":
            # Barabási–Albert
            attach_links = self.uniform_random_choice(self.attach_links_range[0],
                                                      self.attach_links_range[1])
            edges = self.generate_barabasi_albert_graph(size_i, attach_links)
        else:
            raise ValueError(f"Unknown distribution: {dist_choice}")

        return size_i, edges, dist_choice

    def generate_instances(self, n_instance: int):
        """
        Generate a list of random MIS instances: (num_nodes, edges, dist_type).
        """
        instances = []
        for _ in range(n_instance):
            size_i, edges_i, dist_choice = self.generate_one_instance()
            instances.append((size_i, edges_i, dist_choice))
        return instances

    def generate_instances_and_save(
            self,
            n_instance: int,
            file_name_json: str,
            k_neighbors: int = 3,
            save_pkl: bool = False,
            rl_data: bool = False
    ):
        valid_json_data = []
        valid_instances_pkl = []

        pbar = tqdm(total=n_instance, desc="Generating MIS instances (ER or BA)")

        count_valid = 0
        while count_valid < n_instance:
            (num_nodes, edges, dist_choice) = self.generate_one_instance()
            vertices = list(range(num_nodes))

            try:
                # Solve MIS
                independ_set = solve_mis_gurobi(vertices, edges)

                # Tag into JSON
                mis_json = tag_prompt_and_transform_to_json_mis_with_neighbors(
                    num_nodes, edges, independ_set, k_neighbors
                )

                if rl_data:
                    mis_json["instance"] = {
                        "num_nodes": num_nodes,
                        "edges": edges,
                        "distribution": dist_choice
                    }

                valid_json_data.append(mis_json)
                # Also store raw data in a pickle-friendly format
                valid_instances_pkl.append((num_nodes, edges, independ_set))

                count_valid += 1
                pbar.update(1)

            except Exception as e:
                # If Gurobi fails for some reason, skip
                continue

        pbar.close()

        # Save JSON
        with open(file_name_json, 'w') as f:
            json.dump(valid_json_data, f, indent=4)
        print(f"Generated and saved {len(valid_json_data)} MIS instances to {file_name_json}.")

        # Optionally save pickle
        if save_pkl:
            save_dataset(valid_instances_pkl, "./instances.pkl")

###############################################################################
# Example Main
###############################################################################

if __name__ == "__main__":
    # mis_env = MISEnv(
    #     n_node_range=[10, 30],
    #     distributions=["er", "ba"],  # Randomly pick between Erdős–Rényi or Barabási–Albert
    #     p_edge_range=(0.1, 0.4),     # For ER
    #     attach_links_range=[1, 4],   # For BA
    #     seed=6
    # )

    mis_env = MISEnv(
        n_node_range=[50, 50],
        distributions=["ba"],  # Randomly pick between Erdős–Rényi or Barabási–Albert
        # p_edge_range=(0.1, 0.4),     # For ER
        attach_links_range=[4, 4],   # For BA
        seed=6
    )

    mis_env.generate_instances_and_save(
        n_instance=1000,
        file_name_json="data.json",
        k_neighbors=2,
        save_pkl=True,
        rl_data=True
    )
