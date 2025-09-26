
import numpy as np
from subprocess import check_call
from urllib.parse import urlparse
import time
import os
import pickle
import torch
import subprocess
from tqdm import tqdm
from utils import check_extension, save_dataset
import matplotlib.pyplot as plt

def get_lkh_solutions(depots, locs, demands, capacity):
    if isinstance(depots, torch.Tensor):
        depots = depots.detach().cpu().numpy()
    if isinstance(locs, torch.Tensor):
        locs = locs.detach().cpu().numpy()
    if isinstance(demands, torch.Tensor):
        demands = demands.detach().cpu().numpy()
    depot = depots.tolist()
    coords = locs.tolist()
    demand = demands.astype('int').tolist()
    tour, _ = solve_lkh(depot, coords, demand, capacity)
    val, _ = calc_vrp_cost(depot, coords, tour, demand)
    return tour, val


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz"):

    cwd = os.path.abspath(os.path.join("problems", "vrp", "lkh"))
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH.exe")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def solve_lkh(depot, loc, demand, capacity):
    executable = get_lkh_executable()
    filedir = "./problems/vrp/lkh/"
    problem_filename = os.path.join(filedir, "problem.vrp")
    output_filename = os.path.join(filedir, "output.tour")
    param_filename = os.path.join(filedir, "params.par")

    starttime = time.time()
    write_vrplib(problem_filename, depot, loc, demand, capacity, 1)
    params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
    write_lkh_par(param_filename, params)
    fh = open("NUL", "w")
    check_call([executable, param_filename], stderr=fh, stdin=subprocess.DEVNULL, stdout=fh)
    fh.close()
    result = read_vrplib(output_filename, n=len(demand))
    duration = time.time() - starttime
    return result, duration

def calc_vrp_cost(depot, loc, tour, demand):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot), np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    # plt.scatter(sorted_locs[:, 0], sorted_locs[:, 1], c='red', marker='o', label='Cities')
    # plt.plot(sorted_locs[:, 0], sorted_locs[:, 1], linestyle='-', linewidth=2, marker='o', markersize=8,
    #          color='blue', label='Tour')
    # plt.show()

    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum(), sorted_locs


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{} {} {}".format(i + 1, int(x), int(y))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{} {}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")