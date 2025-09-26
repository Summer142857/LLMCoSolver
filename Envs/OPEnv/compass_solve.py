import os
import numpy as np
from subprocess import check_output
import re

MAX_LENGTH_TOL = 1e-5

# def solve_euclidian_op(depot, loc, prize, max_length, threads=0, timeout=10, gap=None):
#     """
#     Solves the Euclidian OP to optimality using a MIP formulation
#     with lazy subtour elimination constraint generation.
#     """
#     from gurobipy import Model, GRB, quicksum, tuplelist
#     import math
#     import itertools
#
#     points = [depot.tolist()] + loc.tolist()
#     n = len(points)
#
#     # Callback - use lazy constraints to eliminate sub-tours
#     def subtourelim(model, where):
#         if where == GRB.Callback.MIPSOL:
#             vals = model.cbGetSolution(model._vars)
#             selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
#             tour = subtour(selected)
#             if tour is not None:
#                 # Lazy subtour elimination constraint
#                 model.cbLazy(quicksum(model._vars[i, j]
#                                       for i, j in itertools.combinations(tour, 2))
#                              <= quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour)))
#
#     def subtour(edges, exclude_depot=True):
#         unvisited = list(range(n))
#         cycle = None
#         while unvisited:
#             thiscycle = []
#             neighbors = unvisited
#             while neighbors:
#                 current = neighbors[0]
#                 thiscycle.append(current)
#                 unvisited.remove(current)
#                 neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
#             if ((cycle is None or len(cycle) > len(thiscycle))
#                 and len(thiscycle) > 1
#                 and not (0 in thiscycle and exclude_depot)):
#                 cycle = thiscycle
#         return cycle
#
#     # Dictionary of Euclidean distances (store only i > j, for instance)
#     dist = {
#         (i, j): math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
#         for i in range(n) for j in range(i)
#     }
#
#     m = Model()
#     m.Params.outputFlag = False
#
#     # Create variables for all (i, j) with i>j
#     vars = m.addVars(dist.keys(), vtype=GRB.BINARY, name='e')
#
#     # Now fill in the symmetric entries (j, i) = (i, j), without changing dict size during iteration
#     original_keys = list(vars.keys())
#     for (i, j) in original_keys:
#         if (j, i) not in vars:
#             vars[j, i] = vars[i, j]
#
#     # Allow depot edges (i=0 or j=0) to be up to 2 in integer sense
#     for (i, j) in vars.keys():
#         if i == 0 or j == 0:
#             vars[i, j].vtype = GRB.INTEGER
#             vars[i, j].ub = 2
#
#     # Create prize (we want to maximize sum of visited prizes => negative cost for Gurobi minimization)
#     prize_dict = {i + 1: -p for i, p in enumerate(prize)}
#     delta = m.addVars(range(1, n), obj=prize_dict, vtype=GRB.BINARY, name='delta')
#
#     # Degree constraints: sum of edges from node i must be 2 if depot, else 2 * delta[i]
#     m.addConstrs(
#         (vars.sum(i, '*') == (2 if i == 0 else 2*delta[i]) for i in range(n)),
#         name='degree'
#     )
#
#     # Length constraint
#     m.addConstr(
#         quicksum(vars[i, j] * dist[i, j] for (i, j) in dist.keys()) <= max_length,
#         name='length'
#     )
#
#     # Set MIP parameters
#     m._vars = vars
#     m._dvars = delta
#     m.Params.lazyConstraints = 1
#     m.Params.threads = threads
#     if timeout:
#         m.Params.timeLimit = timeout
#     if gap:
#         m.Params.mipGap = gap * 0.01  # interpret gap as a percentage
#
#     # Optimize
#     m.optimize(subtourelim)
#
#     # Extract solution
#     vals = m.getAttr('x', vars)
#     selected = tuplelist((i, j) for (i, j) in vals.keys() if vals[i, j] > 0.5)
#     tour = subtour(selected, exclude_depot=False)
#     assert tour[0] == 0, "Tour should start with depot"
#
#     return -m.objVal, tour

# Run install_compass.sh to install
def solve_compass(executable, depot, loc, demand, capacity):
    problem_filename = os.path.join("./", "problem.oplib")

    write_oplib(problem_filename, depot, loc, demand, capacity)
    output = check_output([executable, "opt", "--op-exact", "0", problem_filename])

    obj_pattern = r"Objetive value:\s*([0-9]+\.[0-9]+)"
    cycle_pattern = r"Cycle:\s*([0-9\s]+)"
    obj_match = re.search(obj_pattern, output.decode("utf-8"))

    objective_value, cycle = None, None
    if obj_match:
        objective_value = float(obj_match.group(1))
    else:
        print("Objective value not found.")

    cycle_match = re.search(cycle_pattern, output.decode("utf-8"))
    if cycle_match:
        cycle_str = cycle_match.group(1).strip()  # remove any extra spaces
        cycle = list(map(int, cycle_str.split()))
    else:
        print("Cycle not found.")
    return objective_value, cycle


def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum()


def calc_op_length(depot, loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_compass_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 1000,
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


def read_oplib(filename, n):
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

            if line.startswith("NODE_SEQUENCE_SECTION"):
                started = True

    assert len(tour) > 0, "Unexpected length"
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_oplib(filename, depot, loc, prize, max_length, name="problem"):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc)),
                ("COST_LIMIT", int(max_length)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x), int(y))  # oplib does not take floats
            # "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")

if __name__ == "__main__":
    depot = [0, 0]
    loc = [[1, 1], [1, 2], [2, 2], [2, 1]]
    prize = [0, 1, 2, 3]
    max_length = 100
    write_oplib("problem.oplib", depot, loc, prize, max_length)
    result, output, duration = solve_compass("./op-solver/build/src/op-solver", depot, loc, prize, max_length)
    print(result)
    print(output)
    print(duration)
    print(calc_op_total(prize, result))
    print(calc_op_length(depot, loc, result))
    print("Done")