# PFSPEnv: Environment for Permutation Flow Shop Problem

This environment is designed to generate instances of the Permutation Flow Shop Problem (PFSP) and solve them using the SOTA solver [QIG](https://www.sciencedirect.com/science/article/pii/S0377221722002788?via%3Dihub).

## Preliminaries

### Permutation Flow Shop Problem

The Permutation Flow Shop Problem (PFSP) is a classical scheduling problem where n jobs need to be processed on m machines in the same order. Each job consists of m operations, and the j-th operation of each job must be processed on machine j. The processing time of each operation is known in advance. The goal is to find a permutation (sequence) of jobs that minimizes a certain objective, typically the makespan (total completion time of all jobs) or total flowtime.

Key characteristics of PFSP:
- All jobs follow the same processing route through the machines
- Each machine can process only one job at a time
- Each job can be processed on only one machine at a time
- Once a job starts processing on a machine, it cannot be interrupted
- The job sequence must remain the same on all machines

PFSP is known to be NP-hard for most objective functions when the number of machines is greater than 2, making it a challenging combinatorial optimization problem that has attracted significant research attention.

### QIG

QIG (Q-learning based Iterated Greedy) is a state-of-the-art algorithm for solving the PFSP that combines the Iterated Greedy (IG) framework with Q-learning to adaptively control the perturbation phase.

IG algorithm and its variants are considered as one of the most effective algorithms for PFSP. It consists of four main phases:
1. Destruction: Remove d jobs from the current solution
2. Construction: Reinsert the removed jobs using a greedy heuristic
3. Local Search: Improve the constructed solution
4. Acceptance: Accept or reject the new solution based on a criterion

QIG enhances the IG algorithm by introducing a Q-learning mechanism to dynamically determine the destruction size d. The Q-learning agent:
- Observes the current solution state and solution quality
- Selects a destruction size action based on learned Q-values
- Receives rewards based on solution improvement
- Updates Q-values to learn effective perturbation strategies

Experiments have shown QIG outperforms other IG variants, making it one of the state-of-the-art solvers for PFSP.


## How to use this Environment

1. Ensure `cython` has been install. Enter `QIG/`, and run `python install_me_first.py build_ext --inplace`.