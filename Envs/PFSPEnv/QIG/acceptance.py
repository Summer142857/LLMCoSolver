import math
import random
def acceptance_function(Self):

    if Self.new_solution.makespan <= Self.current_solution.makespan:
        # Accept new solution
        Self.current_solution.sequence = Self.new_solution.sequence.copy()
        Self.current_solution.makespan = Self.new_solution.makespan

        # Check to update best solution
        if Self.current_solution.makespan < Self.best_solution.makespan:
            Self.best_solution.makespan = Self.current_solution.makespan
            Self.best_solution.sequence = Self.current_solution.sequence.copy()
    else:
        # Metropolis acceptance criterion
        delta = Self.new_solution.makespan - Self.current_solution.makespan
        probability = math.exp(- delta / Self.temperature)
        if random.random() <= probability:
            Self.current_solution.sequence = Self.new_solution.sequence.copy()
            Self.current_solution.makespan = Self.new_solution.makespan