def optimality_reward_func_pfsp(completions, ground_truth, instance, **kwargs) -> list[float]:
    """
    Calculate the optimality reward for the PFSP.
    The optimality is measured by how close the makespan is to the optimal makespan.
    """
    scores = []
    responses = completions
    feasible_rewards = feasibility_reward_func_pfsp(completions, instance)
    
    for i, (response, is_feasible) in enumerate(zip(responses, feasible_rewards)):
        if is_feasible != 2.0:
            # Infeasible solution
            scores.append(0.0)
            continue
            
        # Parse the job order
        pred_match = re.search(r"Order:\s*\[([^\]]+)\]", response)
        job_order_str = pred_match.group(1)
        job_order = list(map(int, job_order_str.split(", ")))
        
        # Get processing times and calculate makespan
        try:
            # Try to convert the instance to numpy array if it's not already
            if hasattr(instance[i], 'shape'):
                processing_times = instance[i].T  # Transpose to match calculate_pfsp_makespan format
            else:
                # If it's a list or other format, convert to numpy array
                inst_array = np.array(instance[i])
                processing_times = inst_array.T
                
            m_machines = processing_times.shape[0]
            predicted_makespan = calculate_pfsp_makespan(job_order, processing_times, m_machines)
            
            # Parse the reference (optimal) objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", ground_truth[i])
            if not label_obj_match:
                scores.append(0.0)
                continue
                
            optimal_makespan = float(label_obj_match.group(1))
            
            # Compute gap = (predicted_makespan - optimal_makespan) / optimal_makespan
            # Convert to a score between 0 and 1
            gap = (predicted_makespan - optimal_makespan) / optimal_makespan
            scores.append(max(0.0, 1.0 - gap))
        except Exception as e:
            # Error in calculation
            print(f"Error calculating makespan: {e}")
            scores.append(0.0)
        
    return scores
