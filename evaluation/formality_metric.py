import numpy as np

def eval_form(golden_mungchi, predicted_mungchi):
    # Make sure the two lists have the same length
    if len(golden_mungchi) > len(predicted_mungchi):
        total = len(golden_mungchi)
        predicted_mungchi += [0] * (total - len(predicted_mungchi))
    elif len(golden_mungchi) < len(predicted_mungchi):
        total = len(predicted_mungchi)
        golden_mungchi += [0] * (total - len(golden_mungchi))
    else:
        total = len(golden_mungchi)
        
    # Calculate accuracy
    correct_list = [x == y for x, y in zip(golden_mungchi, predicted_mungchi)]
    accuracy = sum(correct_list) / total
    
    # Calculate MSE
    golden_mungchi_flat = np.array(golden_mungchi).flatten()
    predicted_mungchi_flat = np.array(predicted_mungchi).flatten()
    squred_error_list = [(x - y) ** 2 for x, y in zip(golden_mungchi_flat, predicted_mungchi_flat)]
    mse = np.mean(squred_error_list)
    
    return accuracy, mse

def eval_ours_form(golden_mungchi, predicted_mungchi):
    # Make sure the two lists have the same length
    if len(golden_mungchi) > len(predicted_mungchi):
        total = len(golden_mungchi)
        predicted_mungchi += [0] * (total - len(predicted_mungchi))
    elif len(golden_mungchi) < len(predicted_mungchi):
        total = len(predicted_mungchi)
        golden_mungchi += [0] * (total - len(golden_mungchi))
    else:
        total = len(golden_mungchi)
        
    # Calculate Ours MSE
    golden_mungchi_flat = np.array(golden_mungchi).flatten()
    predicted_mungchi_flat = np.array(predicted_mungchi).flatten()

    ours_mse_list = []
    for golden, predict in zip(golden_mungchi_flat, predicted_mungchi_flat):
        if abs(golden - predict) > 2:
            ours_mse = 0.0
        elif abs(golden - predict) > 0:
            ours_mse = 0.5
        elif abs(golden - predict) == 0:
            ours_mse = 1.0
        ours_mse_list.append(ours_mse)

    ours_mse = np.mean(ours_mse_list)
    
    return ours_mse