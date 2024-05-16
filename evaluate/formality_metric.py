import numpy as np

def eval_form(golden_mungchi_integer, predict_mungchi_integer):
    # Make sure the two lists have the same length
    if len(golden_mungchi_integer) > len(predict_mungchi_integer):
        total = len(golden_mungchi_integer)
        predicted_mungchi += [0] * (total - len(predict_mungchi_integer))
    elif len(golden_mungchi_integer) < len(predict_mungchi_integer):
        total = len(predict_mungchi_integer)
        golden_mungchi_integer += [0] * (total - len(golden_mungchi_integer))
    else:
        total = len(golden_mungchi_integer)
        
    # Calculate accuracy
    correct_list = [x == y for x, y in zip(golden_mungchi_integer, predict_mungchi_integer)]
    accuracy = sum(correct_list) / total
    
    # Calculate MSE
    golden_mungchi_flat = np.array(golden_mungchi_integer).flatten()
    predicted_mungchi_flat = np.array(predict_mungchi_integer).flatten()
    squred_error_list = [(x - y) ** 2 for x, y in zip(golden_mungchi_flat, predicted_mungchi_flat)]
    mse = np.mean(squred_error_list)
    
    return accuracy, mse

def eval_our_form(golden_mungchi_integer, predict_mungchi_integer):
    # Make sure the two lists have the same length
    if len(golden_mungchi_integer) > len(predict_mungchi_integer):
        total = len(golden_mungchi_integer)
        predict_mungchi_integer += [0] * (total - len(predict_mungchi_integer))
    elif len(golden_mungchi_integer) < len(predict_mungchi_integer):
        total = len(predict_mungchi_integer)
        golden_mungchi_integer += [0] * (total - len(golden_mungchi_integer))
    else:
        total = len(golden_mungchi_integer)
        
    # Calculate Ours MSE
    golden_mungchi_flat = np.array(golden_mungchi_integer).flatten()
    predicted_mungchi_flat = np.array(predict_mungchi_integer).flatten()

    ours_mse_list = []
    for golden, predict in zip(golden_mungchi_flat, predicted_mungchi_flat):
        if abs(golden - predict) > 2:
            our_mse = 0.0
        elif abs(golden - predict) > 0:
            our_mse = 0.5
        elif abs(golden - predict) == 0:
            our_mse = 1.0
        ours_mse_list.append(our_mse)

    our_mse = np.mean(ours_mse_list)
    
    return our_mse