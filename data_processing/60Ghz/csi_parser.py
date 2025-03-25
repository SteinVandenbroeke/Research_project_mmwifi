import numpy as np

def Parse_csi(lines):
    
    num_lines = len(lines)
    bad_idxs = []
    
    
    correct = 0
    incorrect = 0
    
    times = []
    magnitudes = np.zeros((num_lines, 32))
    phases = np.zeros((num_lines, 32))
    
    for idx, line in enumerate(lines):
        if "[AOA]" in line and line.count(",") == 71:
            splitted = line.split(",")
            time = float(splitted[1])
            
            if time not in times:
                times.append(time)
            else:
                print("duplicated")
                continue
            
            correct += 1
            if correct-1 >= num_lines:
                break
            for i in range(8, 8+32):
                phases[correct-1][i-8] = float(splitted[i])
                
            for i in range(8+32, 8+32+32):
                magnitudes[correct-1][i-(8+32)] = float(splitted[i])
                
        else:
            incorrect += 1
            bad_idxs.append(idx)
            
    
    
    # print(f"In the file {filename}, there were {correct} valid entries and {incorrect} invalid entries, with a ratio of {correct/(correct+incorrect)}.")
    
    return magnitudes, phases, times, bad_idxs
