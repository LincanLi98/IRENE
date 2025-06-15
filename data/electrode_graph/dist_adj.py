import pandas as pd 
import numpy as np 
import pickle

INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ'
]


def get_adjacency_matrix(distance_df, sensor_ids, dist_k=0.9):
    """
    Args:
        distance_df: data frame with three columns: [from, to, distance].
        sensor_ids: list of sensor ids.
        dist_k: threshold for graph sparsity
    Returns:
        adj_mx: adj
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.    
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[dist_mx > dist_k] = 0
   
    return adj_mx, sensor_id_to_ind



def get_default_adj_mat():
    dist_df = pd.read_csv('distances_3d.csv')
    thresh = 1.1
    adj_mat, sensor_id_to_ind = get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)
    return adj_mat


if __name__ == '__main__':
    adj_mat = get_default_adj_mat()
    print("The shape of adjacency matrix:", adj_mat.shape)
    with open('adj_mx_3d.pkl', 'wb') as f:
        pickle.dump(adj_mat, f)















