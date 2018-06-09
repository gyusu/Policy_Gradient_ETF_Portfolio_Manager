import numpy as np

def batch_shuffling(obs, fps, batch_size):

    obs = np.array(obs)
    fps = np.array(fps)

    obs_batches = []
    fps_batches = []
    for i in range(len(obs)-batch_size+1):
        obs_batches.append(obs[i:i+batch_size])
        fps_batches.append(fps[i:i+batch_size])

    obs_batches = np.array(obs_batches)
    fps_batches = np.array(fps_batches)

    # shuffle
    p = np.random.permutation(len(obs_batches))
    obs_batches = obs_batches[p]
    fps_batches = fps_batches[p]


    return obs_batches, fps_batches
