import numpy as np

def batch_shuffling(obs, fps, batch_size):

    nb_batch = len(obs) // batch_size
    obs_batches = np.array(obs)[-nb_batch*batch_size:]
    fps_batches = np.array(fps)[-nb_batch*batch_size:]

    obs_batches = obs_batches.reshape([nb_batch, batch_size, *obs_batches.shape[1:]])
    fps_batches = fps_batches.reshape([nb_batch, batch_size, *fps_batches.shape[1:]])

    # shuffle
    p = np.random.permutation(nb_batch)
    obs_batches = obs_batches[p]
    fps_batches = fps_batches[p]

    return obs_batches, fps_batches
