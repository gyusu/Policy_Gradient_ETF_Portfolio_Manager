import numpy as np

def batch_shuffling(obs, fps, batch_size, batch_rolling=True):

    obs = np.array(obs)
    fps = np.array(fps)

    obs_batches = []
    fps_batches = []
    if batch_rolling:
        for i in range(len(obs)-batch_size+1):
            obs_batches.append(obs[i:i+batch_size])
            fps_batches.append(fps[i:i+batch_size])
    else:
        # 만약 배치 롤링을 사용하지 않는다면,
        # len(obs) % batch_size 만큼 train 데이터 버림
        offset = len(obs) % batch_size
        for i in range(len(obs)//batch_size):
            idx_from = offset + i * batch_size
            idx_to = idx_from + batch_size
            obs_batches.append(obs[idx_from: idx_to])
            fps_batches.append(fps[idx_from: idx_to])

    obs_batches = np.array(obs_batches)
    fps_batches = np.array(fps_batches)

    # shuffle
    p = np.random.permutation(len(obs_batches))
    obs_batches = obs_batches[p]
    fps_batches = fps_batches[p]

    return obs_batches, fps_batches


def asset_shuffling(obs, fps):
    """
    obs, fps의 asset 부분을 랜덤하게 섞어준다.
    """
    obs = np.array(obs)
    fps = np.array(fps)

    # shuffle
    p = np.random.permutation(obs.shape[1])
    obs = obs[:, p]
    fps = fps[:, p]

    return obs, fps