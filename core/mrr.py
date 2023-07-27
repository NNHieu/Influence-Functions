import numpy as np

def compute_mean_reciprocal_rank(rs):
    '''
    rs: 2d array
    rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    mean_reciprocal_rank(rs)
    0.61111111111111105
    rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    mean_reciprocal_rank(rs)
    0.5
    rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    mean_reciprocal_rank(rs)
    0.75
    '''

    rs = (np.asarray(r).nonzero()[0] for r in rs)
    (1 / np.arange(1, r.size) * r).sum() / r.sum() 
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

if __name__ == "__main__":
    rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    print(mean_reciprocal_rank(rs))