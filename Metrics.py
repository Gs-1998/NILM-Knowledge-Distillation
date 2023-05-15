import numpy as np

def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn


def get_accuracy(pred, ground, threshold):
    pr = np.array([0 if (p) < threshold else 1 for p in pred])
    gr = np.array([0 if p < threshold else 1 for p in ground])

    tp, tn, fp, fn = tp_tn_fp_fn(pr, gr)
    p = np.sum(pr)
    n = len(pr) - p

    res_accuracy = accuracy(tp, tn, p, n)

    return (res_accuracy)


def get_sae(pred, ground):
    E_pred = np.sum(pred)
    E_ground = np.sum(ground)
    return np.abs(E_pred - E_ground) / float(max(E_pred, E_ground))


def get_mae(pred, ground):
    total_sum = np.sum(np.abs(pred - ground))

    return total_sum / len(pred)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)

def get_sae(target, prediction, sample_second):

    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)

    sae = np.abs(r - rhat) / np.abs(r)

    return sae

def get_nde(target, prediction):
    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))