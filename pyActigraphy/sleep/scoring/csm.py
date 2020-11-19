import numpy as np
import pandas as pd
# from ..scoring_base import _window_convolution


def _window_convolution(x, scale, window):

    return scale * np.dot(x, window)


def _calculate_score(activity, wa, wp, p):

    weigths = np.concatenate([wa, wp])

    scores = activity.rolling(len(weigths), center=True).apply(
        _window_convolution, args=(p, weigths), raw=True
    )

    return scores


def _calculate_state(
    activity,
    istate,
    wa,
    wp,
    p,
    positive_rescoring,
    negative_rescoring,
    positive_state
):

    """Sleep (re)scoring rules for the Condor Sleep Model

    Keyword arguments:
    activity -- activity used for the score calculation
    istate -- initial state
    wa -- weights for the epochs before the one being evaluated and for the
    current epoch (current epoch is the last index of the array)
    wp -- weights for the epochs after the one being evaluated
    p -- scaling for the weighted moving average
    positive_rescoring -- number of consecutive indexes of the score array that
    need to be below 1 to change the state to positive state
    negative_rescoring -- number of consecutive indexes of the score array that
    need to be above 1 to do not change the state
    positive_state -- the state that a give index will be changed to if the
    calculated score is below 1
    """
    # create a local copy of the state
    state = istate.copy()

    # calculate the score
    scores = _calculate_score(activity, wa, wp, p)

    # replace NaN produced by the convolution with default scores
    scores.iloc[:len(wa)-1] = 0
    scores.iloc[-len(wp):] = 0

    # apply rescoring
    is_positive_state = False
    for i in range(len(wa)-1, (len(scores)-len(wp))):
        # negative rescoring:
        # if score is lower than 1 (positive state), check if
        if is_positive_state:
            if scores[i] < 1:
                state[i] = positive_state
            else:
                is_positive_state = False
                for subindex in range(i+1, i+negative_rescoring+1):
                    if scores[subindex] < 1:
                        state[i] = positive_state
                        is_positive_state = True
                        break
        else:
            if scores[i] < 1:
                is_positive_state = True
                for subindex in range(i+1, i+positive_rescoring+1):
                    if scores[subindex] >= 1:
                        is_positive_state = False
                        break
            if is_positive_state:
                state[i] = positive_state

    for i in range(0, len(wa)-1):
        state[i] = state[len(wa)-1]

    for i in range(len(scores)-len(wp), len(scores)):
        state[i] = state[len(scores)-len(wp)-1]

    return state


def csm(
    data,
    wa=np.array([34.5, 133, 529, 375, 408, 400.5, 1074, 2048.5, 2424.5]),
    wp=np.array([1920, 149.5, 257.5, 125, 111.5, 120, 69, 40.5]),
    ps=0.000464,  # p for sleep detection
    pr=0.00005,  # p for resting detection
    prs=1,  # positive rescorring for sleep detection
    nrs=0,  # negative rescorring for sleep detection
    prr=0,  # positive rescorring for resting detection
    nrr=0,  # negative rescorring for resting detection
    rest_score=2,
    sleep_score=1
):

    # start with all states awake
    state0 = np.zeros(len(data))

    stateR = _calculate_state(data, state0, wa, wp, pr, prr, nrr, rest_score)
    stateF = _calculate_state(data, stateR, wa, wp, ps, prs, nrs, sleep_score)

    return pd.Series(index=data.index, data=stateF)
