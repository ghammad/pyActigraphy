import numpy as np
from pyActigraphy.sleep.scoring.utils import rescore

# Webster's rules:
# 1. After at least 4 minutes scored as wake, the next 1 minute scored as
# sleep is rescored as wake
# 2. After at least 10 minutes scored as wake, the next 3 minutes scored
# as sleep are rescored as wake;
# 3. After at least 15 minutes scored as wake, the next 4 minutes scored
# as sleep are rescored as wake;
# 4. If a period of 6 minutes or less that is scored as sleep is
# surrounded by at least 10 minutes scored as wake, then rescore to wake;
# 5. If a period of 10 minutes or less that is scored as sleep is
# surrounded by at least 20 minutes scored as wake, then rescore to wake.

test_series = np.asarray(
    [0]*4+[1] +  # 1st rule
    [0]*10+[1, 1, 1] +  # 2nd rule
    [0]*15+[1, 1, 1, 1] +  # 3rd rule
    [0]*10+[1, 1, 1, 1, 1, 1]+[0]*10 +  # 4th rule
    [0]*20+[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]+[0]*20  # 5th rule
)


def test_ck_rescoring():

    # The output of the rescore function is a binary series where:
    # 0 indicates that the epoch should be rescore
    # 1 indicates that the epoch is left as it is.
    # Since the rescoring affects only epochs scored as sleep, the wake epochs
    # are always set to 1 in the output.
    # By consruction, all the epochs scored as sleep in this test have to be
    # rescored.

    test = rescore(test_series, sleep_score=1)
    truth = np.logical_not(test_series).astype(float)

    assert np.all(test == truth)
