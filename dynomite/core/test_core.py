import unittest

import numpy as np
from dynomite.core.time import TimeSeries

class TestCore(unittest.TestCase):
    def test_1(self):
        time = np.linspace(0., 1., num=101)
        time_response = 2 * np.sin(time)
        resp = TimeSeries(time, time_response, label='cat')
        resp.plot(y_units='g', ifig=1, ax=None, linestyle='-o', show=True)

if __name__ == '__main__':
    unittest.main()
