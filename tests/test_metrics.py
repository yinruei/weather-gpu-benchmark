import unittest
import numpy as np
from benchmark import numpy_metrics

class MetricsTest(unittest.TestCase):
    def test_known_errors(self):
        result = numpy_metrics(np.array([1., -2., 3.], dtype=np.float32), np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(result, [2/3, 2, np.sqrt(14/3)], rtol=1e-6)
    def test_invalid_inputs(self):
        for f, o in [(np.zeros(2), np.zeros(1)), (np.array([]), np.array([])), (np.array([np.nan]), np.array([0.]))]:
            with self.assertRaises(ValueError):
                numpy_metrics(f, o)

if __name__ == '__main__':
    unittest.main()
