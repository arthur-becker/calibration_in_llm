from unittest import TestCase
from utils.convert import position_result_to_numpy
from utils.position_result import PositionTopResult
import numpy as np

class TestConverting(TestCase):

    def test_position_result_to_numpy(self):
        # NOTE: This test covers only a fraction of important cases

        logits1 = np.array([1,2,3,4,5,6])
        logits2 = np.array([3,5,3,2,6,5])
        logits3 = np.array([1,1,1,1,1,1])
        logits_combined = np.concatenate((logits1, logits2, logits3))
        print(f'logits_combined: {logits_combined}')
        size = 6

        assert len(logits1) == size
        assert len(logits2) == size
        assert len(logits3) == size

        position_top_result_list = [
            PositionTopResult(logits1, size),
            PositionTopResult(logits2, size),
            PositionTopResult(logits3, size)
        ]
        y_true, y_value, read_size = position_result_to_numpy(position_top_result_list)

        assert read_size == size

        for i in range(len(logits_combined)):
            assert logits_combined[i] == y_value[i]

        