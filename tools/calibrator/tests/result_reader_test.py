from unittest import TestCase
from utils.result_reader import ResultReader
import numpy as np
from utils.position_result import PositionFullResult, PositionTopResult
from decimal import Decimal, getcontext
from utils.softmax import softmax
from experiment_info import read_yaml

class TestResultReader(TestCase):
    """
    Test the ResultReader class

    This test assumes the existance of the following folders:
    - outputs/extractor/mock_writer_type_full/
    - outputs/extractor/mock_writer_type_top/
    """

    def __init__(self, *args, **kwargs):
        super(TestResultReader, self).__init__(*args, **kwargs)
        
        # Read the little_endian from the info.yaml file
        # Assumption: mock_writer_type_full and mock_writer_type_top have the same little_endian
        path = f'./../../outputs/extractor/mock_writer_type_full/'
        run_info = read_yaml('./../../outputs/extractor/mock_writer_type_full/info.yaml')
        self.little_endian = run_info['little_endian']

        # For testing the full result
        self.logits1 = np.array([0.1, 0.2, 0.6, 0.3], dtype=np.float32)
        self.correct_token1 = 1
        self.logits2 = np.array([0.2, 0.5, 0.3, 0.2, 0.8, 0.1], dtype=np.float32)
        self.correct_token2 = 2

        # For testing the top result
        #
        # The logit of the correct token is the first element in the token_data,
        # the rest of the elements are the logits of the top_n-1 tokens
        # in descending order
        self.top_n = 3
        self.top_logits1 = np.array([0.2, 0.6, 0.3], dtype=np.float32)
        self.top_logits2 = np.array([0.3, 0.8, 0.5], dtype=np.float32)
        

    def test_read_full_logits(self):
        reader = ResultReader('./../../outputs/extractor/mock_writer_type_full/output.full.logits', little_endian=self.little_endian)
        results = list(reader.read())
        self.assertEqual(len(results), 2)
        np.testing.assert_equal(results[0].token_data, self.logits1)
        self.assertEqual(results[0].correct_token, self.correct_token1)
        np.testing.assert_equal(results[1].token_data, self.logits2)
        self.assertEqual(results[1].correct_token, self.correct_token2)

    def test_read_top_logits(self):
        reader = ResultReader('./../../outputs/extractor/mock_writer_type_top/output.top.logits', little_endian=self.little_endian)
        results = list(reader.read())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].n, self.top_n)
        np.testing.assert_equal(results[0].token_data, self.top_logits1)
        self.assertEqual(results[1].n, self.top_n)
        np.testing.assert_equal(results[1].token_data, self.top_logits2)

    def test_softmax(self):
        # The sum of the softmax of the logits should be 1
        reader = ResultReader('./../../outputs/extractor/mock_writer_type_full/output.full.logits', little_endian=self.little_endian)
        results = list(reader.read())
        
        for result in results:
            self.assertAlmostEqual(np.sum(softmax(result)), 1.0)

    def test_reading_proba(self):
        """
        Tests the reading of the file with probabilities. Only the full result is tested
        """
        # The sum of the softmax of the logits should be 1
        reader = ResultReader('./../../outputs/extractor/mock_writer_type_full/output.full.proba', little_endian=self.little_endian)
        results = list(reader.read())
        
        logits_full_result1 = PositionFullResult(self.logits1, self.correct_token1)
        logits_full_result2 = PositionFullResult(self.logits2, self.correct_token2)

        proba_full_result1 = PositionFullResult(softmax(logits_full_result1), self.correct_token1)
        proba_full_result2 = PositionFullResult(softmax(logits_full_result2), self.correct_token2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].correct_token, self.correct_token1)
        self.assertEqual(results[1].correct_token, self.correct_token2)

        # TODO: clarify if this precision is enouph
        float32_decimal_precision = 7
        np.testing.assert_almost_equal(
            results[0].token_data, 
            proba_full_result1.token_data, 
            decimal=float32_decimal_precision)
        np.testing.assert_almost_equal(
            results[1].token_data,
            proba_full_result2.token_data,
            decimal=float32_decimal_precision)
        

    