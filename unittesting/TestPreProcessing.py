from KPT.PreProcessing import PreProcessing
import unittest
import numpy as np
import sys
import os
sys.path.append("../")


class TestPreProcessing(unittest.TestCase):

    """
    See this reference why I am not testing for plotting.
    https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib
    """

    def test_construct_function_with_noise(self):
        """
        Test construct testing function when noise is added.
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        PP = PreProcessing(x, y, with_noise=True, log_file_name="Log_Dummy")

        # Getting rid of the Dummy log file
        os.remove("./Log_Dummy.log")
        result = PP._construct_function(x, y)

        # Check that all the entries are different
        self.assertTrue((result != y).all())

    def test_construct_function_no_noise(self):
        """
        Test construct testing function when no noise is added.
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")

        # Getting rid of the Dummy log file
        os.remove("./Log_Dummy.log")
        result = PP._construct_function(x, y)

        # Check that all the entries are equal
        self.assertTrue((result == y).all())

    def test_split_dataset_test_size(self):
        """
        Test split dataset function when the test_size is
        20% of the whole dataset
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")
        dummy = PP._construct_function(x, y)

        result = PP.split_dataset(method=2, test_size=0.2)

        # Check that all the entries are equal
        self.assertEqual(len(result["X_test"]), 2)

    def test_split_dataset_method_three(self):
        """
        Test split dataset function when method = 3.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")
        dummy = PP._construct_function(x, y)

        result = PP.split_dataset(method=3, test_size=0.2)

        print(result.keys())
        # Check they key "X_val" is present
        self.assertTrue("X_val" in result.keys())

    def test_split_dataset_method_two(self):
        """
        Test split dataset function when method = 2.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")
        dummy = PP._construct_function(x, y)

        result = PP.split_dataset(method=2, test_size=0.2)

        print(result.keys())
        # Check they key "X_val" is present
        self.assertTrue("X_val" not in result.keys())

    def test_prepare_input_Keras(self):
        """
        Test prepare_input when framework=keras
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")
        PP._construct_function(x, y)
        PP.split_dataset(method=3, test_size=0.2)
        result = PP.prepare_input("keras")

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # Check they key "X_val" is present
        self.assertTrue("K_X_val" in result.keys())

    def test_prepare_input_PyTorch(self):
        """
        Test prepare_input when framework=PyTorch
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")
        PP._construct_function(x, y)
        PP.split_dataset(method=3, test_size=0.2)
        result = PP.prepare_input("PyTorch")

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # Check they key "X_val" is present
        self.assertTrue("PT_X_val" in result.keys())


if __name__ == '__main__':
    unittest.main()
