import unittest
import numpy as np
import sys
import os
sys.path.append("../")
from KPT.Modelling import Modelling
from KPT.PreProcessing import PreProcessing
from KPT.PostProcessing import PostProcessing


class TestPostProcessing(unittest.TestCase):

    """
    See this reference why I am not testing for plotting.
    https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib
    """

    def build_prepro_and_train_block(self, framework):
        """Build the Pre-Processing Block

        This is not an unittest. It just build the pre-processing
        block in order to test the modelling block.

        Parameters
        ----------
        framework : string
            Either Keras or Pytorch

        Returns
        -------
        Model : instance
            Model instance
        """

        # Create a dummy Pre-processing instance
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        PP = PreProcessing(x, y, with_noise=False, log_file_name="Log_Dummy")
        PP._construct_function(x, y)
        PP.split_dataset(method=3, test_size=0.2)
        result = PP.prepare_input(framework)
        # Modelling
        Model = Modelling(PP)

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]
        ]

        Model.build_model(framework, architecture, No_feature=1)
        Model.train(verbose=False, lr=0.01, patience=1, epoch=1)
        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        return Model

    def test_get_metrics_K(self):
        """
        Test get_metrics when model = keras
        
        It is difficult to test if the metrics are calculated OK due to
        instrinsic nature of the ANNs. We'll pass the test if no other
        exception are thrown. Essentially as long as a numbers is printed
        on the console.
        """

        Model = self.build_prepro_and_train_block("keras")

        # Create post-processing objectss
        result = PostProcessing(Model)
        
        self.assertEqual(result.get_metrics(), "Success")

    def test_get_metrics_PT(self):
        """
        Test get_metrics when model = PyTorch
        
        It is difficult to test if the metrics are calculated OK due to
        instrinsic nature of the ANNs. We'll pass the test if no other
        exception are thrown. Essentially as long as a numbers is printed
        on the console.
        """

        Model = self.build_prepro_and_train_block("PyTorch")

        # Create post-processing objectss
        result = PostProcessing(Model)
        
        self.assertEqual(result.get_metrics(), "Success")

if __name__ == '__main__':
    unittest.main()
