import unittest
import numpy as np
import sys
import os
sys.path.append("../")
from KPT.Modelling import Modelling
from KPT.PreProcessing import PreProcessing

class TestModelling(unittest.TestCase):

    """
    See this reference why I am not testing for plotting.
    https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib
    """

    def build_PreProcessing_Block(self, framework):
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

        return Model

    def test_build_model_K(self):
        """
        Test build_model when the framework is KERAS
        """

        Model = self.build_PreProcessing_Block("keras")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        result = Model.build_model("keras", architecture, No_feature=1)

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the model was build the method returns "Success"
        self.assertEqual(result, "Success")

    def test_build_model_PT(self):
        """
        Test build_model when the framework is PyTorch
        """

        Model = self.build_PreProcessing_Block("PyTorch")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]
        result = Model.build_model("PyTorch", architecture, No_feature=1)

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the model was build the method returns "Success"
        self.assertEqual(result, "Success")

    def test_build_model_exit(self):
        """
        Test if build_model exits when a wrong framework is provided
        """

        Model = self.build_PreProcessing_Block("PyTorch")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the exits
        with self.assertRaises(SystemExit):
            result = Model.build_model("not_known", architecture, No_feature=1)

    def test_summary_model_K(self):
        """
        Test summaty when the framework is KERAS
        """

        Model = self.build_PreProcessing_Block("keras")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        Model.build_model("keras", architecture, No_feature=1)
        result = Model.summary()

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the model was build the method returns "Success"
        self.assertEqual(result, "Success")

    def test_summary_model_PT(self):
        """
        Test summary when the framework is PyTorch
        """

        Model = self.build_PreProcessing_Block("PyTorch")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        Model.build_model("PyTorch", architecture, No_feature=1)
        result = Model.summary()
        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the model was build the method returns "Success"
        self.assertEqual(result, "Success")

    def test_summary_model_exit(self):
        """
        Test if summary exits when a wrong framework is provided
        """

        Model = self.build_PreProcessing_Block("PyTorch")

        architecture = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")

        # If the exits
        with self.assertRaises(SystemExit):
            Model.build_model("not_known", architecture, No_feature=1)
            result = Model.summary()
    
    def test_validate_architecture(self):
        """
        Test validate_architecture.
        """

        Model = self.build_PreProcessing_Block("keras")

        architecture_OK = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1]]

        architecture_Failed1 = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1],
            "Dummy"]

        architecture_Failed2 = [
            ["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 100, "ReLu"],
            ["Dense", 1.]]


        _, test_OK = Model._validate_architecture(architecture_OK)
        _, test_Failed1 = Model._validate_architecture(architecture_Failed1)
        _, test_Failed2 = Model._validate_architecture(architecture_Failed2)

        # Getting rid of the Dummy log file
        os.remove("Log_Dummy.log")
        
        self.assertEqual(test_OK, "Successful")
        self.assertEqual(test_Failed1, "Failed")
        self.assertEqual(test_Failed2, "Failed")

if __name__ == '__main__':
    unittest.main()
