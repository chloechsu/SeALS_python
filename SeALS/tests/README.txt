To run the tests by use the command-line interface from the SeALS directory:
    python -m unittest tests.module_name.TestClass

Example:
    python -m unittest tests.cp_tensor_unittest.TestCPTensorMethods

Or, to run all unittests by the discovery command:
    python -m unittest discover tests/ "*unittest.py"

