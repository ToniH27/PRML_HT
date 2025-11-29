## Guide on how to run the `digit_classify.py` file


### Dependencies
digit_classify function requires that the dependencies in `requirements.txt` are installed.

You can install the dependencies with `pip install -r requirements.txt` command in CLI or `!pip install -r requirements.txt` in Jupyter Notebook cell.

If you use `conda`, then you can use `conda list` command in CLI or `!conda list` in Jupyter Notebook to check if the required dependencies are installed.

Alternatively, if you use `conda`, then you can create a test environment by using `conda create --name <env_name> python numpy scipy` command if performing testing with CLI or `conda create --name <env_name> python numpy scipy ipykernel` command if you are going to use Jupyter Notebook for testing. Note that, you can define Python version or package versions if you want, but the newest versions are recommended.

After testing, you can remove the testing environment with `conda env remove --name <env_name>` command in CLI.


### Running

If you are testing the `digit_classify` function in CLI, then just add needed precodures for predictions at the bottom of the file. If you are testing `digit_classify` in Jupyter Notebook, then import these libraries:
`import numpy as np`
`from digit_classify import digit_classify`

Note that, `digit_classify` returns `int` type values for each time series sample prediction.
