
# Machine Learning Examples

Some simple machine learning examples using Keras and the MNIST database
## Python instalation

For the Python scripts you need to install anaconda with [tensorflow](https://www.tensorflow.org/) and some other libraries. To do so, install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) and run Anaconda Prompt Terminal. Alternatively one can use the [miniconda version](https://docs.conda.io/en/latest/miniconda.html).

From there create an environment and update the packages via the following code:

 ```concole
 conda create -n ML_tensor python=3.9
 conda activate ML_tensor
 pip install tensorflow
 conda install spyder numpy matplotlib seaborn scikit-learn pandas
 conda install -c conda-forge jupyterlab
 ```
 
 Optionally you can run:
 
 ```concole
 conda update -n base -c defaults conda
 conda update --all
 ```

Then you can open Spyder IDE with the interpreter being the virtual environment "ML_tensor". Alternatively, you can start a Jupyter notebook session and read the notebooks of this repository.

## Alternative solution

You may also run the notebooks from the cloud through Binder. Keep in mind this might be a bit slow. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SavvasRaptis/machine-learning-examples/HEAD)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

