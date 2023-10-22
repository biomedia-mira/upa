# Automatic correction of performance drift under acquisition shift in medical image classification 

This repository contains all the necessary code to reproduce the experiments in our article
> M. Roschewitz, G. Khara, J. Yearsley, N. Sharma, J.J. James, E. Ambrózay, A. Heroux, P. Kecskemethy, T. Rijken, B. Glocker.
> [**Automatic correction of performance drift under acquisition shift in medical image classification**]([https://doi.org/10.1016/j.media.2022.102383](https://www.nature.com/articles/s41467-023-42396-y)).
> Nature Communications (2023)

## Preliminaries

This code base runs in Python. You can download a version of python via Anaconda, see instructions [here](https://docs.conda.io/en/latest/miniconda.html).
Additionally, you will need to install the following dependencies via [pip](https://pip.pypa.io/en/stable/installation/).
```
pip install numpy seaborn matplotlib pandas scikit-learn jupyter tqdm
```


## Code structure
This repository has a very simple structure:
* two notebooks necessary to reproduce all the plots for the breast screening task (see [analysis_breast_screening.ipynb](analysis_breast_screening.ipynb)) and the Wilds-CameLyon histopathology task (see [analysis_histopathology.ipynb](analysis_histopathology.ipynb)). Just run all cells to re-create all the plots, it takes about 3 mins to run on the CameLyon data and 30 mins to run for the breast task (if you want to accelerate execution you can reduce the number of bootstrap samples).
* [upa.py](upa.py) is a stand-alone file that contains all the code related to fitting and applying unsupervised prediction alignement. It is built as a class, similar to sklearn estimators with a ``fit`` and ``predict`` method, so that it can easily be applied on any model predictions. See code for in-depth documentation.
* the other files are simply utils files to compute metrics, run the experiments and prepare the plots for the manuscript.
* the necessary model outputs are provided in the [breast_outputs](breast_outputs) and [camelyon](camelyon) folders. They contain csvs with the raw model predictions and labels for each task and each dataset. 
