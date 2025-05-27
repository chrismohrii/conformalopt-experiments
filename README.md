This repository contains the code to reproduce the experiments in Online Conformal Prediction via Online Optimization by Felipe Areces, Christopher Mohri, Tatsunori Hashimoto, and John Duchi.

We have also released documentation for our pypi package, conformalopt, here: https://conformalopt.readthedocs.io/. The package directly implements all of our algorithms. 

To reproduce the main experiments, 

1. Install the necessary requirements: `pip install -r requirements.txt`
2. Reproduce results: `bash run_expts.sh`. Some of the algorithms, like conformal PID, can be very slow. These are commented out from `methods_to_run` in main.py and `PREREGISTERED_METHODS` in preregistered.py to save time. Uncomment them to obtain results for them. The electricity dataset is also large and commented out in `run_expts.sh`; uncomment it to obtain results. 
4. Look at results in reproduced_experiments/
