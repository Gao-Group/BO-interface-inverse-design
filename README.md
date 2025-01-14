# Inverse Design of Non-linear Mechanics of Bio-inspired Materials Through Interface Engineering and Bayesian Optimization
## Introduction
This is the Python code and database for the Inverse design of a Nacre (a type of bio-inspired materials) model realized with the Bayesian optimization. The parameters of the interface law are designed inversely with the given target stress-strain curve. The related paper can be found on arXiv: https://arxiv.org/pdf/2412.14071. The algorithm together with some details and two examples can be found in the paper. 
## Project files description
1. `BO_functions.py`: The python file with functions to implement the Bayesian optimization;
2. `DB_5param_small_50_database.zip`: The 50 initial database used for training of the Bayesian optimization generated from ABAQUS;
3. `Nacre_2D_inp.py`: The python file run in ABAQUS;
4. `Nacre_model.inp`: The modeling file for the Nacre used in ABAQUS;
5. `Run_BO.py`: The python file for running the Bayesian optimization;
6. `general_functions.py`: The python file with functions to implement some general usage, e.g. extract data from the .txt files, and run ABAQUS locally or on the cluster;
7. `target_data.txt`: The target force-displacement data.
## Installation
* Install the `ABAQUS 2021` or higher versions.
* Install the `gpflow` package for Python to run the Bayesian optimization.
## Getting started
* Determine a target force-displacement curve from ABAQUS or by user definition and put it into the file `target_data.txt`.
* Make sure all the `.py` and `.inp` files are put in the same directory.
* If running the ABAQUS locally, open the `general_functions.py` and modify the path of ABAQUS in the function `run_abaqus_locally`.
* Open `Run_BO.py`, and modify the path of files if needed.
1. `pickle_filename`: The path of file storing the optimization data;
2.  `tar_txt_name`: The name of the file storing the target force-displacement curve, which will be converted into stress-strain data;
3. `db_directory`: The directory storing the initial database for the Bayesian optimization;
4. `db_csv_name`: The name of the .csv file storing the design parameters corresponding to the initial database;
5. `local_directory`: The path going to run the code of finite element simulations and Bayesian optimization;
6. `abaqus_script`: The .py file run as a script in ABAQUS;
7. `inp_file_name`: The name of the .inp file which will be used for modeling in ABAQUS.
* If running the ABAQUS on the cluster, change the function of `run_abaqus_locally` (line 175) into `run_abaqus`.
* Run `Run_BO.py` and begin the calculation of the Bayesian optimization.
