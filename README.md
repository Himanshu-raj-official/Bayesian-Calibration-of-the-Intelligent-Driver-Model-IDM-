Bayesian Calibration of the Intelligent Driver Model (IDM)
Calibrates the Intelligent Driver Model (IDM), a 5-parameter car-following model, against real highway vehicle trajectory data using both classical least-squares and Bayesian inference.

____________________________________________________________________________________________________________________________________________________________________________________________


Table of Contents
Description
Key Finding
Built With
Installation
How to run the project
Outputs
Learning Outcomes
Project Structure
License

____________________________________________________________________________________________________________________________________________________________________________________________


Description
This project estimates the five IDM driving behavior parameters (v0, T, a_max, b, and s0) using real vehicle-following data from the NGSIM highway dataset. It compares two methods: the traditional SciPy least-squares approach and a Bayesian approach using PyMC. The main finding is that, in free-flow traffic, the traditional method always gives almost the same unrealistic value for s0, even when started from different initial values. This suggests that s0 cannot be estimated reliably from this type of data. In contrast, the Bayesian method does not hide this uncertainty. Instead of giving one potentially misleading value, it shows a range of possible values, making the uncertainty clear.

____________________________________________________________________________________________________________________________________________________________________________________________


Key Finding

Regime	Parameter	Classical LSQ (8 restarts)	Bayesian posterior (mean ± sd)
Free-flow	s0	0.050 m (identical every restart)	2.51 ± 1.55 m
Congested	v0	40.6 – 45.0 m/s (unstable)	26.8 ± 8.6 m/s
Each traffic regime is blind to the parameters the other regime is good at recovering — free-flow data can't pin down minimum-gap behavior, congested data can't pin down desired free-flow speed.

____________________________________________________________________________________________________________________________________________________________________________________________


Built With
Python 3.13.14
PyMC — Bayesian modeling & NUTS sampling
ArviZ — posterior diagnostics & visualization
SciPy — classical least-squares optimization
NumPy / pandas — numerical computing & data handling
Matplotlib — plotting & animation
Jupyter — notebooks

____________________________________________________________________________________________________________________________________________________________________________________________


Installation
Step 1: Install Anaconda

Step 2: Download the project
	Download the project from GitHub or clone the repository.

Step 3: Open Anaconda Prompt

Step 4: Navigate to the Project folder
	---bash
	cd "File location"

Step 5: Create an Environment
	Environment name - idm

Step 6: Install following modules
	---bash
	a) conda install -c conda-forge pymc arviz -y
	b) pip install pandas matplotlib scipy jupyter notebook
	c) python scripts\run_full_pipeline.py --quick
	d) conda install -c conda-forge m2w64-toolchain -y
	e) conda install -c conda-forge pymc arviz m2w64-toolchain -y
	f) conda install -c conda-forge pymc pytensor --force-reinstall
	g) pip install -U --force-reinstall pymc

Step 7: Launch Jupyter Notebook
	---bash
	jupyter notebook
	"Your default web browser will be open automatically."

Full analysis on real NGSIM data:
Download vehicle trajectory data from data.transportation.gov — search "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data" and export as CSV.
Place the CSV in (data/raw/)
or you can load and filter to your locations of interest.

____________________________________________________________________________________________________________________________________________________________________________________________

How to run the Project
Step 1: Open Notebooks
	run each cell of idm_bayesian_calibration.ipynb
Step 2: open outputs
	├── figures/
	│	├── s0_real_data_comparison.png
        │       ├── idm_simulation.gif
	│
        └── results/
		├── cong_flow_bayesian_calibration.csv
		├── free_flow_bayesian_calibration.csv
		├── real_data_s0_comparison.csv

____________________________________________________________________________________________________________________________________________________________________________________________
	
Outputs
Leader–Follower Vehicle Pair Selection
Vehicle Trajectory Visualization
Least Squares (LSQ) Calibration Results
Bayesian Calibration Results
Comparison Between Free-Flow and Congested Driving
CSV Output Files

____________________________________________________________________________________________________________________________________________________________________________________________


Learning Outcomes
Intelligent Driver Model (IDM)
Data Processing
Least Squares Optimization
Bayesian Inference

____________________________________________________________________________________________________________________________________________________________________________________________


Project Structure

bayesian_idm_calibration/
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── raw/NGSIM_Data_2026.csv
├── src/
│   ├── idm_simulator.py         <- IDM physics: forward sim + one-step prediction
│   ├── data_loader.py           <- NGSIM loader + synthetic data generator
│   ├── classical_calibration.py <- scipy least-squares baseline
│   ├── bayesian_calibration.py  <- PyMC single-vehicle + hierarchical models
│   └── utils.py                 <- metrics & plotting helpers
├── notebooks/
│   └── idm_animation_real_data.py
└── outputs/
    ├── figures/
    └── results/

____________________________________________________________________________________________________________________________________________________________________________________________

License
This project is released under the MIT License.
You are free to use, modify, and distribute this work with proper attribution.
See the `LICENSE` file for complete license information.
____________________________________________________________________________________________________________________________________________________________________________________________
