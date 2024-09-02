# Descriptions
This repository includes the python sccripts that used for numerical expirements in Chapter 1:Priors' Impact on Thompson Sampling for Dynamic Pricing of Multiple Substitutable Products.
1. mnl_model.py implements a Multinomial Logistic Demand Model using PyTorch, designed for modeling and estimating demand in pricing problems. provides methods for calculating choice probabilities, mean demand, and profit of products at specific prices, as well as methods for maximum likelihood estimation (MLE) and Langevin dynamics for parameter estimation.
2. mnl_env.py provides an environment for simulating experiments based on the Multinomial Logistic Model (MNL). This environment is designed to interact with pricing algorithms, allowing users to test and evaluate various pricing strategies under uncertain demand conditions. The environment uses the MultipleMNLModel class to model the demand and simulate user choices based on prices set by the algorithm.
3. The rest scripts simulate a dynamic pricing experiment using the Multinomial Logistic Model (MNL) within an environment (MNLEnvironment) and evaluates the performance of the Thompson Sampling with Lapalce approximation and Thompson sampling with Langevin dynamics under three differebt priors:<br>
   (1). MLE estimated(well-specified prior), where I marked as "_random_explore"<br>
   (2). Uniform distribution(non-informative prior)<br>
   (3). Normal distribution (mis-specified prior)<br>
These scripts utilize MPI for parallel execution, allowing the experiment to be distributed across multiple processors. and the current version use the Synthetic data from Section 1.5.2 with batchsize =200. For coffee case, data in the 'coffee_data' file should be import and the relvant global parameters should be changed as well.<br>
# Prerequisites
This repo is tested with Python 3.10.6 mpirun (Open MPI) 4.1.2 and PyTorch 2.3.1+cu118 in Ubuntu 22.04.1 environment. 
