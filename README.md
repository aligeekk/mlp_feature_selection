# Implementation
Neural Network to predict daily directional movement of ASX200.
The input vector for the final network is selected based on neuron relevance or a genetic algorithm.

### Prerequisites
Please use Python3 or high when using this software.
To run this software you will need to install the torch, numpy and matplotlib packages.
These packages can be installed using the following commands:
	* pip3 install torch
	* pip3 install numpy
	* pip3 install matplotlib 

### Using the software
If you require a network generated using the genetic algorithm:
	python3 src/main_GA.py
If you require a network generated neuron relevance:
	python3 src/main_relevance.py

Note, the required data files must be located in src/data/

## Authors

* **Darren Lawton**

