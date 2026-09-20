# BENV0119_CW

## Task 3: Reinforcement Learning

### 3.1) Problem Statement

The aim of task 3 is for you to demonstrate your understanding of Reinforcement Learning through the undertaking of an example exercise. You’ll be working with the simulation environment *[pymgrid](https://github.com/Total-RD/pymgrid)*, solving the problem proposed by *[Wenuka](https://github.com/Wenuka/RL_for_energy_tutorial)*. Reinforcement learning will be used to manage the energy flow of a building powered by solar panel system with a battery storage as shown in _Figure 1_. When solar PV generation is not sufficient to cater to the demand, the main grid supplies the mismatch at a varying (with time) price. Further the cost of CO<sub>2</sub> produced at a specific time by the grid is also considered for the cost. Your task is to charge or discharge the battery for the mismatch accordingly so that the overall cost is minimized using Q-learning algorithm introduced in class with fine tuned hyperparameters.

<div align="center">
    <img src="https://github.com/Ziyan0825/BENV0019_Coursework/blob/main/figures/Environment.png" width="500" height="350">
</div>

<div align="center">
<em>Figure 1: Set up of building energy management system</em>
</div>

**The main objectives of this task are as follows:**

*-Show understanding of Reinforcement learning and how it can be applied to smart buildings.*  

*-Implement the Q-learning algorithm through the simulation environment [pymgrid](https://github.com/Total-RD/pymgrid).*

*-Investigate the hyperparameters within the Q-learning algorithm to facilitate more effective learning of the agent.*

### 3.2) Required Outputs

This task requires the following outputs to in your Jupyter Notebook:

*-Successful implementation of Q-learning following **pilot_codes**.*

*-Identify the set of hyperparameters that minimize the the overall cost.*

*-Answer the question to show your understanding of both project background and Reinforcement Learning.*
