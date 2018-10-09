[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

**Environment solved criterion:** The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
The project has been tested on Linux only.

#### Download the environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

2. Place the file in the `p1_navigation/` source folder, and unzip (or decompress) the file (the repo already contains extracted environment for Linux).

#### Install all the dependencies
1. Install virtualenv
```
sudo apt install virtualenv
```
2. Create a virtualenv for python3
```
virtualenv -p python3 drlnd
```
3. Activate the environment
```
source drlnd/bin/activate
```
4. Install all the dependencies
```
cd p1_navigation/
pip3 install python/
```
Note that the setup assumes that Tkinter is installed for python3. If not, install using the following
```
sudo apt install python3-tk
```
