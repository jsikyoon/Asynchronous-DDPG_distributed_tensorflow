Asynchronous-DDPG_distributed_tensorflow
===========

Distributed Tensorflow Implementation of asynchronous ddpg.

Implementation is on Tensorflow 1.2.1.

DDPG script is based on songrotek's repo. https://github.com/songrotek/DDPG.git

One of popular pain-points of reinforcement learning is too long learning time. Thus, A3C was proposed for parallel learning to efficiently learn the agent. However, for DDPG, one of strong alogrithm for continuous action episode, there are a few research for parallel learning. One of them is intentional unintentional agent, which is to learn several tasks simultaneously (https://arxiv.org/abs/1707.03300). In here, I validate parallel learning of ddpg for simpler experiment than IU agent's one. Each workers learn just one task. After learning several episodes, their training information is merged with parameter server.   

GYM Reacher-v1 game
-------------------

`
./auto_run.sh 
`

You need to set your hostname and port number in gym_addpg.py code. The number of parameter servers and workers can be set in auto_run.sh script file.

### Settings

Almost Settings are same to songrotek's ones, except learning rate of critic networks.
The number of parameter server and workers is 1 and 4, repectively.

### Results

![alt tag](https://github.com/jaesik817/Asynchronous-DDPG_distributed_tensorflow/blob/master/figures/addpg_res.PNG)


