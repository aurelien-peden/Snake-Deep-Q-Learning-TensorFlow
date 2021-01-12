# Snake-Deep-Q-Learning-TensorFlow

Deep Q-Learning implemented using TensorFlow 2.0 on a custom Snake game environment.

[](https://media.giphy.com/media/w038mWYadaQgpBFxMq/giphy.gif)

## Project's structure

* [game.py](./game.py): Contains the game logic and then environment.

* [agent.py](./agent.py): Contains the agent that will use, train the model, estimate the best actions to perform and play the game.

* [model.py](./model.py): Contains the model used by the Agent.

## How to run the project

In addition to python 3.8.5, you will need to install the following packages:

* Pygame 2.0

* TensorFlow 2.3.1

* Numpy

Once the previous packages are installed, all you need to do is to run the following command:

```
python agent.py
```

## Possible improvements

The agent manages to play on its own decently, but there is room for improvements, here are the following things that could be implemented:

* Building a CNN and feeding it with a stack of low-resolution images representing the current game state. Stacked frames are required so the CNN can get an overview of the current snake's direction.

* Implementing an Actor-Critic algorithm. An Actor-Critic agent consists of two neural networks, one is the DQN (critic) trained using the agent's experiences, the other one is a policy net (actor), its role is to approximate the policy by relying on the action values estimated by the DQN. See the following paper for more information: [A survey of actor-critic reinforcement learning: standard and natural policy gradients](https://hal.archives-ouvertes.fr/hal-00756747/document).

* Tuning the agent and model parameters: epsilon decay, batch size...

* Changing the rewards. The current rewards are the following:
  
  * +1 each step the snake get closer to the apple.
  
  * -1 each steps the snake get further from the apple.
  
  * -100 if the snake collide with itself or with a wall.
  
  * +10 for eating the apple.
