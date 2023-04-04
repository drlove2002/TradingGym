# Stock Market Analysis Gym Environment with Reinforcement Learning

This repository provides a Gym environment for analyzing the stock market using reinforcement learning techniques. The environment is compatible with MuZero and enables researchers and practitioners to train and evaluate RL agents in simulated stock market scenarios.

## Installation
To install the `TradingGym` package from this GitHub repository, run the following command:
```bash
pip install git+https://github.com/drlove2002/TradingGym.git
```

## Usage
To use the `TradingGym`, import it into your Python code as follows:
```python
import gymnasium as gym
import trading_gym as tg

env = gym.make('stocks-v0')
```

For more information on how to use Gym environments, see the [Gym documentation](https://www.gymlibrary.dev/).

## Example
To see an example of how to use the `TradingGym` package, check out the [test_render.ipynb](https://github.com/drlove2002/TradingGym/blob/master/tests/test_render.ipynb) notebook in the `tests` folder. This notebook demonstrates how to create and render a simple stock market environment using the TradingGym package.

You can run the notebook locally by cloning the TradingGym repository and installing the necessary dependencies. For example:
```bash
git clone https://github.com/drlove2002/TradingGym.git
cd TradingGym
pip install -r requirements.txt
jupyter notebook tests/test_render.ipynb
```
This will open the notebook in your browser, where you can run each cell to see the environment in action.

## Compatibility
This Gym environment is compatible with MuZero, an RL algorithm that can learn to play complex games and solve planning problems without knowing the rules in advance. For more information on MuZero, see the [original paper](https://arxiv.org/abs/1911.08265).

## Contributing
Contributions to this repository are welcome! If you find a bug or would like to suggest an enhancement, please open an issue or submit a pull request.

Before contributing, please read the [contributing guidelines](CONTRIBUTING.md) for more information on how to contribute to this project.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
