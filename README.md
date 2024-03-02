![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![license MIT](https://img.shields.io/badge/licence-MIT-green)

![DreamerV3 Logo](misc/img/logo.png)

# Ray DreamerV3 Project Template

This template is designed for experimenting with the Ray DreamerV3 algorithm, specifically tailored for use with custom environments. It provides a structured, modular setup for training Reinforcement Learning (RL) models using the Ray RLlib library, focusing on the DreamerV3 algorithm and its integration with custom environments compliant with the OpenAI Gym API.

## Project Structure:

```plaintext
.
├── LICENSE
├── README.md
├── activate_env.sh
├── build.sh
├── docker
│   ├── Dockerfile
│   └── requirements.txt
├── poetry.lock
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── abstract
│   │   ├── __init__.py
│   │   ├── base_construct.py
│   │   └── registration.py
│   ├── common
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   └── methods.py
│   ├── conf
│   │   ├── trainable
│   │   │   ├── dreamerv3-xs.yaml
│   │   │   └── ppo-256-256-no-attention.yaml
│   │   └── trial.yaml
│   ├── construct
│   │   ├── __init__.py
│   │   ├── dreamerv3.py
│   │   └── ppo.py
│   ├── delegator
│   │   ├── __init__.py
│   │   ├── trainable.py
│   │   └── tuner.py
│   ├── environment
│   │   ├── __init__.py
│   │   └── corridor.py
│   └── tune.py
└── tests
    └── integration
        ├── __init__.py
        └── test_dreamerv3.py
```

## Key Features

- **Custom Environment Integration**: This template is tailored for the DreamerV3 algorithm, emphasizing effortless integration with custom environments.
- **Modular Algorithm Configuration**: DreamerV3 and other reinforcement learning (RL) algorithms are designed as distinct constructs, enhancing modularity and code reuse. You can find and extend these constructs within the `src/construct` directory.
- **Nested Hyperparameter Configuration**: Configure and fine-tune the hyperparameters for DreamerV3 and other algorithms using YAML files in `src/conf/trainable`. Create distinct configuration files for different training scenarios.
- **Training Control and Hyperparameter Optimization**: Utilize the main configuration file and integrate with Ray Tune to establish stopping criteria, checkpointing strategies, and hyperparameter tuning.
- **Registry-Based Design**: The template employs a registry-based approach for algorithm constructs, supporting easy experimentation and modular design. Simply register new algorithm constructs to facilitate their usage.

## Getting Started

1. **Activate Environment**: Initialize and activate the project environment using the command `source activate_env.sh`.
2. **Create/Modify Configuration Files**: Adjust the training parameters by providing YAML configuration files in `src/conf`. Specify the RL algorithm construct and its hyperparameters.
3. **Run Training**: Start the training process by executing `src/tune.py`. This script will interpret the configuration files, instantiate the necessary delegator classes with the provided settings, and kick off the RL training.
4. **Extend with New Algorithms**: To incorporate new RL algorithms, register their constructs in the relevant registry and adhere to the naming and structural conventions. Ensure that the new construct is placed in `src/construct` and implements the required class methods.

This template serves as a robust and scalable foundation for conducting RL research with the DreamerV3 algorithm and custom environments using the Ray RLlib library. Enjoy your experimentation journey!
