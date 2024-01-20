# Basic Ray RLlib Project Skeleton

This repository provides a structured, modular setup for training Reinforcement Learning (RL) models using the Ray RLlib library. It accommodates environments compliant with the OpenAI Gym API, enabling you to work with a vast array of environments out-of-the-box. The project centers on a registry-based design, allowing extensibility, and an elegant plug-and-play style configuration for RL training setups.

## Project Structure:

```
.
├── basic_project
│   ├── common                # Module containing commonly used constants and methods
│   ├── config                # Module containing configuration files (main config and trainable configurations)
│   ├── environment           # Module containing custom environment implementation along with creator function
│   ├── delegator             # Main module containing classes responsible for delegating tasks
│   │   ├── abstract          # Submodule containing abstract base classes
│   │   ├── construct         # Submodule containing different RL algorithm constructs
│   ├── tune.py               # Main script to start the training process
```

## Key Features:

- **Environment Activation:** Activate the working environment using a shell script. This script sets up and triggers the environment required to run the project.

- **OpenAI Gym Environment Compatibility:** The framework accommodates environments adhering to the OpenAI Gym API.

- **Modular RL Algorithm Configuration:** Specify the RL algorithm for training as a separate construct. This approach encourages modularity and code reuse. Constructs for RL algorithms can be found in `delegator/construct` and can be easily extended with additional algorithms.

- **Nested Hyperparameter Configuration:** Define and adjust the hyperparameters for the RL algorithm via the YAML configuration files located in the `config` module. Different configuration files can be created for various training scenarios.

- **Training Control:** Define stopping conditions and checkpointing policies for the training process through the main configuration file.

- **Hyperparameter Optimization:** Integration with Ray Tune for hyperparameter optimization of RL models.

- **Registry-Based Design:** The registry-based design handles multiple training algorithms, promoting extensibility and a modular design. New algorithm constructs are registered in the `ConstructRegistry`.

## Getting Started:

1. **Activate Environment:** Use the following command to set up and activate the necessary environment for running the project:

`source activate_env.sh`

2. **Create/Modify Configuration Files:** Provide a YAML configuration file detailing the training parameters under `config`. This includes specifying the RL algorithm construct to use and its associated hyperparameters.

3. **Run Training:** Execute the `tune.py` script. The script reads the configuration files, initializes the appropriate delegator classes with the loaded configuration, applies the parameters, and commences the RL training process. 

4. **Extend with New Algorithms:** When adding new RL algorithms to the project, ensure to register the new constructs in the `ConstructRegistry` and follow the established conventions for naming and file structure. The new construct should implement the 'from_construct_registry_directive' class method. The constructs are Python classes representing RL algorithms, located in the `delegator/construct` directory.

This project provides a flexible and extendable base for RL experimentation with the Ray RLlib library. Happy experimenting!
