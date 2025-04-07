import subprocess
import torch
from torch.utils.data import DataLoader


# Function to call the training script from command line
def train_model(params):
    """ Call the train.py script with the given parameters """
    
    # Extract parameters from the tuning configuration
    num_epochs = params["num_epochs"]
    lr = params["lr"]
    lambda_cycle = params["lambda_cycle"]
    lambda_identity = params["lambda_identity"]
    lambda_style = params["lambda_style"]
    save_model = params["save_model"]
    log = params["log"]
    name = params["name"]
    
    # Command to run the training script
    command = [
        "python", "train.py", 
        "--num_epochs", str(num_epochs),
        "--lr", str(lr),
        "--lambda_cycle", str(lambda_cycle),
        "--lambda_identity", str(lambda_identity),
        "--lambda_style", str(lambda_style),
        "--save_model", str(save_model),
        "--log", str(log),
        "--name", name
    ]
    
    # Run the training script
    return subprocess.run(command)
    


# Tuning configurations
hyperparameters = [
    {
        "num_epochs": 20,
        "lr": 0.0001,
        "lambda_cycle": 10.0,
        "lambda_identity": 0.5,
        "save_model": True,
        "log": True,
        "name": "config_1"
    },
    {
        "num_epochs": 40,
        "lr": 0.0005,
        "lambda_cycle": 15.0,
        "lambda_identity": 1.0,
        "save_model": True,
        "log": True,
        "name": "config_2"
    },
    {
        "num_epochs": 60,
        "lr": 0.00005,
        "lambda_cycle": 5.0,
        "lambda_identity": 0.1,
        "save_model": True,
        "log": True,
        "name": "config_3"
    }
]


# Run the experiments
if __name__ == "__main__":
    train_model(hyperparameters)
