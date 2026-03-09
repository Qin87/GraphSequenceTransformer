import torch.nn as nn
import torch.nn.functional as F
import yaml


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "best_hyperparameters.yml"
    # print(os.getcwd())
    # # os.chdir("..")      # Qin
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")
    # print(args)
    return args
