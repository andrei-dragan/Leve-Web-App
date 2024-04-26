import torch
from model.leve import Leve


def get_model(state_path):
    model = Leve()
    if state_path != "":
        model_state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(model_state_dict)
    return model
