import torch
from utils.get_model import get_model
from utils.prepare_image import prepare_image
from utils.utils import inverse_depth_norm

# load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_path = "model/trained_model/leve.pth"
model = get_model(state_path=state_path).to(device)


def infer(image):
    image = prepare_image(image)
    # perform inference
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        real_output = inverse_depth_norm(output)
    return real_output
