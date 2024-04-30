from components.upload_media_component import UploadJPGComponent, UploadPNGComponent
from threading import Lock
from abc import ABC, abstractmethod
import streamlit as st

from utils.image_transform import to_tensor, inverse_depth_norm
from utils.get_model import get_model
from torchvision import transforms


class InferMediaComponent(ABC):
    _model = None
    _lock: Lock = Lock()

    def __init__(self):
        with InferMediaComponent._lock:
            if not InferMediaComponent._model:
                InferMediaComponent._model = get_model(state_path="model/trained_model/leve.pth")
                print("created model...")

    @abstractmethod
    def factory(self):
        pass

    def build(self):
        canvas = self.factory()
        canvas.build()
        media_frames = canvas.get_media_frames()

        if media_frames:
            with st.spinner("We appreciate your patience...😅"):
                model = InferMediaComponent._model

                infered_frames = []
                for media_frame in media_frames:
                    frame = self._prepare_image(media_frame)
                    model.eval()
                    output = model(frame)
                    real_output = inverse_depth_norm(output)
                    real_output = transforms.Resize(media_frame.shape[:2])(real_output)
                    infered_frames.append(real_output)

                canvas.display(infered_frames)

    def _prepare_image(self, image):
        image = to_tensor(image)
        image = transforms.Resize((480, 640))(image.unsqueeze(0))
        return image


class InferJPGComponent(InferMediaComponent):
    def __init__(self):
        super().__init__()

    def factory(self):
        return UploadJPGComponent()


class InferPNGComponent(InferMediaComponent):
    def __init__(self):
        super().__init__()

    def factory(self):
        return UploadPNGComponent()
