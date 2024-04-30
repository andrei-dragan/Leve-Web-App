import streamlit as st
from abc import ABC, abstractmethod


class UploadMediaComponent(ABC):
    def __init__(self):
        self.content = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def get_media_frames(self):
        pass

    @abstractmethod
    def display(self, infered_frames):
        pass


class UploadJPGComponent(UploadMediaComponent):
    def __init__(self):
        super().__init__()
        self.image = None

    def build(self):
        self.image = st.file_uploader("✨Upload a JPG image and see the magic happen!✨", type=["jpg"])

    def get_media_frames(self):
        from PIL import Image
        import numpy as np
        import os

        frames = []

        if self.image is not None:
            with st.spinner("Hold on a sec...processing your image!🧐"):
                image_name = self.image.name
                with open(image_name, "wb") as f:
                    f.write(self.image.read())
                image = np.array(Image.open(image_name))
                frames = [image]
                os.remove(image_name)

        return frames

    def display(self, infered_frames):
        from matplotlib import pyplot as plt
        import numpy as np
        from io import BytesIO

        buffer = BytesIO()

        for infered_frame in infered_frames:
            infered_frame = infered_frame.detach().cpu().numpy()
            plt.figure()
            plt.imshow(np.squeeze(infered_frame))
            plt.axis("off")
            plt.savefig(buffer, format="jpg")
            plt.close()

        buffer.seek(0)
        st.image(buffer, use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>Tha-daa🎉</h4>", unsafe_allow_html=True)


class UploadPNGComponent(UploadMediaComponent):
    def __init__(self):
        super().__init__()
        self.image = None

    def build(self):
        self.image = st.file_uploader("⭐Prepare to be amazed!⭐", type=["png"])

    def get_media_frames(self):
        from PIL import Image
        import numpy as np
        import os

        frames = []

        if self.image is not None:
            with st.spinner("Hold on a sec...processing your image!🧐"):
                image_name = self.image.name
                with open(image_name, "wb") as f:
                    f.write(self.image.read())

                # transform png to jpg
                image = Image.open(image_name)
                image = image.convert("RGB")
                image = np.array(image)
                frames = [image]
                os.remove(image_name)

        return frames

    def display(self, infered_frames):
        from matplotlib import pyplot as plt
        import numpy as np
        from io import BytesIO

        buffer = BytesIO()

        for infered_frame in infered_frames:
            infered_frame = infered_frame.detach().cpu().numpy()
            plt.figure()
            plt.imshow(np.squeeze(infered_frame))
            plt.axis("off")
            plt.savefig(buffer, format="jpg")
            plt.close()

        buffer.seek(0)
        st.image(buffer, use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>Tha-daa🎉</h4>", unsafe_allow_html=True)
