from abc import ABC, abstractmethod
import streamlit as st


class PageTemplate(ABC):
    def __init__(self):
        pass

    def render_template(self):
        self.set_generic_header()
        self.set_body()

    def set_generic_header(self):
        st.markdown("🦉 **Leve - our lightweight depth estimation model**")
        st.divider()

    @abstractmethod
    def set_body(self):
        pass


class HomePage(PageTemplate):
    def __init__(self):
        super().__init__()

    def set_body(self):
        st.title("Welcome to Leve - our Lightweight Depth Estimation Model")
        st.write("This app is a simple demonstration of depth estimation using our Leve pre-trained model.")


class AboutPage(PageTemplate):
    def __init__(self):
        super().__init__()

    def set_body(self):
        st.title("About Leve")
        st.write("...")


class LevePage(PageTemplate):
    def __init__(self):
        super().__init__()

    def set_body(self):
        from components.infer_media_component import InferJPGComponent, InferPNGComponent

        option = st.selectbox("What type of media would you like to check depth for?", ("JPG Image", "PNG Image"))
        if option == "JPG Image":
            image_media_component = InferJPGComponent()
            image_media_component.build()
        elif option == "PNG Image":
            video_media_component = InferPNGComponent()
            video_media_component.build()
