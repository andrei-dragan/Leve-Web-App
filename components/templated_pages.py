from abc import ABC, abstractmethod
import streamlit as st


class PageTemplate(ABC):
    def __init__(self):
        pass

    def render_template(self):
        self.set_generic_header()
        self.set_body()

    def set_generic_header(self):
        st.markdown("☀️ **Leve - the lightweight depth estimation model**")
        st.divider()

    @abstractmethod
    def set_body(self):
        pass


class HomePage(PageTemplate):
    def __init__(self):
        super().__init__()

    def set_body(self):
        st.title("Welcome to Leve! :wave:")
        st.markdown("From latin: *levis/leve, levis M [adj] - simple, light*")
        st.write("#### Leve is a lightweight depth estimation model that can be used to estimate depth from images.")
        st.write("######  You can find more about us by navigating through the sidebar. Or if you want to directly try Leve, click the button below.")
        st.link_button("Get Started", "Leve")


class LevePage(PageTemplate):
    def __init__(self):
        super().__init__()

    def set_body(self):
        from components.infer_media_component import InferJPGComponent, InferPNGComponent

        st.write("#### What type of media would you like to check depth for?")
        option = st.selectbox("#", ("JPG Image", "PNG Image"))
        if option == "JPG Image":
            image_media_component = InferJPGComponent()
            image_media_component.build()
        elif option == "PNG Image":
            video_media_component = InferPNGComponent()
            video_media_component.build()
