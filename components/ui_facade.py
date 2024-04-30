from components.templated_pages import LevePage, HomePage, AboutPage


class UIFacade:
    def __init__(self, page_name):
        self.page_name = page_name

    def render_page(self):
        if self.page_name == "home":
            home = HomePage()
            home.render_template()
        elif self.page_name == "about":
            about = AboutPage()
            about.render_template()
        elif self.page_name == "leve":
            leve = LevePage()
            leve.render_template()
