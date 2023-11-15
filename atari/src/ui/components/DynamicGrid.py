from kivy.uix.gridlayout import GridLayout
from kivymd.uix.gridlayout import MDGridLayout


class DynamicGrid(MDGridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols_minimum = {i: 0 for i in range(self.cols)}
        self.bind(minimum_width=self.update_cols_minimum)

    def update_cols_minimum(self, *args):
        for i in range(self.cols):
            self.cols_minimum[i] = 0
        for widget in self.walk():
            if isinstance(widget, GridLayout):
                for i, col_width in enumerate(widget.cols_minimum.values()):
                    self.cols_minimum[i] = max(self.cols_minimum[i], col_width)
        self.cols_minimum = self.cols_minimum
