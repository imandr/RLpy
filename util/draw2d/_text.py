from .geoms import Geom
from pyglet.text import Label

class Text(Geom):
    def __init__(self, text, x, y, anchor_x="right", anchor_y="bottom",
                    size=12, font="Arial"):
        Geom.__init__(self)
        self.Text = text
        self.AnchorX = anchor_x
        self.AnchorY = anchor_y
        self.Size = size
        self.Font = font
        self.X = x
        self.Y = y
        
    def render1(self):
        Label(self.Text, font_name=self.Font, font_size=self.Size,
             anchor_x = self.AnchorX, anchor_y = self.AnchorY,
             x = self.X, y = self.Y
        ).draw()
        