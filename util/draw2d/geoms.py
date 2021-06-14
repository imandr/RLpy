import math
from .attrs import Color, LineWidth
from .transform import Transform

RAD2DEG = 180.0/math.pi



try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

class Geom(object):
    def __init__(self, transform=None, hidden=False):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
        self.transform = transform
        self.hidden = hidden
        
    def hide(self):
        self.hidden = True
        return self
    
    def show(self):
        self.hidden = False
        return self
    
    def render(self, transforms=[], enable_transforms=True):
        if not self.hidden:
            for attr in reversed(self.attrs):
                #print("Geom.render: attr:", attr)
                attr.enable()
            t = transforms
            if self.transform is not None:
                if enable_transforms:
                    self.transform.enable()
                t = transforms + [self.transform]
                
            self.render1(t)

            if self.transform is not None and enable_transforms:
                self.transform.disable()
            for attr in self.attrs:
                attr.disable()
                
    def render1(self, transforms):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)
        return self
        
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)
        return self
    color = set_color

    def line_width(self, w):
        return self.add_attr(LineWidth(w))
        
    def transform(self, **args):
        self.transform = Transform(**args)
        return self
        
    def move_to(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.move_to(*params)
        return self

    def move_by(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.move_by(*params)
        return self

    def rotate_by(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.rotate_by(*params)
        return self

    def rotate_to(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.rotate_to(*params)
        return self
        
    def scale_to(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.scale_to(*params)
        return self

    def scale_by(self, *params):
        if self.transform is None:  self.transform = Transform()
        self.transform.scale_by(*params)
        return self
        
class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
        
    def render1(self, transforms):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
        
    def render1(self, transforms):
        #print("rendeing FilledPolygon:", self.v)
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
        
class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self, transforms):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self, transforms):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self, transforms):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

class Text(Geom):
    def __init__(self, text="", anchor_x="right", anchor_y="bottom",
                    size=12, font="Arial", rotation="inherit", color=(1.0,1.0,1.0)):
        Geom.__init__(self)
        self.Text = text
        self.AnchorX = anchor_x
        self.AnchorY = anchor_y
        self.Size = size
        self.Font = font
        self.Rotation = rotation
        if len(color) == 3:
            color = color + (1.0,)
        self.Color = [int(c*255.0) for c in color]
        
    def render1(self, transforms):
        from pyglet.text import Label
        x, y, a = 0.0, 0.0, 0.0
        #print("Text.render1: input ",x,y,a)
        for t in reversed(transforms):
            #print(t)
            x, y, a = t(x, y, a)
        #print ("Text.render1: output", x, y, a)
        if self.Rotation != "inherit":
            a = self.Rotation

        glPushMatrix()
        glLoadIdentity()
        glTranslatef(x, y, 0) # translate to GL loc point
        glRotatef(RAD2DEG * a, 0, 0, 1.0)

        Label(self.Text, font_name=self.Font, font_size=self.Size,
             anchor_x = self.AnchorX, anchor_y = self.AnchorY, color=self.Color,
             x = 0, y = 0
        ).draw()
        
        glPopMatrix()
        
class Marker(Geom):
    def __init__(self, geom, rotation="inherit"):
        Geom.__init__(self)
        self.Rotation = rotation
        self.Geom = geom
        
    def render1(self, transforms):
        # build transform from all stacked transforms, ignoring scale
        #print("Marker.render: calculating transforms...")
        #x, y, a = self.transform.apply(0.0, 0.0, 0.0)
        x, y, a = 0.0, 0.0, 0.0
        for t in reversed(transforms):
            #print("Marker.render: t:", t)
            x1, y1, a1 = t(x, y, a)
            #print("                 ", x, y, a, "->", x1, y1, a1)
            x, y, a = x1, y1, a1
        
        if self.Rotation != "inherit":  a = self.Rotation
        
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(x, y, 0) # translate to GL loc point
        glRotatef(RAD2DEG * a, 0, 0, 1.0)

        self.Geom.render(enable_transforms=False)
        
        glPopMatrix()

def Circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)
        
def Polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def Rectangle(left, right, top, bottom, filled=True):
    return Polygon([
        (left, bottom), (right, bottom), (right, top), (left, top)
    ], filled=filled)
    

    