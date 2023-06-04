from .attrs import Attr
import math

try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

if True:
    try:
        from pyglet.gl import *
    except ImportError as e:
        raise ImportError('''
        Error occurred while running `from pyglet.gl import *`
        HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
        If you're running on a server, you may need a virtual frame buffer; something like this should work:
        'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
        ''')

RAD2DEG = 180.0/math.pi

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
        
    def __str__(self):
        return "<Transform t:%s r:%s, s:%s>" % (self.translation, self.rotation, self.scale)
    
    __repr__ = __str__
        
    @staticmethod
    def zero():
        return Transform()
        
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
        #print("Transform enabled:", self)
        
        
    def disable(self):
        #print("%s: disable" % (self,))
        glPopMatrix()
        #print("Transform disabled:", self)

    __enter__ = enable
    
    def __exit__(self, *params):
        self.disable()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
        return self
        
    move_to = set_translation
    def move_by(self, deltax, deltay):
        return self.move_to(self.translation[0]+deltax, self.translation[1]+deltay)

    def set_rotation(self, new):
        self.rotation = float(new)
        return self
        
    rotate_to = set_rotation
    def rotate_by(self, delta):
        return self.rotate_to(self.rotation+delta)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))
        return self

    scale_to = set_scale
    def scale_by(self, deltax, deltay):
        return self.scale_to(self.scale[0]*deltax, self.scale[1]*deltay)

    def __add__(self, other):
        return Transform(
            translation = (self.translation[0] + other.translation[0], self.translation[1] + other.translation[1]),
            rotation = self.rotation + other.rotation,
            scale = (self.scale[0]*other.scale[0], self.scale[1]*other.scale[1])
        )
        
    def apply(self, x, y, a):
        x *= self.scale[0]
        y *= self.scale[1]
        s,c = math.sin(self.rotation), math.cos(self.rotation)
        x, y = x*c - y*s, y*c + x*s
        x += self.translation[0]
        y += self.translation[1]
        a += self.rotation
        return x, y, a
        
    __call__ = apply
    
class Frame(object):
    def __init__(self, geoms=[], transform=None, hidden=False, **args):
        self.NamedGeoms = {}
        self.UnnamedGeoms = []
        self.transform = transform if transform is not None else Transform(**args)
        self.hidden = hidden
        
    def __str__(self):
        return "<Frame: %s>" % (repr(self.transform),)
        
    __repr__ = __str__
    
    def translate(self, x, y, a):
        return self.transform.apply(x, y, a)

    def hide(self):
        self.hidden = True
        return self
    
    def show(self):
        self.hidden = False
        return self
    
    def add(self, g, label=None, at=None, rotation=None, scale=None):
        if label is None:
            self.UnnamedGeoms.append(g)
        else:
            self.NamedGeoms[label] = g
        if at is not None:  g.move_to(*at)
        if rotation is not None:    g.rotate_to(rotation)
        if scale is not None:   g.scale_to(*scale)
        return self
            
    def __getitem__(self, label):
        return self.NamedGeoms[label]
        
    def render(self, transforms=[]):
        if not self.hidden:
            with self.transform:
                for g in self.UnnamedGeoms + list(self.NamedGeoms.values()):
                    g.render(transforms+[self.transform])
                        
    def remove_all(self):
        self.NamedGeoms = {}
        self.UnnamedGeoms = []
        
    clear = remove_all

    def remove(self, label):
        if label in self.NamedGeoms:
            del self.NamedGeoms[label]

    def __getattr__(self, name):
        return getattr(self.transform, name)
        
