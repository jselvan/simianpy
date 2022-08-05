import numpy as np
from collections import namedtuple

Press = namedtuple('Press', ['x', 'y', 'w', 'h', 'ar', 'mx', 'my'])

class DraggableResizeableRectangle:
    """
    Draggable and resizeable rectangle with the animation blit techniques.
    Based on example code at http://matplotlib.sourceforge.net/users/event_handling.html
    If *allow_resize* is *True* the recatngle can be resized by dragging its
    lines. *border_tol* specifies how close the pointer has to be to a line for
    the drag to be considered a resize operation. Dragging is still possible by
    clicking the interior of the rectangle. *fixed_aspect_ratio* determines if
    the recatngle keeps its aspect ratio during resize operations.
    """

    lock = None  # only one can be animated at a time

    def __init__(
        self, rect, border_tol=0.15, allow_resize=True, fixed_aspect_ratio=False, on_update_callback=None
    ):
        self.rect = rect
        self.border_tol = border_tol
        self.allow_resize = allow_resize
        self.fixed_aspect_ratio = fixed_aspect_ratio
        self.press = None
        self.background = None
        self.on_update_callback = on_update_callback

    def connect(self):
        "connect to all the events we need"
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event):
        "on button press we will see if the mouse is over us and store some data"
        if event.inaxes != self.rect.axes:
            return
        if DraggableResizeableRectangle.lock is not None:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        # print 'event contains', self.rect.xy
        x, y = self.rect.xy
        w, h = self.rect.get_width(), self.rect.get_height()
        aspect_ratio = w/h if self.fixed_aspect_ratio else None
        self.press = Press(x,y,w,h,aspect_ratio,event.xdata,event.ydata)
        DraggableResizeableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        "on motion we will move the rect if the mouse is over us"
        if DraggableResizeableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes:
            return
        self.dx = event.xdata - self.press.mx
        self.dy = event.ydata - self.press.my

        self.update_rect()
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        "on release we reset the press data"
        if DraggableResizeableRectangle.lock is not self:
            return

        self.press = None
        DraggableResizeableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        "disconnect all the stored connection ids"
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    def get_click_type(self):
        wtol, htol = self.border_tol*self.press.w, self.border_tol*self.press.h
        if not self.allow_resize or (
            self.press.x+wtol <= self.press.mx <= self.press.x+self.press.w-wtol
            and self.press.y+htol <= self.press.my <= self.press.y+self.press.h-htol
        ):
            return 'drag'
        elif abs(self.press.x - self.press.mx) < wtol:
            return 'resize_w'
        elif abs(self.press.y - self.press.my) < htol:
            return 'resize_n'
        elif abs(self.press.x + self.press.w - self.press.mx) < wtol:
            return 'resize_e'
        elif abs(self.press.y + self.press.h - self.press.my) < htol:
            return 'resize_s'

    def update_rect(self):
        x, y, w, h, ar = self.press.x, self.press.y, self.press.w, self.press.h, self.press.ar
        dx, dy = self.dx, self.dy

        click_type = self.get_click_type()
        if click_type == 'drag':
            bounds = x+dx, y+dy, w, h
        elif click_type == 'resize_w':
            dy = 0 if ar is None else dx/ar
            bounds = x+dx, y+dy, w-dx, h-dy
        elif click_type == 'resize_n':
            dx = 0 if ar is None else dy*ar
            bounds = x+dx, y+dy, w-dx, h-dy
        elif click_type == 'resize_e':
            dy = 0 if ar is None else dx/ar
            bounds = x, y, w+dx, h+dy
        elif click_type == 'resize_s':
            dx = 0 if ar is None else dy*ar
            bounds = x, y, w+dx, h+dy
        self.rect.set_bounds(bounds)

        if self.on_update_callback is not None:
            self.on_update_callback(bounds)

if __name__ == '__main__':
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    colours = 'rgb'
    rects = [Rectangle((i-2, i-2), 1, 1, facecolor=colours[i]) for i in range(3)]
    dr_rects = [DraggableResizeableRectangle(rect, on_update_callback=print) for rect in rects]

    fig, ax = plt.subplots()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    
    for rect in rects: ax.add_artist(rect)
    for dr_rect in dr_rects: dr_rect.connect()