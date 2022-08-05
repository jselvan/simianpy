import matplotlib.lines as lines
import matplotlib.transforms as transforms


class DraggableLine:
    def __init__(
        self,
        ax,
        pos,
        orientation="h",
        pick_radius=5,
        line_params=None,
        pick_callback=None,
        release_callback=None,
    ):
        self.canvas = ax.get_figure().canvas
        self.pos = pos
        self.orientation = orientation
        self.pick_callback = pick_callback
        self.release_callback = release_callback
        self.line_params = line_params or {}
        self.lines = []
        self.pick_radius = pick_radius

        self.add_to_axes(ax)
        self.picked = False
        self.connect()

    def add_to_axes(self, ax):
        if self.orientation == "h":
            transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            x = [0, 1]
            y = [self.pos, self.pos]
        elif self.orientation == "v":
            transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            x = [self.pos, self.pos]
            y = [0, 1]
        line = lines.Line2D(x, y, transform=transform, picker=True, **self.line_params)
        line.set_pickradius(self.pick_radius)

        ax.add_line(line)
        self.lines.append(line)
        self.canvas.draw_idle()

    def connect(self):
        self.cid_pick = self.canvas.mpl_connect("pick_event", self.on_pick)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release = self.canvas.mpl_connect(
            "button_release_event", self.on_release
        )

    def on_pick(self, event):
        if any(event.artist == line for line in self.lines):
            self.picked = True
            if self.pick_callback is not None:
                self.pick_callback()

    def on_motion(self, event):
        self.pos = event.ydata if self.orientation == "h" else event.xdata
        if not self.picked:
            return
        if self.orientation == "h":
            for line in self.lines:
                line.set_ydata([self.pos, self.pos])
        else:
            for line in self.lines:
                line.set_xdata([self.pos, self.pos])
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.orientation == "h":
            self.pos = self.lines[0].get_ydata()[0]
        else:
            self.pos = self.lines[0].get_xdata()[0]

        self.picked = False
        if self.release_callback is not None:
            self.release_callback(self.pos)
