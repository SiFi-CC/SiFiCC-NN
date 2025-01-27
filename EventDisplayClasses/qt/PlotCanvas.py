from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas for plotting 3D event displays.

    Attributes:
        axes (Axes3D): The 3D axes for plotting.

    Methods:
        __init__(parent=None):
            Initializes the PlotCanvas with a 3D subplot.

        plot_event(event, detector, event_idx, show_sipms=True, show_cluster_area=True, show_photon_hits=True):
            Plots a given event on the 3D canvas.
            Args:
                event: The event data to be plotted.
                detector: The detector configuration.
                event_idx: The index of the event to be plotted.
                show_sipms (bool): Whether to show SiPMs in the plot. Default is True.
                show_cluster_area (bool): Whether to show the cluster area in the plot. Default is True.
                show_photon_hits (bool): Whether to show photon hits in the plot. Default is True.
    """

    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111, projection="3d")
        super().__init__(fig)
        self.setParent(parent)

    def plot_event(
        self,
        event,
        detector,
        event_idx,
        show_sipms=True,
        show_cluster_area=True,
        show_compton_hits=True,
        show_CMphoton_hits=True,
    ):
        self.axes.clear()
        event.plot(
            detector,
            event_idx=event_idx,
            ax=self.axes,
            show_sipms=show_sipms,
            show_cluster_area=show_cluster_area,
            show_compton_hits=show_compton_hits,
			show_CMphoton_hits=show_CMphoton_hits,
        )
        self.draw()