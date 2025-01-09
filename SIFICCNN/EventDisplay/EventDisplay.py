import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QFileDialog, QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from utils import DatasetReader, Detector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class NumericTableWidgetItem(QTableWidgetItem):
    """
    A custom QTableWidgetItem that allows for numeric comparison.

    This class overrides the less-than operator to enable sorting of table items
    based on their numeric value rather than their string representation.

    Methods:
        __lt__(other): Compares the numeric value of this item with another NumericTableWidgetItem.
    """
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return int(self.text()) < int(other.text())
        return super().__lt__(other)

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
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        self.setParent(parent)

    def plot_event(self, event, detector, event_idx, show_sipms=True, show_cluster_area=True, show_compton_hits=True):
        self.axes.clear()
        event.plot(detector, event_idx=event_idx, ax=self.axes, show_sipms=show_sipms, show_cluster_area=show_cluster_area, show_compton_hits=show_compton_hits)
        self.draw()

class DatasetLoaderThread(QThread):
    """
    A QThread subclass that loads a dataset in a separate thread.

    Attributes:
        dataset_loaded (pyqtSignal): Signal emitted when the dataset is loaded.
        dataset_path (str): Path to the dataset file.

    Methods:
        __init__(dataset_path):
            Initializes the DatasetLoaderThread with the given dataset path.
        run():
            Reads the dataset using DatasetReader and emits the dataset_loaded signal.
    """
    dataset_loaded = pyqtSignal(object)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

    def run(self):
        reader = DatasetReader(self.dataset_path)
        self.dataset_loaded.emit(reader)

class DatasetViewer(QMainWindow):
    """
    A GUI application for viewing and interacting with datasets.

    Attributes:
        dataset_path (str): Path to the dataset directory.
        reader (DatasetReader): Reader object for loading dataset.
        detector (Detector): Detector object for event visualization.
        current_block (list): List of events in the current block.
        current_page (int): Current page number.
        block_size (int): Number of events per block.
        central_widget (QWidget): Central widget of the main window.
        path_label (QLabel): Label displaying the current dataset path.
        browse_button (QPushButton): Button to browse and select dataset.
        show_sipms_checkbox (QCheckBox): Checkbox to toggle SiPMs display.
        show_cluster_area_checkbox (QCheckBox): Checkbox to toggle cluster area display.
        show_photon_hits_checkbox (QCheckBox): Checkbox to toggle photon hits display.
        table_widget (QTableWidget): Table widget to display event information.
        plot_canvas (PlotCanvas): Canvas for plotting events.
        prev_button (QPushButton): Button to load the previous block of events.
        page_label (QLabel): Label displaying the current page number.
        next_button (QPushButton): Button to load the next block of events.

    Methods:
        select_dataset(): Opens a file dialog to select the dataset directory.
        load_dataset(): Loads the dataset using a separate thread.
        on_dataset_loaded(reader): Callback when the dataset is loaded.
        load_first_block(): Loads the first block of events.
        update_table(): Updates the table widget with the current block of events.
        load_block(page, initial=False): Loads a block of events from the dataset.
        load_previous(): Loads the previous block of events.
        load_next(): Loads the next block of events.
        view_event(row, column): Views the selected event in a popup window.
        plot_event(event): Plots the selected event on the plot canvas.
        populate_table(): Populates the table with the current block of events.
        update_plot(): Updates the plot based on the selected checkboxes.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.resize(800, 600)

        # DatasetReader and Detector
        self.dataset_path = None
        self.reader = None
        self.detector = Detector()

        # State
        self.current_block = []
        self.current_page = 0
        self.block_size = 100

        # UI Elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        # Dataset Path Selector
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Current Dataset Path: Not Selected")
        self.path_label.setWordWrap(True)
        self.browse_button = QPushButton("Select Dataset")
        path_layout.addWidget(self.path_label, stretch=1)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)

        # Checkboxes for plot options
        self.show_sipms_checkbox = QCheckBox("Show SiPMs")
        self.show_cluster_area_checkbox = QCheckBox("Show Cluster Area")
        self.show_compton_hits_checkbox = QCheckBox("Show Compton Hits")

        # Connect the stateChanged signal to the update_plot method
        self.show_sipms_checkbox.stateChanged.connect(self.update_plot)
        self.show_cluster_area_checkbox.stateChanged.connect(self.update_plot)
        self.show_compton_hits_checkbox.stateChanged.connect(self.update_plot)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.show_sipms_checkbox)
        checkbox_layout.addWidget(self.show_cluster_area_checkbox)
        checkbox_layout.addWidget(self.show_compton_hits_checkbox)
        main_layout.addLayout(checkbox_layout)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Event Index", "Num Clusters", "Is Compton Event"])
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setSortingEnabled(True)
        splitter.addWidget(self.table_widget)

        # Plot Canvas
        self.plot_canvas = PlotCanvas(self)
        splitter.addWidget(self.plot_canvas)

        # Navigation Layout
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.page_label = QLabel("Page 0")
        self.next_button = QPushButton("Next")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.page_label, alignment=Qt.AlignCenter)
        nav_layout.addWidget(self.next_button)

        # Add to Layout
        main_layout.addLayout(nav_layout)

        # Signals
        self.browse_button.clicked.connect(self.select_dataset)
        self.prev_button.clicked.connect(self.load_previous)
        self.next_button.clicked.connect(self.load_next)
        self.table_widget.cellDoubleClicked.connect(self.view_event)

    def select_dataset(self):
        """
        Opens a file dialog for the user to select a dataset directory. 
        If a directory is selected, updates the dataset path and label, 
        and loads the dataset.

        Uses QFileDialog to open a directory selection dialog with options 
        to not use the native dialog.

        Attributes:
            dataset_path (str): The path to the selected dataset directory.
            path_label (QLabel): The label displaying the current dataset path.
        
        Methods:
            load_dataset(): Loads the dataset from the selected directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", "", options=options)
        if directory:
            self.dataset_path = directory
            self.path_label.setText(f"Current Dataset Path: {directory}")
            self.load_dataset()

    def load_dataset(self):
        """
        Loads the dataset using a separate thread.

        This method initializes a DatasetLoaderThread with the specified dataset path,
        connects the dataset_loaded signal to the on_dataset_loaded slot, and starts
        the thread to load the dataset asynchronously.
        """
        self.loader_thread = DatasetLoaderThread(self.dataset_path)
        self.loader_thread.dataset_loaded.connect(self.on_dataset_loaded)
        self.loader_thread.start()

    def on_dataset_loaded(self, reader):
        """
        Callback function that is called when a dataset is loaded.

        Parameters:
        reader (object): The dataset reader object that contains the loaded data.
        """
        self.reader = reader
        self.load_first_block()

    def load_first_block(self):
        """
        Load the first block of data and update the display table.

        This method sets the current page to the first page (index 0) and loads the
        corresponding block of data. If the block is successfully loaded, it updates
        the display table with the new data.
        """
        self.current_page = 0
        self.current_block = self.load_block(self.current_page, initial=True)
        if self.current_block:
            self.update_table()

    def update_table(self):
        """
        Updates the table widget with the current block of events.

        This method sets the row count of the table widget to the number of events
        in the current block and populates each row with event details. Each row
        displays the event ID, the number of clusters in the event, and whether the
        event contains a coupling hit.

        The event ID is calculated based on the current page and block size.

        Attributes:
            table_widget (QTableWidget): The table widget to be updated.
            current_block (list): The current block of events to be displayed.
            current_page (int): The current page number.
            block_size (int): The number of events per block.

        """
        self.table_widget.setRowCount(len(self.current_block))
        for row, event in enumerate(self.current_block):
            event_id = self.current_page * self.block_size + row
            self.table_widget.setItem(row, 0, NumericTableWidgetItem(str(event_id)))
            self.table_widget.setItem(row, 1, NumericTableWidgetItem(str(event.nClusters)))
            self.table_widget.setItem(row, 2, NumericTableWidgetItem(str(event.contains_non_compton_hit==False)))

    def load_block(self, page, initial=False):
        """
        Load a block of events from the dataset.

        Parameters:
        page (int): The page number to load.
        initial (bool): Flag indicating whether this is the initial load. Default is False.

        Returns:
        block: The loaded block of events if successful, None otherwise.

        Raises:
        Exception: If there is an error while loading the block, a critical message box is displayed and None is returned.
        """
        if not self.reader:
            QMessageBox.warning(self, "No Dataset Selected", "Please select a dataset path first.")
            return None
        try:
            block = next(self.reader.read(block_size=self.block_size, start_index=page * self.block_size, initial=initial))
            self.page_label.setText(f"Page {self.current_page}")
            self.update_table()
            return block
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load block: {str(e)}")
            self.current_page -= 1
            return None

    def load_previous(self):
        """
        Load the previous block of data and update the display table.

        This method decreases the current page index by one and loads the corresponding block of data.
        If no dataset is selected, a warning message is displayed.

        Preconditions:
        - A dataset must be selected and assigned to `self.reader`.
        - `self.current_page` must be greater than 0.

        Postconditions:
        - `self.current_page` is decremented by one.
        - `self.current_block` is updated with the data from the new current page.
        - The display table is updated with the new data.

        Raises:
        - Displays a warning message if no dataset is selected.
        """

        if not self.reader:
            QMessageBox.warning(self, "No Dataset Selected", "Please select a dataset path first.")
            return

        if self.current_page > 0:
            self.current_page -= 1
            self.current_block = self.load_block(self.current_page)
            if self.current_block:
                self.update_table()

    def load_next(self):
        """
        Load the next block of events.

        This method increments the current page number and loads the next block of events
        from the dataset. If no dataset is selected, it shows a warning message.

        Raises:
            QMessageBox.warning: If no dataset is selected.
        """

        if not self.reader:
            QMessageBox.warning(self, "No Dataset Selected", "Please select a dataset path first.")
            return

        self.current_page += 1
        self.current_block = self.load_block(self.current_page)

    def view_event(self, row, column):
        """
        View the selected event.

        Parameters:
        row (int): The row index of the event to view.
        column (int): The column index of the event to view (currently unused).

        Returns:
        None
        """
        if 0 <= row < len(self.current_block):
            selected_event = self.current_block[row]
            self.plot_event(selected_event)

    def plot_event(self, event):
        """
        Plots the given event on the canvas with various display options.

        Args:
            event (Event): The event object to be plotted.

        Returns:
            None

        Displays:
            - SiPMs if the corresponding checkbox is checked.
            - Cluster area if the corresponding checkbox is checked.
            - Photon hits if the corresponding checkbox is checked.

        Prints:
            The index of the event being plotted.
        """
        show_sipms = self.show_sipms_checkbox.isChecked()
        show_cluster_area = self.show_cluster_area_checkbox.isChecked()
        show_compton_hits = self.show_compton_hits_checkbox.isChecked()
        event_idx = self.current_page * self.block_size + self.current_block.index(event)
        print(f"Plotting event {event_idx}")
        self.plot_canvas.plot_event(event, self.detector, event_idx, show_sipms, show_cluster_area, show_compton_hits)

    def populate_table(self):
        """
        Populate the table with the current block of events.

        This method sets the number of rows in the table widget to the length of the current block of events.
        For each event in the current block, it calculates the event number, and populates the table with
        the event number, the number of clusters, and whether the event contains a coupling hit.

        Attributes:
            self.table_widget (QTableWidget): The table widget to populate.
            self.current_block (list): The current block of events to display.
            self.current_page (int): The current page number.
            self.block_size (int): The number of events per block.

        """
        self.table_widget.setRowCount(len(self.current_block))
        for row, event in enumerate(self.current_block):
            event_number = self.current_page * self.block_size + row
            self.table_widget.setItem(row, 0, NumericTableWidgetItem(str(event_number)))
            self.table_widget.setItem(row, 1, NumericTableWidgetItem(str(event.nClusters)))
            self.table_widget.setItem(row, 2, QTableWidgetItem("No" if event.contains_non_compton_hit else "Yes"))

    def update_plot(self):
        # Get the selected row number from the table
        selected_row = self.table_widget.currentRow()
        # Call the view_plot method with the selected row number
        self.view_event(selected_row, 0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DatasetViewer()
    viewer.show()
    sys.exit(app.exec_())

"""import matplotlib.pyplot as plt

from ..utils import  vector_angle
from .utils import get_edges, get_compton_cone_aachen, get_compton_cone_cracow


class EventDisplay:

    def __init__(self,
                 coordinate_system="AACHEN"):
        # main plotting objects
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Internal event slot
        self.event = None

        # internal variables
        self.coordinate_system = coordinate_system

        # generate canvas
        self._generate_canvas()

    def _generate_canvas(self):
        if self.coordinate_system == "CRACOW":
            self.ax.set_box_aspect(aspect=(3, 1, 1))
            self.ax.set_xlim3d(-10, 300)
            self.ax.set_ylim3d(-55, 55)
            self.ax.set_zlim3d(-55, 55)
            self.ax.set_xlabel("x-axis [mm]")
            self.ax.set_ylabel("y-axis [mm]")
            self.ax.set_zlabel("z-axis [mm]")

        if self.coordinate_system == "AACHEN":
            self.ax.set_box_aspect(aspect=(1, 1, 3))
            self.ax.set_xlim3d(-55, 55)
            self.ax.set_ylim3d(-55, 55)
            self.ax.set_zlim3d(-10, 300)
            self.ax.set_xlabel("x-axis [mm]")
            self.ax.set_ylabel("y-axis [mm]")
            self.ax.set_zlabel("z-axis [mm]")

    def set_coordinate_system(self, coordinate_system):
        self.coordinate_system = coordinate_system
        self._generate_canvas()

    def load_event(self, event):
        self.event = event

    def _verify_event(self):
        if self.event is None:
            raise TypeError

    def draw_detector(self, color="blue"):
        self._verify_event()

        list_edge_scatterer = get_edges(self.event.scatterer.posx,
                                        self.event.scatterer.posy,
                                        self.event.scatterer.posz,
                                        self.event.scatterer.dimx,
                                        self.event.scatterer.dimy,
                                        self.event.scatterer.dimz)
        for i in range(len(list_edge_scatterer)):
            self.ax.plot3D(list_edge_scatterer[i][0],
                           list_edge_scatterer[i][1],
                           list_edge_scatterer[i][2],
                           color=color)
        list_edge_absorber = get_edges(self.event.absorber.posx,
                                       self.event.absorber.posy,
                                       self.event.absorber.posz,
                                       self.event.absorber.dimx,
                                       self.event.absorber.dimy,
                                       self.event.absorber.dimz)
        for i in range(len(list_edge_absorber)):
            self.ax.plot3D(list_edge_absorber[i][0],
                           list_edge_absorber[i][1],
                           list_edge_absorber[i][2],
                           color="blue")

    def draw_reference_axis(self):
        self._verify_event()

        endpoint = [0, 0, 0]
        if self.coordinate_system == "CRACOW":
            endpoint = [270 + 46.8 / 2, 0, 0]
        if self.coordinate_system == "AACHEN":
            endpoint = [0, 0, 270 + 46.8 / 2]

        self.ax.plot3D([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
                       color="black",
                       linestyle="--")

    def draw_promptgamma(self):
        self._verify_event()

        a = 250
        self.ax.plot3D([self.event.MCPosition_source.x, self.event.MCComptonPosition.x],
                       [self.event.MCPosition_source.y, self.event.MCComptonPosition.y],
                       [self.event.MCPosition_source.z, self.event.MCComptonPosition.z],
                       color="red")

        self.ax.plot3D([self.event.MCComptonPosition.x,
                        self.event.MCComptonPosition.x + a * self.event.MCDirection_scatter.x],
                       [self.event.MCComptonPosition.y,
                        self.event.MCComptonPosition.y + a * self.event.MCDirection_scatter.y],
                       [self.event.MCComptonPosition.z,
                        self.event.MCComptonPosition.z + a * self.event.MCDirection_scatter.z],
                       color="red")

    def draw_interactions(self):
        self._verify_event()

        for pos in self.event.MCPosition_e:
            self.ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

        for pos in self.event.MCPosition_p:
            self.ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

    def draw_cone_targets(self):
        self._verify_event()
        # target_energy_e, target_energy_p = self.event.get_target_energy()
        target_position_e, target_position_p = self.event.get_target_position()

        self.ax.plot3D(target_position_e.x, target_position_e.y, target_position_e.z,
                       "x", color="red", markersize=15, zorder=10)

        self.ax.plot3D(target_position_p.x, target_position_p.y, target_position_p.z,
                       "x", color="red", markersize=15, zorder=10)

        self.ax.plot3D(self.event.MCPosition_source.x, self.event.MCPosition_source.y,
                       self.event.MCPosition_source.z,
                       "o", color="red", markersize=4)

    def draw_cone_true(self):
        self._verify_event()

        # Main vectors needed for cone calculations
        target_energy_e, target_energy_p = self.event.get_target_energy()
        target_position_e, target_position_p = self.event.get_target_position()
        vec_ax1 = target_position_e
        vec_ax2 = target_position_p - target_position_e
        vec_src = self.event.MCPosition_source
        theta = vector_angle(vec_ax1 - vec_src, vec_ax2)

        list_cone = []
        if self.coordinate_system == "CRACOW":
            list_cone = get_compton_cone_cracow(vec_ax1, vec_ax2, vec_src, theta, sr=128)
        if self.coordinate_system == "AACHEN":
            list_cone = get_compton_cone_aachen(vec_ax1, vec_ax2, vec_src, theta, sr=128)

        for i in range(1, len(list_cone)):
            self.ax.plot3D([list_cone[i - 1][0], list_cone[i][0]],
                           [list_cone[i - 1][1], list_cone[i][1]],
                           [list_cone[i - 1][2], list_cone[i][2]],
                           color="black")
        for i in [8, 16, 32, 64]:
            self.ax.plot3D([vec_ax1.x, list_cone[i - 1][0]],
                           [vec_ax1.y, list_cone[i - 1][1]],
                           [vec_ax1.z, list_cone[i - 1][2]],
                           color="black")

    def draw_cluster_hits(self):
        self._verify_event()

        list_cluster_x = []
        list_cluster_y = []
        list_cluster_z = []
        for cl in self.event.RecoClusterPosition:
            list_cluster_x.append(cl.x)
            list_cluster_y.append(cl.y)
            list_cluster_z.append(cl.z)

        # plot fiber hits + cluster hits
        b = 10  # marker-size scaling factor
        for i in range(len(list_cluster_x)):
            """"""
            # fiber hits
            list_surface = surface_list(list_cluster_x[i], 0, list_cluster_z[i], 1.3, 100.0, 1.3)
            for j in range(len(list_surface)):
                ax.plot_wireframe(*list_surface[i], alpha=0.5, color="green")
            """"""
            # cluster hits
            self.ax.plot3D(list_cluster_x[i], list_cluster_y[i], list_cluster_z[i],
                           "X", color="orange",
                           markersize=b)

    def draw_fibre_hits(self):
        self._verify_event()
        # fibre hits plus boxes
        for i in range(len(self.event.FibreHit.FibrePosition)):
            self.ax.plot3D(self.event.FibreHit.FibrePosition[i].x,
                           self.event.FibreHit.FibrePosition[i].y,
                           self.event.FibreHit.FibrePosition[i].z,
                           "o",
                           color="lime")
            list_fibre_edges = get_edges(self.event.FibreHit.FibrePosition[i].x,
                                         0,
                                         self.event.FibreHit.FibrePosition[i].z,
                                         1.94,
                                         100,
                                         1.94)
            for j in range(len(list_fibre_edges)):
                self.ax.plot3D(list_fibre_edges[j][0],
                               list_fibre_edges[j][1],
                               list_fibre_edges[j][2],
                               color="lime")

    def draw_sipm_hits(self):
        self._verify_event()

        for i in range(len(self.event.SiPMHit.SiPMPosition)):
            list_sipm_edges = get_edges(self.event.SiPMHit.SiPMPosition[i].x,
                                        self.event.SiPMHit.SiPMPosition[i].y,
                                        self.event.SiPMHit.SiPMPosition[i].z,
                                        4.0,
                                        0,
                                        4.0)
            for j in range(len(list_sipm_edges)):
                self.ax.plot3D(list_sipm_edges[j][0],
                               list_sipm_edges[j][1],
                               list_sipm_edges[j][2],
                               color="darkgreen")
                
    def draw_all(self, coordinate_system):
        self.set_coordinate_system(coordinate_system)
        #self.draw_cluster_hits()
        self.draw_cone_targets()
        self.draw_cone_true()
        self.draw_detector()
        self.draw_fibre_hits()
        self.draw_interactions()
        self.draw_promptgamma()
        self.draw_reference_axis()
        self.draw_sipm_hits()

    @staticmethod
    def show():
        plt.show()
"""