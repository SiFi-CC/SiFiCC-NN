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

    def plot_event(self, event, detector, event_idx, show_sipms=True, show_cluster_area=True, show_photon_hits=True):
        self.axes.clear()
        event.plot(detector, event_idx=event_idx, ax=self.axes, show_sipms=show_sipms, show_cluster_area=show_cluster_area, show_photon_hits=show_photon_hits)
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
        self.block_size = 10000

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
        self.show_photon_hits_checkbox = QCheckBox("Show Photon Hits")

        # Connect the stateChanged signal to the update_plot method
        self.show_sipms_checkbox.stateChanged.connect(self.update_plot)
        self.show_cluster_area_checkbox.stateChanged.connect(self.update_plot)
        self.show_photon_hits_checkbox.stateChanged.connect(self.update_plot)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.show_sipms_checkbox)
        checkbox_layout.addWidget(self.show_cluster_area_checkbox)
        checkbox_layout.addWidget(self.show_photon_hits_checkbox)
        main_layout.addLayout(checkbox_layout)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Event Index", "Num Clusters", "Contains Coupling Hit"])
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
            self.table_widget.setItem(row, 2, NumericTableWidgetItem(str(event.contains_coupling_hit)))

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
        show_photon_hits = self.show_photon_hits_checkbox.isChecked()
        event_idx = self.current_page * self.block_size + self.current_block.index(event)
        print(f"Plotting event {event_idx}")
        self.plot_canvas.plot_event(event, self.detector, event_idx, show_sipms, show_cluster_area, show_photon_hits)

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
            self.table_widget.setItem(row, 2, QTableWidgetItem("Yes" if event.contains_coupling_hit else "No"))

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
