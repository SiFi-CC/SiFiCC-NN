from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QCheckBox,
)
from PyQt5.QtCore import Qt
import logging
from .PlotCanvas import PlotCanvas
from .DatasetLoaderThread import DatasetLoaderThread
from .NumericTableWidgetItem import NumericTableWidgetItem
from EventDisplayClasses.geometry.Detector import Detector


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

    def __init__(self, dataset_path, mode):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.resize(800, 600)

        # Mode and path
        self.mode = mode
        self.dataset_path = dataset_path
        logging.info(f"Dataset path: {self.dataset_path}")
        logging.info(f"Mode: {self.mode}")

        # DatasetReader
        self.reader = None

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

        # Detector
        self.detector = Detector(self.mode)

        # Checkboxes for plot options
        self.show_sipms_checkbox = QCheckBox("Show SiPMs")
        self.show_cluster_area_checkbox = QCheckBox("Show Cluster Area")
        if self.mode == "CC-4to1":
            self.show_compton_hits_checkbox = QCheckBox("Show Compton Hits")
        elif self.mode == "CM-4to1":
            self.show_photon_hits_checkbox = QCheckBox("Show Photon Hits")

        # Connect the stateChanged signal to the update_plot method
        self.show_sipms_checkbox.stateChanged.connect(self.update_plot)
        self.show_cluster_area_checkbox.stateChanged.connect(self.update_plot)
        if self.mode == "CC-4to1":
            self.show_compton_hits_checkbox.stateChanged.connect(self.update_plot)
        elif self.mode == "CM-4to1":
            self.show_photon_hits_checkbox.stateChanged.connect(self.update_plot)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.show_sipms_checkbox)
        checkbox_layout.addWidget(self.show_cluster_area_checkbox)
        if self.mode == "CC-4to1":
            checkbox_layout.addWidget(self.show_compton_hits_checkbox)
        elif self.mode == "CM-4to1":
            checkbox_layout.addWidget(self.show_photon_hits_checkbox)
        main_layout.addLayout(checkbox_layout)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        if self.mode == "CC-4to1":
            interaction_string = "Is Compton Event"
        elif self.mode == "CM-4to1":
            interaction_string = "Constains Coupling Hit"
        if self.mode not in ["CC-4to1", "CM-4to1"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        self.table_widget.setHorizontalHeaderLabels(
            ["Event Index", "Num Clusters", interaction_string]
        )
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

        # Set the initial path label
        self.path_label.setText(f"Current Dataset Path: {self.dataset_path}")

        # Load Dataset
        self.load_dataset()

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
        directory = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", "", options=options
        )
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
            self.table_widget.setItem(
                row, 1, NumericTableWidgetItem(str(event.nClusters))
            )
            if self.mode == "CC-4to1":
                self.table_widget.setItem(
                    row,
                    2,
                    NumericTableWidgetItem(str(event.contains_non_compton_hit == False)),
                )
            elif self.mode == "CM-4to1":
                self.table_widget.setItem(
                    row, 2, NumericTableWidgetItem(str(event.contains_coupling_hit))
                )

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
            QMessageBox.warning(
                self, "No Dataset Selected", "Please select a dataset path first."
            )
            return None
        try:
            logging.info(f"Loading block {page} ...")
            block = next(
                self.reader.read(
                    block_size=self.block_size,
                    start_index=page * self.block_size,
                    initial=initial,
                )
            )
            logging.info(f"Successfully loaded block {page}")
            self.page_label.setText(f"Page {self.current_page}")
            logging.info(f"Page {self.current_page} opened")
            self.update_table()
            logging.info("Table updated")
            return block
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load block: {str(e)}")
            self.current_page -= 1
            logging.error(f"Failed to load block: {str(e)}")
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
            QMessageBox.warning(
                self, "No Dataset Selected", "Please select a dataset path first."
            )
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
            QMessageBox.warning(
                self, "No Dataset Selected", "Please select a dataset path first."
            )
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
        if self.mode == "CC-4to1":
            show_CMphoton_hits = False
            show_compton_hits = self.show_compton_hits_checkbox.isChecked()
            logging.info(f"Compton hits are shown: {show_compton_hits}")
        elif self.mode == "CM-4to1":
            show_CMphoton_hits = self.show_photon_hits_checkbox.isChecked()
            show_compton_hits = False
            logging.info(f"Photon hits are shown: {show_CMphoton_hits}")
        event_idx = self.current_page * self.block_size + self.current_block.index(event)
        logging.info(f"Plotting event {event_idx}")
        self.plot_canvas.plot_event(
            event,
            self.detector,
            event_idx,
            show_sipms,
            show_cluster_area,
            show_compton_hits=show_compton_hits,
			show_CMphoton_hits=show_CMphoton_hits,
        )

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
            self.table_widget.setItem(
                row, 1, NumericTableWidgetItem(str(event.nClusters))
            )
            if self.mode == "CC-4to1":
                self.table_widget.setItem(
                    row,
                    2,
                    QTableWidgetItem("No" if event.contains_non_compton_hit else "Yes"),
                )
            elif self.mode == "CM-4to1":
                self.table_widget.setItem(
                    row, 2, QTableWidgetItem("Yes" if event.contains_coupling_hit else "No")
                )

    def update_plot(self):
        # Get the selected row number from the table
        selected_row = self.table_widget.currentRow()
        # Call the view_plot method with the selected row number
        self.view_event(selected_row, 0)