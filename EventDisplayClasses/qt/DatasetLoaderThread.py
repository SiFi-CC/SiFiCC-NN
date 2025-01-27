from PyQt5.QtCore import QThread, pyqtSignal
from EventDisplayClasses.data.DatasetReader import DatasetReader

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
        self.mode = reader.get_mode()
        self.dataset_loaded.emit(reader)
