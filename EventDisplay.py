import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from EventDisplayClasses.qt.DatasetViewer import DatasetViewer
from EventDisplayClasses.data.DatasetReader import DatasetReader




if __name__ == "__main__":
    app = QApplication([])

    # Open dataset selection dialog before showing the main window
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    dataset_path = QFileDialog.getExistingDirectory(None, "Select Dataset Directory", options=options)
    if dataset_path:
        # Determine mode
        reader = DatasetReader(dataset_path)
        mode = reader.get_mode()
        # Show main window
        window = DatasetViewer(dataset_path, mode)
        window.show()
        sys.exit(app.exec_())
