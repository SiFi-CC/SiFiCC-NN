from PyQt5.QtWidgets import QTableWidgetItem

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