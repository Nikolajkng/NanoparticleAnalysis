from PyQt5 import QtWidgets
class TableData():
    def __init__(self, data: dict[list[float]]):
        self.data = data

    def __set_row_items_by_label(self, table_widget, row_label, new_values):
        # Find the row index by checking the vertical header items
        for row in range(table_widget.rowCount()):
            header_item = table_widget.verticalHeaderItem(row)
            if header_item and header_item.text() == row_label:
                # Update each column in the row with new values
                for col in range(table_widget.columnCount()):
                    item = table_widget.item(row, col)
                    if item:
                        item.setText(str(new_values[col]))  # Update text
                    else:
                        # Create a new QTableWidgetItem if it doesn't exist
                        table_widget.setItem(row, col, QtWidgets.QTableWidgetItem(str(new_values[col])))
                return True  # Indicate success
        return False  # Indicate failure (label not found)

    def insertIn(self, table: QtWidgets.QTableWidget):
        for i, (row_label, values) in enumerate(self.data.items()):
            self.__set_row_items_by_label(table, row_label, values)

    
