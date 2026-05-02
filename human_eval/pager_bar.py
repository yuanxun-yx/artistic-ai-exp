from PySide6.QtCore import Signal, Qt

from PySide6.QtWidgets import QWidget, QPushButton, QHBoxLayout, QLabel


class PagerBar(QWidget):
    page_changed = Signal(int)

    def __init__(self, total: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.total = total
        self._index = 0

        self.prev_btn = QPushButton("<")
        self.prev_btn.setToolTip("Previous")
        self.prev_btn.clicked.connect(lambda: self.set_index(self._index - 1))
        self.next_btn = QPushButton(">")
        self.next_btn.setToolTip("Next")
        self.next_btn.clicked.connect(lambda: self.set_index(self._index + 1))
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        layout = QHBoxLayout(self)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.next_btn)

        self.on_page_changed()

    @property
    def index(self) -> int:
        return self._index

    def set_index(self, index: int) -> None:
        index = max(0, min(self.total - 1, index))
        if index == self._index:
            return
        self._index = index
        self.on_page_changed()
        self.page_changed.emit(self._index)

    def on_page_changed(self) -> None:
        self.label.setText(f"Page {self._index + 1} of {self.total}")
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(self._index < self.total - 1)
