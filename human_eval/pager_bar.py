from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class PagerBar(QWidget):
    page_changed = Signal(int)

    def __init__(self, total: int, /, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total = total
        self._index = 0

        self.prev_btn = QPushButton("<")
        self.prev_btn.setToolTip("Previous")
        self.prev_btn.clicked.connect(lambda: self.set_index(self._index - 1))
        self.next_btn = QPushButton(">")
        self.next_btn.setToolTip("Next")
        self.next_btn.clicked.connect(lambda: self.set_index(self._index + 1))
        front_label = QLabel("Page")
        self.page_num = QLineEdit(alignment=Qt.AlignmentFlag.AlignCenter)
        self.page_num.setFixedWidth(self.page_num.minimumSizeHint().width())
        self.page_num.setValidator(QIntValidator(1, self.total))
        self.page_num.editingFinished.connect(
            lambda: self.set_index(int(self.page_num.text()) - 1)
        )
        end_label = QLabel(f"of {self.total}")
        layout = QHBoxLayout(self)
        layout.addWidget(self.prev_btn)
        layout.addStretch(1)
        layout.addWidget(front_label)
        layout.addWidget(self.page_num)
        layout.addWidget(end_label)
        layout.addStretch(1)
        layout.addWidget(self.next_btn)

        self.on_page_changed()

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
        self.page_num.setText(str(self._index + 1))
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(self._index < self.total - 1)
