from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pager_bar import PagerBar


class PairwiseWidget(QWidget):
    OPTIONS = [
        "A is much better than B",
        "A is slightly better than B",
        "About the same",
        "B is slightly better than A",
        "B is much better than A",
    ]

    def __init__(self, pairs: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.pairs = pairs
        self.result = [None] * len(self.pairs)

        self.setWindowTitle("Preference Judgment")

        self.left_text = QTextEdit()
        self.left_text.setReadOnly(True)
        left_label = QLabel("Text A", alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout = QVBoxLayout()
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_text, stretch=1)
        self.right_text = QTextEdit()
        self.right_text.setReadOnly(True)
        right_label = QLabel("Text B", alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout = QVBoxLayout()
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.right_text, stretch=1)
        display_layout = QHBoxLayout()
        display_layout.addLayout(left_layout)
        display_layout.addLayout(right_layout)

        choice_layout = QVBoxLayout()
        question = QLabel("Which text is better overall?")
        choice_layout.addWidget(question)
        self.btn_group = QButtonGroup(self)  # important
        for i, text in enumerate(self.OPTIONS):
            btn = QRadioButton(text)
            self.btn_group.addButton(btn, i)
            choice_layout.addWidget(btn)
        self.btn_group.idClicked.connect(self.on_choice_clicked)

        self.pager_bar = PagerBar(len(self.pairs))
        self.pager_bar.page_changed.connect(self.on_page_changed)

        layout = QVBoxLayout(self)
        layout.addLayout(display_layout, stretch=1)
        layout.addLayout(choice_layout)
        layout.addWidget(self.pager_bar)

        self.on_page_changed(self.pager_bar.index)

    def on_page_changed(self, index: int) -> None:
        # update text
        left, right = self.pairs[index]
        self.left_text.setText(left)
        self.right_text.setText(right)
        # update choice
        choice = self.result[index]
        self.btn_group.setExclusive(False)
        for i in range(len(self.OPTIONS)):
            btn = self.btn_group.button(i)
            btn.setChecked(choice == i)
        self.btn_group.setExclusive(True)

    def on_choice_clicked(self, id: int) -> None:
        self.result[self.pager_bar.index] = id

    def closeEvent(self, event: QCloseEvent) -> None:
        unanswered = [i + 1 for i, r in enumerate(self.result) if r is None]
        if unanswered:
            idx = ", ".join(map(str, unanswered))
            QMessageBox.warning(
                self,
                "Incomplete",
                f"Unanswered pairs: {idx}.\nPlease complete all before exiting.",
            )
            event.ignore()
        else:
            event.accept()
