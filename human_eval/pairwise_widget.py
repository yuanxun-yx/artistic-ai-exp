from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PairwiseWidget(QWidget):
    OPTIONS = [
        "A is much better than B",
        "A is slightly better than B",
        "About the same",
        "B is slightly better than A",
        "B is much better than A",
    ]

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        super().__init__()

        self.pairs = pairs
        self.result = [None] * len(self.pairs)

        self.setWindowTitle("Preference Judgment")

        self.left_text = QTextEdit()
        self.left_text.setReadOnly(True)
        left_label = QLabel("Text A")
        left_layout = QVBoxLayout()
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_text, stretch=1)
        self.right_text = QTextEdit()
        self.right_text.setReadOnly(True)
        right_label = QLabel("Text B")
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

        self.prev_btn = QPushButton("<")
        self.prev_btn.setToolTip("Previous")
        self.prev_btn.clicked.connect(self.on_prev_clicked)
        self.next_btn = QPushButton(">")
        self.next_btn.setToolTip("Next")
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.page_text = QLabel()
        bar_layout = QHBoxLayout()
        bar_layout.addWidget(self.prev_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.page_text)
        bar_layout.addStretch()
        bar_layout.addWidget(self.next_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(display_layout, stretch=1)
        layout.addLayout(choice_layout)
        layout.addLayout(bar_layout)

        self.i = 0
        self.on_page_changed()

    def on_page_changed(self) -> None:
        # update text
        left, right = self.pairs[self.i]
        self.left_text.setText(left)
        self.right_text.setText(right)
        # update choice
        choice = self.result[self.i]
        self.btn_group.setExclusive(False)
        for i in range(len(self.OPTIONS)):
            btn = self.btn_group.button(i)
            btn.setChecked(choice == i)
        self.btn_group.setExclusive(True)
        # update bar
        if self.i == 0:
            self.prev_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(True)
        if self.i == len(self.pairs) - 1:
            self.next_btn.setEnabled(False)
        else:
            self.next_btn.setEnabled(True)
        self.page_text.setText(f"Pair {self.i + 1} of {len(self.pairs)}")

    def on_prev_clicked(self) -> None:
        self.i -= 1
        self.on_page_changed()

    def on_next_clicked(self) -> None:
        self.i += 1
        self.on_page_changed()

    def on_choice_clicked(self, id: int) -> None:
        self.result[self.i] = id

    def closeEvent(self, event: QCloseEvent) -> None:
        unanswered = [i for i, r in enumerate(self.result) if r is None]
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
