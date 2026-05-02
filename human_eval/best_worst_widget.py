from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QTextEdit, QButtonGroup, QRadioButton, \
    QMessageBox

from human_eval.qt_utils import set_button_group_checked
from pager_bar import PagerBar


class BestWorstWidget(QWidget):
    QUESTIONS = ["Best", "Worst"]

    def __init__(self, choices: list[tuple[str, ...]], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.choices = choices
        if not all([len(c) == 4 for c in choices]):
            raise NotImplementedError("only 4 choices are supported")
        self.result: list[list[int | None]] = [[None] * len(self.QUESTIONS) for _ in range(len(self.choices))]

        self.setWindowTitle("Best\u2013Worst Judgment")

        text_ids = [chr(ord("A") + i) for i in range(4)]

        self.texts: list[QTextEdit] = []
        display_layout = QGridLayout()
        for i in range(len(text_ids)):
            layout = QVBoxLayout()
            label = QLabel(f"Text {text_ids[i]}", alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            text = QTextEdit()
            text.setReadOnly(True)
            layout.addWidget(text, stretch=1)
            display_layout.addLayout(layout, i // 2, i % 2)
            self.texts.append(text)

        self.btn_groups: list[QButtonGroup] = []
        question_layout = QGridLayout()
        for row, q in enumerate(self.QUESTIONS):
            label = QLabel(f"{q}:")
            question_layout.addWidget(label, row, 0)
            group = QButtonGroup(self)
            for i, id in enumerate(text_ids):
                btn = QRadioButton(id)
                group.addButton(btn, i)
                question_layout.addWidget(btn, row, i + 1)
            group.idClicked.connect(lambda id, question=row: self.on_choice_clicked(question, id))
            self.btn_groups.append(group)

        self.pager_bar = PagerBar(len(self.choices))
        self.pager_bar.page_changed.connect(self.on_page_changed)

        layout = QVBoxLayout(self)
        layout.addLayout(display_layout)
        layout.addLayout(question_layout)
        layout.addWidget(self.pager_bar)

        self.on_page_changed(self.pager_bar.index())

    def on_page_changed(self, index: int) -> None:
        choices = self.choices[index]
        # update text
        for text, widget in zip(choices, self.texts):
            widget.setText(text)
        # update buttons
        prev_choices = self.result[index]
        for c, group in zip(prev_choices, self.btn_groups):
            set_button_group_checked(group, c)

    def on_choice_clicked(self, question: int, id: int) -> None:
        current = self.result[self.pager_bar.index()]
        current[question] = id
        for i, group in enumerate(self.btn_groups):
            if i == question:
                continue
            if group.checkedId() != id:
                continue
            set_button_group_checked(group, None)
            current[i] = None

    def closeEvent(self, event: QCloseEvent) -> None:
        unanswered = [i + 1 for i, r in enumerate(self.result) if any(c is None for c in r)]
        if unanswered:
            idx = ", ".join(map(str, unanswered))
            QMessageBox.warning(
                self,
                "Incomplete",
                f"Unanswered groups: {idx}.\nPlease complete all before exiting.",
            )
            event.ignore()
        else:
            event.accept()
