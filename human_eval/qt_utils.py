from PySide6.QtWidgets import QButtonGroup


def set_button_group_checked(group: QButtonGroup, id: int | None) -> None:
    group.setExclusive(False)
    for i in range(len(group.buttons())):
        btn = group.button(i)
        # check doesn't trigger clicked, only toggled
        btn.setChecked(id == i)
    group.setExclusive(True)
