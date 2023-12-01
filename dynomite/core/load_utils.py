from typing import Union


def _update_label(label: Union[list[str], str]) -> list[str]:
    if isinstance(label, str):
        label = [label]
    elif isinstance(label, list):
        pass
    else:
        raise NotImplementedError(label)
    return label

