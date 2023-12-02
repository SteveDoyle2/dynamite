from typing import Union
import numpy as np

def _response_squeeze(response: np.ndarray) -> np.ndarray:
    if response.shape[1] == 1:
        response = response[:, 0]
    return response


def _update_label(label: Union[list[str], str]) -> list[str]:
    if isinstance(label, str):
        label = [label]
    elif isinstance(label, list):
        pass
    else:
        raise NotImplementedError(label)
    return label

