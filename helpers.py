import numpy as np
import Gene
import colour

def fitness(gene: Gene, canvas: np.ndarray, target_image) -> float:
    gene.render(canvas)
    if canvas is None or target_image is None:
        raise ValueError("Canvas or target image is None")
    return np.sum(colour.difference.delta_e.delta_E_CIE1976(canvas, target_image))
