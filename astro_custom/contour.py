from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astromodels import Model
from typing import List
import betagen

def contour_plot(
    samples: np.ndarray,
    model: Model,
    function,
    energies: np.ndarray,
    levels: List[float] = [65, 95, 99],
    unit: str = "erg / cm2 / s",
    thin: int = 10,
        base_color = '#CE33FF',
    ax=None,
    **kwargs
):
    y = []

    b = betagen.BetaGen(base_color)


    for sample in tqdm(samples[::thin]):
        model.set_free_parameters(sample)
        y.append((energies**2 * function(energies)).to(unit).value.tolist())

    y = np.array(y)

    levels.sort()

    if ax is None:
        fig, ax = plt.subplots()

    else:
        fig = ax.get_figure()

    for level, color in zip(levels[::-1], [b.dark, b.mid, b.light][::-1]):
        low, high = np.percentile(
            y, [50.0 - level / 2.0, 50 + level / 2.0], axis=0
        )

        ax.fill_between(energies.value, low, high, color=color, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")

    return fig
