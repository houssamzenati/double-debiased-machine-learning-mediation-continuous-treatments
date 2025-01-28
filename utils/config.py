import numpy as np
from itertools import cycle, islice

### Plot options

number_classes = [2]

colors = np.array(
    list(
        islice(
            cycle(
                [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                ]
            ),
            int(max(number_classes) + 1),
        )
    )
)

markers = np.array(
    list(
        islice(
            cycle(
                [
                    ".",
                    "+",
                    "x",
                    "v",
                    "s",
                    "p",
                ]
            ),
            int(max(number_classes) + 1),
        )
    )
)

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.e-12
