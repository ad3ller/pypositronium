from math import exp
from numba import njit


@njit
def radial_integral(n1, l1, n2, l2, step=0.005, rmin=0.65, p=1):
    """Use the Numerov method to calculate:

    integrate(R_nl(n1, l1) * r^(2 + p) * R_nl(n2, l2), (r, 0, oo))
    """
    w11 = -0.5 * n1**-2.0
    w12 = (l1 + 0.5) ** 2.0
    w21 = -0.5 * n2**-2.0
    w22 = (l2 + 0.5) ** 2.0
    nmax = max(n1, n2)
    rmax = 2 * nmax * (nmax + 15)
    r_in1 = n1**2.0 - n1 * (n1**2.0 - l1 * (l1 + 1.0)) ** 0.5
    r_in2 = n2**2.0 - n2 * (n2**2.0 - l2 * (l2 + 1.0)) ** 0.5
    step_sq = step**2.0

    # initialise
    r_sub2 = rmax
    r_sub1 = rmax * exp(-step)

    g1_sub2 = 2.0 * r_sub2**2.0 * (-1.0 / r_sub2 - w11) + w12
    g1_sub1 = 2.0 * r_sub1**2.0 * (-1.0 / r_sub1 - w11) + w12
    g2_sub2 = 2.0 * r_sub2**2.0 * (-1.0 / r_sub2 - w21) + w22
    g2_sub1 = 2.0 * r_sub1**2.0 * (-1.0 / r_sub1 - w21) + w22

    y1_sub2 = 1e-10
    y1_sub1 = y1_sub2 * (1.0 + step * g1_sub2**0.5)
    y2_sub2 = 1e-10
    y2_sub1 = y2_sub2 * (1.0 + step * g2_sub2**0.5)

    norm1 = y1_sub2**2.0 * r_sub2**2.0 + y1_sub1**2.0 * r_sub1**2.0
    norm2 = y2_sub2**2.0 * r_sub2**2.0 + y2_sub1**2.0 * r_sub1**2.0

    integral = y1_sub2 * y2_sub2 * r_sub2 ** (2.0 + p) + y1_sub1 * y2_sub1 * r_sub1 ** (
        2.0 + p
    )

    i = 2
    r = r_sub1
    dr1 = exp(-step) ** (-l1 - 1) - 1.0
    dr2 = exp(-step) ** (-l2 - 1) - 1.0
    while r >= rmin:
        # Numerov method
        r = rmax * exp(-i * step)
        g1 = 2.0 * r**2.0 * (-1.0 / r - w11) + w12
        g2 = 2.0 * r**2.0 * (-1.0 / r - w21) + w22
        y1 = (
            y1_sub2 * (g1_sub2 - (12.0 / step_sq))
            + y1_sub1 * (10.0 * g1_sub1 + (24.0 / step_sq))
        ) / ((12.0 / step_sq) - g1)
        y2 = (
            y2_sub2 * (g2_sub2 - (12.0 / step_sq))
            + y2_sub1 * (10.0 * g2_sub1 + (24.0 / step_sq))
        ) / ((12.0 / step_sq) - g2)

        # check for divergence
        if r < r_in1:
            dy1 = abs((y1 - y1_sub1) / y1_sub1)
            if dy1 > dr1:
                break

        if r < r_in2:
            dy2 = abs((y2 - y2_sub1) / y2_sub1)
            if dy2 > dr2:
                break

        # store vals
        norm1 += y1**2.0 * r**2.0
        norm2 += y2**2.0 * r**2.0
        integral += y1 * y2 * r ** (2.0 + p)

        # next iteration
        r_sub1 = r

        g1_sub2 = g1_sub1
        g1_sub1 = g1
        g2_sub2 = g2_sub1
        g2_sub1 = g2

        y1_sub2 = y1_sub1
        y1_sub1 = y1
        y2_sub2 = y2_sub1
        y2_sub1 = y2
        i += 1
    return integral * (norm1 * norm2) ** -0.5
