import numpy as np


def vector_angle(vec1, vec2):
    """
    Calculates the angle between two given vectors. Vectors are normalized and their angle is
    calculated from the vector dot product. The range of the resulting angle is clipled from
    -1 to 1.

    Args:
         vec1: ndarray (3,); 3-dim origin vector
         vec2: ndarray (3,); 3-dim origin vector

    Returns:
         float; Angle between vectors in radians
    """

    vec1 /= np.sqrt(np.dot(vec1, vec1))
    vec2 /= np.sqrt(np.dot(vec2, vec2))

    return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))


def compton_scattering_angle(e1, e2):
    """
    Calculates the compton scattering angle from given energies.

    Energies are given as primary gamma energy and gamma energy after scattering.
    Compton scattering angle is derived from the formula:

        cos(\theta) = 1.0 - kMe * (\frac{1}{E_2} - \frac{1}{E_1})

    Args:
         e1: float; initial gamma energy
         e2: float; gamma energy after scattering

    Return:
        theta: float; compton scattering angle in rad
    """
    # Exception for unphysical energy value
    if e1 <= 0.0 or e2 <= 0.0:
        return 0.0

    kMe = 0.510999  # MeV/c^2
    costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / e1)

    # TODO: check if exception for cosine range makes sense
    if abs(costheta) > 1:
        return 0.0
    else:
        theta = np.arccos(costheta)  # rad
        return theta
