import numpy as np


def calculate_mass(points, scale=1, echiniscus=False):
    # If the mass of the species Echiniscus is estimated, use different equation
    # TODO: pass nn input image scale factors and multiply with length and width
    head, ass, right, left = points
    length = np.linalg.norm(head - ass) * scale
    width = np.linalg.norm(right - left) * scale
    R = length / width
    density = 1.04

    if echiniscus:
        mass = (1 / 12) * length * np.pi * (length / R) ** 0.5 * density  # [grams]
    else:
        mass = length * np.pi * (length / 2 * R) ** 0.5 * density  # [grams]

    print(f"length / width: {R}, mass: {mass}")
    return mass
