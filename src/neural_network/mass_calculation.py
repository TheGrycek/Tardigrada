import numpy as np


def calculate_mass(points, echiniscus=False):
    # if mass of Echiniscus specie is  estimating, use different equation
    # TODO: pass nn input image scale factors and multiply with length and width
    head, ass, right, left = points
    length = np.linalg.norm(head - ass)
    width = np.linalg.norm(right - left)
    R = length / width
    density = 1.04

    if echiniscus:
        mass = (1 / 12) * length * np.pi * (length / R) ** 0.5 * density  # [grams]
    else:
        mass = length * np.pi * (length / 2 * R) ** 0.5 * density  # [grams]

    print(f"length / width: {R}, mass: {mass}")
    return mass
