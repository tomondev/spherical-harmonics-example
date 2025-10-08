import numpy as np


def fibonacci_sphere_spherical(samples=1000):
    """
    Generates points on a sphere using the Fibonacci lattice method.

    The points are returned in spherical coordinates (azimuthal angle theta,
    polar angle phi). This method ensures a relatively even distribution
    of points.

    Args:
        samples (int): The number of points to generate.

    Returns:
        numpy.ndarray: An array of shape (samples, 2) where each row
                       is a [theta, phi] coordinate pair in radians.
                       - phi (azimuth) is in [0, 2*pi]
                       - theta (polar/inclination) is in [0, pi]
    """
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # Create an array of indices
    i = np.arange(0, samples)

    # Calculate the polar angle (theta) from the Z-coordinate
    # The Z-coordinate is evenly spaced from (1 - 1/samples) to (-1 + 1/samples)
    # The offset of 0.5 helps to avoid placing points directly at the poles.
    z = 1 - (2 * (i + 0.5)) / samples
    theta = np.arccos(z)

    # Calculate the azimuthal angle (phi) using the golden angle
    # The result is wrapped to the interval [0, 2*pi] using the modulo operator
    phi = (golden_angle * i) % (2 * np.pi)

    # Stack the coordinates into an (n, 2) array
    points = np.column_stack((theta, phi))

    return points
