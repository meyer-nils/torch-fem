"""Generate shape function plots.

Usage: python docs/images/shape_functions/plot_elements.py
"""

from pathlib import Path

from torchfem.elements import Bar1, Bar2, Quad1, Quad2, Tria1, Tria2

ELEMENTS = [Bar1, Bar2, Tria1, Tria2, Quad1, Quad2]

# Write next to this script, which does not move with an installed torchfem.
IMAGES_DIR = Path(__file__).parent


def main():
    for element in ELEMENTS:
        element.plot(path=IMAGES_DIR)
        print(f"Saved {element.__name__} shape functions")


if __name__ == "__main__":
    main()
