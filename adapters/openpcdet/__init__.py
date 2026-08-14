"""OpenPCDet integration adapter.

The OpenPCDet package itself is an external dependency. Copy or import the
dataset class from this directory inside a compatible OpenPCDet checkout.
"""

from .marscenes3d_dataset import MarScenes3DDataset

__all__ = ["MarScenes3DDataset"]
