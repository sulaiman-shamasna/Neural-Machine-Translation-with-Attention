import typing
from typing import Any, Tuple, Dict
import einops
import tensorflow as tf


class ShapeChecker:
    def __init__(self):
        """
        Initialize a ShapeChecker instance.

        This class is used to verify that tensors have consistent shapes across different operations.
        """
        self.shapes = {}

    def __call__(self, tensor: tf.Tensor, names: str, broadcast: bool = False) -> None:
        """
        Check the shape of the given tensor against the expected shapes stored in the cache.

        Args:
            tensor (tf.Tensor): The tensor to check.
            names (str): A string representing the names of the dimensions.
            broadcast (bool): If True, allows broadcasting for dimensions with size 1.

        Raises:
            ValueError: If a shape mismatch is found for any dimension.
        """
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)
            
            if broadcast and new_dim == 1:
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


# if __name__ == "__main__":
#     # Instantiate the ShapeChecker
#     shape_checker = ShapeChecker()

#     # Example tensors for checking
#     tensor1 = tf.random.uniform((2, 3, 4))
#     tensor2 = tf.random.uniform((2, 3, 4))

#     # Check shapes
#     shape_checker(tensor1, 'batch height width')
#     shape_checker(tensor2, 'batch height width')

#     print("Shape check passed successfully.")

#     # This would raise an error due to shape mismatch
#     # tensor3 = tf.random.uniform((2, 4, 4))
#     # shape_checker(tensor3, 'batch height width')