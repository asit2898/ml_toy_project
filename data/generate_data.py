from typing import NamedTuple, Optional, Tuple, Generator
from data_utils import noisy_circle, show_circle, generate_examples
import time
import os
import numpy as np
import argparse


def create_eval_datasets(
    data_save_dir: str, n_valid: int = 20000, n_test: int = 10000
) -> None:
    """
	Training samples are generated on the fly. However for validation and testing we need a fixed dataset
	These are smaller datasets so should be fine generating at once

	:param data_save_dir: directory to save the datasets to
	:param n_valid: size of the validation dataset
	:param n_test: size of the test dataset
	"""

    # Create data_save_dir if it does not exist
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # Generate the validation and test datasets
    print(
        f"Generating validation and test data with {n_valid} and {n_test} samples respectively"
    )
    validation_data_images, validation_data_circles = generate_dataset(n_valid)
    test_data_images, test_data_circles = generate_dataset(n_test)

    # Save the datasets
    print(f"Saving validation and test data in {data_save_dir}")
    np.save(
        os.path.join(data_save_dir, "validation_data_images.npy"),
        validation_data_images,
    )
    np.save(
        os.path.join(data_save_dir, "validation_data_circles.npy"),
        validation_data_circles,
    )
    np.save(os.path.join(data_save_dir, "test_data_images.npy"), test_data_images)
    np.save(os.path.join(data_save_dir, "test_data_circles.npy"), test_data_circles)

    return


def generate_dataset(
    length: int,
    img_size: int = 128,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    noise_level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
	Generate a dataset of specified length made up of noisy circles 

	:param length: number of samples in the dataset
	:param img_size: size of the image
	:param min_radius: minimum radius of the circles
	:param max_radius: maximum radius of the circles
	:param noise_level: noise level for the dataset
	"""

    # Generate the dataset
    images = []
    circles = []

    params = f"{length=}, {noise_level=}, {img_size=}, {min_radius=}, {max_radius=}"
    print(f"Using parameters: {params}")

    for i in range(length):
        image, circle = next(
            generate_examples(noise_level, img_size, min_radius, max_radius)
        )
        images.append(image)

        row = circle.row
        col = circle.col
        radius = circle.radius
        circles.append(np.array([row, col, radius]))

    # Convert to numpy arrays
    images = np.array(images)
    circles = np.array(circles)

    return (images, circles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_save_dir",
        type=str,
        default="datasets",
        help="directory to save the datasets to",
    )
    parser.add_argument(
        "--n_valid", type=int, default=20000, help="size of the validation dataset"
    )
    parser.add_argument(
        "--n_test", type=int, default=10000, help="size of the test dataset"
    )

    args = parser.parse_args()

    create_eval_datasets(
        data_save_dir=args.data_save_dir, n_valid=args.n_valid, n_test=args.n_test
    )
