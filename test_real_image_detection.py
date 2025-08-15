import unittest
import cv2
import numpy as np
import requests
import os
from image_processor import ScoreImageProcessor

class TestRealImageDetection(unittest.TestCase):
    """
    Tests the image processor with a real-world image provided by the user.
    """

    def setUp(self):
        self.processor = ScoreImageProcessor()
        # Using the original Imgur URL again.
        self.image_url = "https://i.imgur.com/nipNHSe.png"
        self.image_path = "test_data/score.png" # Using the new name provided by user
        self.output_path = "test_output/real_image_detection_debug.png"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Download the image
        print(f"Attempting to download image from {self.image_url}...")
        try:
            response = requests.get(self.image_url, timeout=20)
            response.raise_for_status()
            with open(self.image_path, "wb") as f:
                f.write(response.content)
            print("Image downloaded successfully.")
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to download test image: {e}")

    def test_detect_regions_on_real_image(self):
        """
        Ensures that the screen and regions are detected on the real image.
        """
        # Load the image
        image = cv2.imread(self.image_path)
        self.assertIsNotNone(image, f"Failed to load the downloaded image from {self.image_path}.")

        # Run detection
        regions = self.processor.detect_score_regions(image)

        # Assert that regions were found
        self.assertIsNotNone(regions, "detect_score_regions should not return None.")
        self.assertIsInstance(regions, dict, "Regions should be a dictionary.")
        self.assertGreater(len(regions), 0, "At least one region should be detected.")
        print(f"Successfully detected {len(regions)} regions.")

        # Save debug image for manual verification
        debug_image, _ = self.processor.debug_detection(image)
        cv2.imwrite(self.output_path, debug_image)
        print(f"Debug image saved to {self.output_path}")

if __name__ == "__main__":
    unittest.main()
