import sys
import psutil
import numpy as np
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale

# Load your own image
image_path = sys.argv[1]
image = io.imread(image_path, as_gray=True)

# Rescale the image based on the second command-line argument
scale_factor = float(sys.argv[2])
image = rescale(image, scale_factor, anti_aliasing=True)

# Register the image against itself; the answer should
# always be (0, 0), but that's fine, right now we just care
# about memory usage.
shift, error, diffphase = phase_cross_correlation(image, image)

# Print the size of the image in kilopixels
print("Image size (Kilo pixels):", image.size / 1024)

# Print the peak memory usage in MiB
process = psutil.Process()
print("Peak memory (MiB):", int(process.memory_info().peak_wset / 1024 / 1024))