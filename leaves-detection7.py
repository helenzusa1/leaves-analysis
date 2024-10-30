#This one just draw contour without picking color
#use grid image and object image separately
import cv2
import numpy as np
import sys
import pickle

if len(sys.argv) < 8:
    print("Usage: python script.py <grid_image> <object_image> <output_image> <contour_text> <grid_threshold_size> <min_object_size> <max_object_size>")
    sys.exit(1)

# Open the output file from argv
output_file = open(sys.argv[4], 'w')

#Step 1: Detect Grids in the Grid Image
# Load the grid image
grid_image = cv2.imread(sys.argv[1])

# Convert the image to grayscale
gray_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply thresholding
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Use morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Detect edges
edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)

# Find contours for grids
grid_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define reference shapes (hexagon and square)
hexagon = np.array([[50, 0], [100, 25], [100, 75], [50, 100], [0, 75], [0, 25]], dtype=np.int32).reshape((-1, 1, 2))
square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32).reshape((-1, 1, 2))

# Match contours to reference shapes
grid_coordinates = []
grid_id = 0
output_file.write('Grid\n')
for contour in grid_contours:
    if cv2.contourArea(contour) < int(sys.argv[5]):  # Adjust the threshold value as needed
        continue
    
    match_hexagon = cv2.matchShapes(contour, hexagon, cv2.CONTOURS_MATCH_I1, 0.0)
    match_square = cv2.matchShapes(contour, square, cv2.CONTOURS_MATCH_I1, 0.0)
    
    if match_hexagon < 0.2 or match_square < 0.2:  # Adjust the match threshold as needed
        grid_coordinates.append(contour)
        grid_id += 1
        x, y, w, h = cv2.boundingRect(contour)
        size = cv2.contourArea(contour)
        points = contour.tolist()
        print(f'Grid {grid_id}: ({x}, {y}), ({x+w}, {y+h}), Size: {size}')
        output_file.write(f'Grid {grid_id}: ({x}, {y}), ({x+w}, {y+h}), Size: {size}\n')
        output_file.write(f'{grid_id}, {points}\n')  # Write grid number and points to the file
        cv2.drawContours(grid_image, [contour], -1, (0, 0, 255), 2)  # Draw grid in red


# Save grid coordinates to a file using pickle
with open('grid_coordinates.pkl', 'wb') as f:
    pickle.dump(grid_coordinates, f)


# Step 2: Detect objects of a specific color and find contours

object_image = cv2.imread(sys.argv[2])

gray_object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
blurred_object_image = cv2.GaussianBlur(gray_object_image, (5, 5), 0)
_, binary_object_image = cv2.threshold(blurred_object_image, 100, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
cleaned_object_image = cv2.morphologyEx(binary_object_image, cv2.MORPH_OPEN, kernel, iterations=2)
edges_object_image = cv2.Canny(cleaned_object_image, 50, 150, apertureSize=3)
object_contours, _ = cv2.findContours(edges_object_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_object_size = int(sys.argv[6])
max_object_size = int(sys.argv[7])
output_file.write('Object Contours\n')
output_file.write('Object Number, Points\n')


def detect_objects():
    object_id = 0
    for contour in object_contours:
        size = cv2.contourArea(contour)
        if size < min_object_size or size > max_object_size:
            continue
        object_id += 1
        x, y, w, h = cv2.boundingRect(contour)
        points = contour.tolist()
        print(f'Object {object_id}: ({x}, {y}), ({x+w}, {y+h}), Size: {size}')
        output_file.write(f'Object {object_id}: ({x}, {y}), ({x+w}, {y+h}), Size: {size}\n')
        output_file.write(f'{object_id}, {points}\n')
        cv2.drawContours(object_image, [contour], -1, (0, 255, 0), 2)
        check_object_in_grid(contour, object_id)

# Step 3: Check if object contour belongs to any grid contour

# Load grid coordinates from the file using pickle
with open('grid_coordinates.pkl', 'rb') as f:
    grid_coordinates = pickle.load(f)

def unpack_point(point):
    flat_list = np.array(point).flatten()
    if len(flat_list) == 2:
        return tuple(map(int, flat_list))
    else:
        raise ValueError("Input does not have the expected structure [[x y]]")

grid_objects_inside = {}
grid_objects_overlap = {}

def check_object_in_grid(object_contour, object_id):
    for grid_id, grid_contour in enumerate(grid_coordinates, start=1):
        inside_count = 0
        overlapping_points = []
        for point in object_contour:
            point = unpack_point(point)
            if cv2.pointPolygonTest(grid_contour, point, False) >= 0:
                inside_count += 1
                overlapping_points.append(point)
        if inside_count == len(object_contour):
            output_file.write(f'Object {object_id} is fully inside Grid {grid_id}\n')
            print(f'Object {object_id} is fully inside Grid {grid_id}')
            if grid_id not in grid_objects_inside:
                grid_objects_inside[grid_id] = []
            grid_objects_inside[grid_id].append((object_id, cv2.contourArea(object_contour)))
        elif inside_count > 0:
            output_file.write(f'Object {object_id} overlaps with Grid {grid_id}\n')
            print(f'Object {object_id} overlaps with Grid {grid_id}')
            cv2.drawContours(object_image, [object_contour], -1, (255, 0, 0), 2)
            if grid_id not in grid_objects_overlap:
                grid_objects_overlap[grid_id] = []
            grid_objects_overlap[grid_id].append((object_id, cv2.contourArea(object_contour)))

    output_file.write(f'Object {object_id} is not inside any grid\n')
    print(f'Object {object_id} is not inside any grid')

detect_objects()

# Draw grid contours in red
for grid_contour in grid_coordinates:
    cv2.drawContours(object_image, [grid_contour], -1, (0, 0, 255), 2)

# Save the processed image as a PNG file
cv2.imwrite(sys.argv[3], object_image)

# Print and write consolidated results at the end
output_file.write('\nConsolidated Results\n')

output_file.write('Objects completely inside the grids:\n')
print('Objects completely inside the grids:')
for grid_id, objects in grid_objects_inside.items():
    total_area = sum(area for _, area in objects)
    output_file.write(f'Grid {grid_id} includes these objects with total area size {total_area}: ')
    for obj_id, area in objects:
        output_file.write(f'object {obj_id}, area {area}; ')
    output_file.write('\n')
    print()

output_file.write('\nObjects overlapping with the grids:\n')
print('Objects overlapping with the grids:')
for grid_id, objects in grid_objects_overlap.items():
    total_area = sum(area for _, area in objects)
    output_file.write(f'Grid {grid_id} overlaps with these objects with total area size {total_area}: ')
    for obj_id, area in objects:
        output_file.write(f'object {obj_id}, area {area}; ')
    output_file.write('\n')
    print()

# Close the output file
output_file.close()


"""
Certainly! The thresholding parameters in the code are used to convert the grayscale image to a binary image. This step is crucial for detecting the grid. Here are the lines and numbers related to thresholding in the code:

# Apply thresholding
_, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
In this line, the cv2.threshold function is used to apply thresholding. The parameters are as follows:

blurred: The input image (grayscale and blurred).
128: The threshold value. Pixels with intensity values above this threshold will be set to 0 (black), and pixels with intensity values below this threshold will be set to 255 (white).
255: The maximum value to use with the cv2.THRESH_BINARY_INV thresholding type.
cv2.THRESH_BINARY_INV: The type of thresholding. In this case, it is an inverted binary threshold, meaning that the binary image will have inverted pixel values compared to a regular binary threshold.
You can experiment with different threshold values to see if it improves the grid detection. For example, you can try changing the threshold value from 128 to 100 or 150:

# Apply thresholding with a different threshold value
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

The line lower_color = np.array([hsv_color - 10, 100, 100]) is creating a NumPy array that represents the lower bound of the HSV color range for color detection. Here's a breakdown of what it does:

hsv_color - 10: This subtracts 10 from the hue value of the selected color. The hue value is the first element of the hsv_color array. This creates a range around the selected hue to account for slight variations in the color.
100: This is the saturation value. It sets the lower bound for saturation to 100.
100: This is the value (brightness) component. It sets the lower bound for brightness to 100.
So, the resulting lower_color array will be something like [hue - 10, 100, 100], where hue is the hue value of the selected color. This array is used to create a mask that identifies all pixels in the image that fall within this lower bound of the HSV color range.

"""