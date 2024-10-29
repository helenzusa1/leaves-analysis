import cv2
import numpy as np
import sys
import os

def detect_contours_and_properties(image, min_area, max_area):
    print("Converting image to HSV...")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    print("Converting image to grayscale...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Applying threshold...")
    _, threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    print("Finding contours...")
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")
    
    contour_data = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if min_area <= contour_area <= max_area:
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
            v_channel = masked_hsv[:, :, 2]
            max_intensity_hsv = np.max(v_channel[mask == 255])
            masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            max_intensity_gray = np.max(masked_gray[mask == 255])
            _, max_intensity_minmaxloc, _, _ = cv2.minMaxLoc(masked_gray, mask=mask)
            M = cv2.moments(contour)
            cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
            contour_data.append({
                'Contour': contour,
                'Center': (cX, cY),
                'Area': contour_area,
                'MaxIntensityHSV': max_intensity_hsv,
                'MaxIntensityGray': max_intensity_gray,
                'MaxIntensityMinMaxLoc': max_intensity_minmaxloc
            })
    
    print("Drawing contours on the image...")
    output_image = image.copy()
    cv2.drawContours(output_image, [c['Contour'] for c in contour_data], -1, (0, 255, 0), 2)
    
    print("Contour detection completed.")
    return contour_data, output_image

def draw_grid(image, num_rows, num_cols, start_center_x, start_center_y, end_center_x, end_center_y, grid_width):
    space_x = (end_center_x - start_center_x) / (num_cols - 1)
    space_y = (end_center_y - start_center_y) / (num_rows - 1)
    grid_data = []
    grid_id = 1
    for row in range(num_rows):
        for col in range(num_cols):
            center_x = int(start_center_x + col * space_x)
            center_y = int(start_center_y + row * space_y)
            
            top_left_x = center_x - grid_width // 2
            top_left_y = center_y - grid_width // 2
            bottom_right_x = center_x + grid_width // 2
            bottom_right_y = center_y + grid_width // 2
            
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
            cv2.putText(image, str(grid_id), (top_left_x + 5, top_left_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            grid_data.append({
                'ID': grid_id,
                'Center': (center_x, center_y),
                'BoundingBox': (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            })
            grid_id += 1
    return grid_data

def assign_contours_to_grids(image, contour_data, grid_data):
    grid_aggregated_data = {}
    for contour in contour_data:
        cX, cY = contour['Center']
        assigned_grid_id = None
        for grid in grid_data:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = grid['BoundingBox']
            if top_left_x <= cX <= bottom_right_x and top_left_y <= cY <= bottom_right_y:
                assigned_grid_id = grid['ID']
                break
        if assigned_grid_id is None:
            min_distance = float('inf')
            for grid in grid_data:
                grid_center_x, grid_center_y = grid['Center']
                distance = np.sqrt((cX - grid_center_x) ** 2 + (cY - grid_center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    assigned_grid_id = grid['ID']
        if assigned_grid_id is not None:
            if assigned_grid_id not in grid_aggregated_data:
                grid_aggregated_data[assigned_grid_id] = {
                    'TotalArea': 0,
                    'MaxIntensityHSV': 0,
                    'MaxIntensityGray': 0,
                    'MaxIntensityMinMaxLoc': 0
                }
            grid_aggregated_data[assigned_grid_id]['TotalArea'] += contour['Area']
            grid_aggregated_data[assigned_grid_id]['MaxIntensityHSV'] = max(grid_aggregated_data[assigned_grid_id]['MaxIntensityHSV'], contour['MaxIntensityHSV'])
            grid_aggregated_data[assigned_grid_id]['MaxIntensityGray'] = max(grid_aggregated_data[assigned_grid_id]['MaxIntensityGray'], contour['MaxIntensityGray'])
            grid_aggregated_data[assigned_grid_id]['MaxIntensityMinMaxLoc'] = max(grid_aggregated_data[assigned_grid_id]['MaxIntensityMinMaxLoc'], contour['MaxIntensityMinMaxLoc'])
            cv2.putText(image, str(assigned_grid_id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return grid_aggregated_data

def select_and_detect_contours(image, min_area, max_area):
    points = []
    clicked_hsv_values = []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            hsv_color = hsv_image[y, x]
            clicked_hsv_values.append(hsv_color)
            print(f"Point selected: ({x}, {y}), HSV value: {hsv_color}")

    cv2.imshow('Select Points', image)
    cv2.setMouseCallback('Select Points', mouse_callback)
    print("Click on the image to select points. Press Enter to finish.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
    cv2.destroyAllWindows()

    if not points:
        print("No points selected.")
        return None, None

    clicked_hsv_values_np = np.array(clicked_hsv_values)
    lower_bound = np.min(clicked_hsv_values_np, axis=0)
    upper_bound = np.max(clicked_hsv_values_np, axis=0)

    print(f"Selected color range: Lower bound: {lower_bound}, Upper bound: {upper_bound}")

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")
    
    contour_data = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if min_area <= contour_area <= max_area:
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
            v_channel = masked_hsv[:, :, 2]
            max_intensity_hsv = np.max(v_channel[mask == 255])
            print(f"Contour area: {contour_area}, Max intensity HSV: {max_intensity_hsv}")
            masked_gray = cv2.bitwise_and(mask, mask, mask=mask)
            max_intensity_gray = np.max(masked_gray[mask == 255])
            _, max_intensity_minmaxloc, _, _ = cv2.minMaxLoc(masked_gray, mask=mask)
            M = cv2.moments(contour)
            cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
            contour_data.append({
                'Contour': contour,
                'Center': (cX, cY),
                'Area': contour_area,
                'MaxIntensityHSV': max_intensity_hsv,
                'MaxIntensityGray': max_intensity_gray,
                'MaxIntensityMinMaxLoc': max_intensity_minmaxloc
            })
    
    output_image = image.copy()
    cv2.drawContours(output_image, [c['Contour'] for c in contour_data], -1, (0, 255, 0), 2)
    
    cv2.imshow('Contours', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return contour_data, output_image

def main():
    if len(sys.argv) < 10:
        print("Usage: python script.py <input_image_path> <min_area> <max_area> <num_rows> <num_cols> <start_center_x> <start_center_y> <end_center_x> <end_center_y> <grid_width>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    min_area = float(sys.argv[2])
    max_area = float(sys.argv[3])
    num_rows = int(sys.argv[4])
    num_cols = int(sys.argv[5])
    start_center_x = int(sys.argv[6])
    start_center_y = int(sys.argv[7])
    end_center_x = int(sys.argv[8])
    end_center_y = int(sys.argv[9])
    grid_width = int(sys.argv[10])

    input_dir = os.path.dirname(input_image_path)
    input_base_name = os.path.basename(input_image_path).split('.')
    output_image_path = os.path.join(input_dir, f"{input_base_name}_output_image.png")
    output_file_path = os.path.join(input_dir, f"{input_base_name}_output.txt")

    image = cv2.imread(input_image_path)

    print("Choose detection method:")
    print("1. Auto detect contours")
    print("2. User select color for contour detection")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        contour_data, output_image = detect_contours_and_properties(image, min_area, max_area)
    elif choice == '2':
        contour_data, output_image = select_and_detect_contours(image, min_area, max_area)
        if contour_data is None or output_image is None:
            print("No color range selected. Exiting.")
            sys.exit(1)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Draw grid on the output image
    grid_data = draw_grid(output_image, num_rows, num_cols, start_center_x, start_center_y, end_center_x, end_center_y, grid_width)
    grid_aggregated_data = assign_contours_to_grids(output_image, contour_data, grid_data)

    with open(output_file_path, 'w') as file:
        file.write("input_image_path,min_area,max_area,num_rows,num_cols,start_center_x,start_center_y,end_center_x,end_center_y,grid_width\n")
        file.write(f'Arguments: {sys.argv[1:]}\n')
        for grid_id in range(1, len(grid_data) + 1):
            if grid_id in grid_aggregated_data:
                data = grid_aggregated_data[grid_id]
                file.write(f'Grid {grid_id}: Total Area = {data["TotalArea"]}, Max intensity (HSV) = {data["MaxIntensityHSV"]}, Max intensity (Gray) = {data["MaxIntensityGray"]}, Max intensity (minMaxLoc) = {data["MaxIntensityMinMaxLoc"]}\n')
            else:
                file.write(f'Grid {grid_id}:\n')

    cv2.imwrite(output_image_path, output_image)
    # cv2.imshow('Image with Contours and Grid', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()