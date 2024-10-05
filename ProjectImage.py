import sys
import numpy as np
from PIL import Image


def project_pixel_onto_plane(pixel: np.array, normal: np.array, bias: float) -> np.array:
    """
    Projects pixel vector onto plane specified by normal and bias
    """
    A = np.array([[normal[1], -normal[0], 0],
                  [0, normal[2], -normal[1]],
                  [normal[0], normal[1], normal[2]]])
    b = np.array([normal[1] * pixel[0] - normal[0] * pixel[1], normal[2] * pixel[1] - normal[1] * pixel[2], bias])
    return np.linalg.solve(A, b)


def project_pixel_onto_line(pixel: np.array, line_vec: np.array, point: np.array) -> np.array:
    bias = np.dot(pixel, line_vec)
    normal = line_vec

    return project_pixel_onto_plane(point, normal, bias)

if __name__ == '__main__':
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    colors_path = sys.argv[3]

    input_image = np.asarray(Image.open(input_image_path))

    if 'txt' not in colors_path:
        print('Path to color coordinates must contains .txt part!')
        exit()

    with open(colors_path, 'r') as f:
        color_coordinates = [int(numeric_string) for numeric_string in f.read().split(' ')]

    first_color = np.array(color_coordinates[0:3])
    second_color = np.array(color_coordinates[3:6])
    third_color = np.array(color_coordinates[6:9])

    if (not np.array_equal(first_color, second_color) and not np.array_equal(first_color, third_color)
            and not np.array_equal(second_color, third_color)): # make projection onto plane
        first2second = second_color - first_color
        first2third = third_color - first_color

        cross_product = np.cross(first2second, first2third)
        normal = cross_product / np.linalg.norm(cross_product)
        bias = np.dot(first_color, normal)


        def project_pixel(pixel: np.array):
            return project_pixel_onto_plane(pixel, normal, bias)


        output_image = np.apply_along_axis(project_pixel, 2, input_image).round().astype(np.uint8)
        image_to_save = Image.fromarray(output_image)
        image_to_save.save(output_image_path)
    elif np.array_equal(first_color, second_color) and np.array_equal(second_color, third_color):
        print('At least 2 color must be different!')
    else: # make projection onto line
        if np.array_equal(first_color, second_color):
            first_color = third_color

        line_vec = second_color - first_color
        point = first_color

        def project_pixel(pixel: np.array):
            return project_pixel_onto_line(pixel, line_vec, point)

        output_image = np.apply_along_axis(project_pixel, 2, input_image).round().astype(np.uint8)
        image_to_save = Image.fromarray(output_image)
        image_to_save.save(output_image_path)



