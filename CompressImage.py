import sys
import numpy as np
from PIL import Image

from ProjectImage import project_pixel_onto_plane, project_pixel_onto_line

if __name__ == '__main__':
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    colors_path = sys.argv[3]
    compression_ratio = float((sys.argv[4]))

    input_image = np.asarray(Image.open(input_image_path))

    if 'txt' not in colors_path:
        print('Path to color coordinates must contains .txt part!')
        exit()

    with open(colors_path, 'r') as f:
        color_coordinates = [int(numeric_string) for numeric_string in f.read().split(' ')]

    first_color = np.array(color_coordinates[0:3])
    second_color = np.array(color_coordinates[3:6])
    third_color = np.array(color_coordinates[6:9])

    def shift_image_towards_color(image, target_color, shift_fraction):
        image = image.astype(np.float32)
        target_color = np.array(target_color, dtype=np.float32)

        shift_vector = target_color - image
        shifted_image = image + shift_fraction * shift_vector
        shifted_image = np.clip(shifted_image, 0, 255)
        return shifted_image.astype(np.uint8)

    def shift_image_towards_line(image, first_color, second_color, shift_fraction):
        line_vec = second_color - first_color
        point = first_color

        def project_pixel(pixel: np.array):
            return project_pixel_onto_line(pixel, line_vec, point)

        projected_image = np.apply_along_axis(project_pixel, 2, input_image).round().astype(np.uint8)
        shift_vector = projected_image - image
        shifted_image = image + shift_fraction * shift_vector
        shifted_image = np.clip(shifted_image, 0, 255)
        return shifted_image.astype(np.uint8)

    def shift_image_towards_plane(image, first_color, second_color, third_color, shift_fraction):
        first2second = second_color - first_color
        first2third = third_color - first_color

        cross_product = np.cross(first2second, first2third)
        normal = cross_product / np.linalg.norm(cross_product)
        bias = np.dot(first_color, normal)

        def project_pixel(pixel: np.array):
            return project_pixel_onto_plane(pixel, normal, bias)

        image_plane_projection = np.apply_along_axis(project_pixel, 2, input_image).round().astype(np.uint8)
        shift_vector = image_plane_projection - image
        shifted_image = image + shift_fraction * shift_vector
        shifted_image = np.clip(shifted_image, 0, 255)
        return shifted_image.astype(np.uint8)


    if (not np.array_equal(first_color, second_color) and not np.array_equal(first_color, third_color)
            and not np.array_equal(second_color, third_color)): # make projection onto plane
        output_image = shift_image_towards_plane(input_image, first_color, second_color, third_color, compression_ratio)
        image_to_save = Image.fromarray(output_image)
        image_to_save.save(output_image_path)
    elif np.array_equal(first_color, second_color) and np.array_equal(second_color, third_color):
        output_image = shift_image_towards_color(input_image, first_color, compression_ratio)
        image_to_save = Image.fromarray(output_image)
        image_to_save.save(output_image_path)
    else: # make projection onto line
        if np.array_equal(first_color, second_color):
            first_color = third_color
        output_image = shift_image_towards_line(input_image, first_color, second_color, compression_ratio)
        image_to_save = Image.fromarray(output_image)
        image_to_save.save(output_image_path)

    exit()
