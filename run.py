import cv2, os
import numpy as np
import configparser as parser
from glob import glob
from time import sleep
from common.config import IMAGE_FILE_EXTESION, IMAGE_RESOLUTION, IMAGE_UPSCALE_AFTER_PROGRESS, IMAGE_UPSCALE_AFTER_PROGRESS_RESOLUTION, IMAGE_EQUALIZE, INTERPOLATE_STEPS, INTERPOLATE_WIDTH, INTERPOLATE_DIAGONAL, IO_SOURCE_FOLDER, IO_FILE_OUTPUT_NAME

config = parser.ConfigParser()

def load_images_from_folder(folder):
    image_files = sorted(glob(os.path.join(folder, IMAGE_FILE_EXTESION)))
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file)
        img = cv2.resize(img, dsize=(int(IMAGE_RESOLUTION), int(IMAGE_RESOLUTION)))
        print(img_file + " Loaded")
        images.append(img)
    return images

def average_images(images):
    sum_image = np.array(images[0], dtype=np.float32)

    for i in range(1, len(images)):
        sum_image += np.array(images[i], dtype=np.float32)

    average_image = np.divide(sum_image, len(images))

    result_image = np.array(average_image, dtype=np.uint8)
    cv2.imwrite("outputs/average.png", result_image)
    print("Average image Created")
    return result_image

def equalize_histogram(image):
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        ycrcb = cv2.merge(channels)
        result_image_hitogram = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result_image_hitogram = cv2.equalizeHist(image)
    return result_image_hitogram

def create_mask_from_images(images):
    output = images[0]
    for i, image in enumerate(images[1:]):
        result_mask = cv2.add(output, image)
        
    result_mask = np.uint8(result_mask)
    cv2.imwrite("outputs/mask.png", result_mask)
    print("Mask image Created")
    

def save_image(image, output_path): 
    cv2.imwrite(output_path, image)

def interpolate_pixels(image, num_steps=int(INTERPOLATE_STEPS), interpolation_width=int(INTERPOLATE_WIDTH)):
    height, width = image.shape[:2]
    result = np.copy(image)

    for y in range(height):
        print("Interpolating Horizontal " + str(image.shape[:2]) + " / " + str(y))
        for x in range(width - interpolation_width):
            left_color = image[y, x].astype(np.float32)
            right_color = image[y, x + interpolation_width].astype(np.float32)

            for step in range(1, num_steps + 1):
                alpha = step / (num_steps + 1)
                interpolated_color = (1 - alpha) * left_color + alpha * right_color
                result[y, x + step % interpolation_width] = (result[y, x + step % interpolation_width].astype(np.float32) * (1 - alpha) + interpolated_color * alpha).astype(np.uint8)

    for x in range(width):
        print("Interpolating Vertical " + str(image.shape[:2]) + " / " + str(x))
        for y in range(height - interpolation_width):
            top_color = result[y, x].astype(np.float32)
            bottom_color = result[y + interpolation_width, x].astype(np.float32)

            for step in range(1, num_steps + 1):
                alpha = step / (num_steps + 1)
                interpolated_color = (1 - alpha) * top_color + alpha * bottom_color
                result[y + step % interpolation_width, x] = (result[y + step % interpolation_width, x].astype(np.float32) * (1 - alpha) + interpolated_color * alpha).astype(np.uint8)

    if INTERPOLATE_DIAGONAL == 'True':
        print("Diagonal Interpolate = True")
        for y in range(height - interpolation_width):
            print("Interpolating Diagonal X " + str(image.shape[:2]) + " / " + str(y))
            for x in range(width - interpolation_width):
                top_left_color = result[y, x].astype(np.float32)
                bottom_right_color = result[y + interpolation_width, x + interpolation_width].astype(np.float32)

                for step in range(1, num_steps + 1):
                    alpha = step / (num_steps + 1)
                    interpolated_color = (1 - alpha) * top_left_color + alpha * bottom_right_color
                    result[y + step % interpolation_width, x + step % interpolation_width] = (result[y + step % interpolation_width, x + step % interpolation_width].astype(np.float32) * (1 - alpha) + interpolated_color * alpha).astype(np.uint8)
               
        for y in range(interpolation_width, height):
            print("Interpolating Diagonal Y " + str(image.shape[:2]) + " / " + str(y))
            for x in range(width - interpolation_width):
                bottom_left_color = result[y, x].astype(np.float32)
                top_right_color = result[y - interpolation_width, x + interpolation_width].astype(np.float32)

                for step in range(1, num_steps + 1):
                    alpha = step / (num_steps + 1)
                    interpolated_color = (1 - alpha) * bottom_left_color + alpha * top_right_color
                    result[y - step % interpolation_width, x + step % interpolation_width] = (result[y - step % interpolation_width, x + step % interpolation_width].astype(np.float32) * (1 - alpha) + interpolated_color * alpha).astype(np.uint8)
    else:
        print("Diagonal Interpolate = False")

    return result

# I/O Descriptions
input_folder = IO_SOURCE_FOLDER
output_file = IO_FILE_OUTPUT_NAME

images = load_images_from_folder(input_folder)
sleep(0.1)
mask = create_mask_from_images(images)
sleep(0.1)

mask_image = cv2.imread("outputs/mask.png", cv2.IMREAD_GRAYSCALE)
average_image = average_images(images)
sleep(0.1)


image_path = "outputs/average.png"
image_average = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

sdf_image = interpolate_pixels(image_average)

if IMAGE_EQUALIZE == 'True':
    sdf_image = equalize_histogram(sdf_image)
    sdf_image = interpolate_pixels(sdf_image, num_steps=1, interpolation_width=2)

if IMAGE_UPSCALE_AFTER_PROGRESS == 'True':
    sdf_image = cv2.resize(sdf_image, dsize=(int(IMAGE_UPSCALE_AFTER_PROGRESS_RESOLUTION),int(IMAGE_UPSCALE_AFTER_PROGRESS_RESOLUTION)))
    print("Progress END with resize")
    save_image(sdf_image, output_file)
    sleep(3)
else:
    print("Progress END")
    save_image(sdf_image, output_file)
    sleep(3)

cv2.waitKey(0)
cv2.destroyAllWindows()