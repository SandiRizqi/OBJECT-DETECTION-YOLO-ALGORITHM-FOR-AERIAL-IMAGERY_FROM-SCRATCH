import slidingwindow
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 9331200000000
import argparse



args = argparse.ArgumentParser(description='Process annoatate images')

args.add_argument('-i','--img_file', type=str, help='images_file', required=True)
args.add_argument('-s','--chips_size', type=int,help='chips_size', required=True)
args.add_argument('-o','--output_dir', type=str,help='output_directory', required=True)



argumens = args.parse_args()



image = Image.open(argumens.img_file)
image = np.asarray(image)


def compute_windows(numpy_image, patch_size, patch_overlap):

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


# Compute sliding window index
windows = compute_windows(image, argumens.chips_size, 0.5)

for index, window in enumerate(windows):
            # Crop window and predict
            crop = image[windows[index].indices()]

            #Crop is RGB channel order, change to BGR
            #crop = crop[..., ::-1]
            img = Image.fromarray(crop, 'RGB')
            digit = "000000000"
            number = index
            long = len(str(number))
            name = digit[:-long] + str(number)
            img.save( argumens.output_dir + "/" + name + ".jpg")
            
            print(name)