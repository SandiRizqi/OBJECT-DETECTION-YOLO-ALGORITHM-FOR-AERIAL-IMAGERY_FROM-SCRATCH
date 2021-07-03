import tensorflow
import argparse



args = argparse.ArgumentParser(description='Process Training model')
args.add_argument('-i','--img_dir', type=str, help='images_directory', required=True)
args.add_argument('-m','--model_dir', type=str, help='model_directory', required=True)
args.add_argument('-s','--resized_size', type=int,help='chips_size', required=True)
args.add_argument('-a','--annotations', type=int,help='chips_size', required=True)

argumens = args.parse_args()

#Create Config
class config:
    annotations_file = argumens.annotations
    image_dir = argumens.img_dir + '/'
    image_size = 1000
    resized_size = argumens.resized_size
    train_ratio = 0.8
    checkpoint = argumens.model_dir + '/'
    saved_model = argumens.model_dir + "/object_detection_model.h5'
