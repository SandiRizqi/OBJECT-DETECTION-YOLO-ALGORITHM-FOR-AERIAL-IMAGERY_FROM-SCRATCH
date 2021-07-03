from PIL import Image
import os
import xml.etree.ElementTree as ET
import os
import pandas as pd
import argparse



args = argparse.ArgumentParser(description='Process annoatate images')

args.add_argument('-i','--img_dir', type=str, help='images_directory', required=True)
args.add_argument('-l','--label_dir', type=str,help='labels_directory', required=True)
args.add_argument('-o','--output_dir', type=str,help='output_directory', required=True)




argumens = args.parse_args()




class config:
    imagedir = argumens.img_dir
    labeldir = argumens.label_dir
    annotation_dir = argumens.output_dir
    
        




images = os.listdir(config.imagedir)
labels = os.listdir(config.labeldir)




def parse_annotation(ann_dir, img_dir, labels=[], x_ratio =1, y_ratio = 1):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text))*x_ratio)
                            if 'ymin'  in dim.tag:
                                obj['ymin'] = int(round(float(dim.text))*y_ratio)
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text))*x_ratio)
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text))*y_ratio)

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels



LABELS = ['1'] # array containing labels. Can be more than one.
train_image_folder = config.imagedir + "/"
train_annot_folder = config.labeldir + "/"

train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS,)

labels_ids = {'1': 'palm'}




data = []
for img_number in range(0,len(train_imgs)):
    image = train_imgs[img_number]
    im_name = image['filename'].split('/')[-1]
    objects = image['object']

    for objs in objects:
        xmin = objs['xmin']
        ymin = objs['ymin']
        xmax = objs['xmax']
        ymax = objs['ymax']
        c_id = labels_ids[objs['name']]
        data.append((im_name,xmin, ymin, xmax, ymax, [xmin, ymin, xmax-xmin, ymax-ymin],c_id))

    
data = pd.DataFrame(data, columns=('image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'yolo_bbox','label'))


data.to_csv(config.annotation_dir + "/annotations.csv", index=False)

