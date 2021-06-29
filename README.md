# OBJECT-DETECTION-YOLO-ALGORITHM-FOR-AERIAL-IMAGERY
This project is implementation of Object detection YOLO algorithm to detect object in aerial imagery or satellite imagery data

To implement this project you need to meke envirinment and install some essential packages:
- Tensorflow
- Keras
- slidingwindow
- matplotlib
- shapely

or just install from Requirement.txt 

by pip install -r Requirement.txt

this project also able to run in google colab



MAKE SAMPLES :
- Just run make_chip_samples.py to your imagery data to make sample that you can annotate

example: python make_chip_samples.py -i xxx\xxx\xx\xxx.jpg -s 1000 -o xx\xx\xx\xx\

- to make and annotate you samples you need to install Labelimg
- Annotate your sample and save in VOC format


CONVERT XML TO CSV

- run xml_to_csv.py to make csv annotations samples


example: python xml_to_csv.py -i C:\DeepTree\samplemanual_tm_tbm\images -l C:\DeepTree\samplemanual_tm_tbm\labels -s 1000 -o C:\DeepTree\samplemanual_tm_tbm
