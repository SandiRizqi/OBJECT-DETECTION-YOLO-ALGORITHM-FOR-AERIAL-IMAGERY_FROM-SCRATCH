# OBJECT-DETECTION-YOLO-ALGORITHM-FOR-AERIAL-IMAGERY
This project is implementation of Object detection YOLO algorithm to detect object in aerial imagery or satellite imagery data

To implement this project you need to make environment and install some essential packages:
- Tensorflow
- Keras
- slidingwindow
- matplotlib
- shapely
- rasterio
- geopandas

or just install from Requirement.txt 

by pip install -r Requirement.txt

this project is also able to run in google colab



### MAKE SAMPLES :
- Just run make_chip_samples.py to your imagery data to make sample that you can annotate

example: python make_chip_samples.py -i xxx\xxx\xx\xxx.jpg -s 1000 -o xx\xx\xx\xx\

- to make and annotate your samples you need to install Labelimg
- Annotate your sample and save in VOC format


### CONVERT XML TO CSV

- run xml_to_csv.py to make csv annotations samples


example: python xml_to_csv.py -i C:\DeepTree\samplemanual_tm_tbm\images -l C:\DeepTree\samplemanual_tm_tbm\labels -s 1000 -o C:\DeepTree\samplemanual_tm_tbm



### TRAIN YOLO MODEL

- run train_yolo.py to train and save your model

example : python train_yolo.py -a annotations.csv -i Dataset/images -s 250 -m Model -e 10

you can train your model eather in you local machine or cloud service like google colab, aws, or gcp

![image](https://user-images.githubusercontent.com/26809652/124358923-c4a98000-dc4c-11eb-8909-042ea7a30000.png)




### DETECT OBJECT IN AERIAL IMAGERY
![image](https://user-images.githubusercontent.com/26809652/124380578-9d53c100-dce7-11eb-93a9-d5a269edbbd6.png)
