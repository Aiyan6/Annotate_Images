# Annotate_Images
Place all of the images to be annotated in a folder along with their xml labels.
Keep the annotate.py file outside of this folder.



Run the following commands:

pip install requirements.txt

python annotate.py --dir *image_folder*



Instructions on how to label images

A bounding box will already be shown on the image which has to be adjusted further

* Press the j/J,k/K keys to rotate the bounding box

* Press the w/W,a/A,s/S,d/D keys to translate the bounding box

* Press the c/C key to confirm a bounding box, if the image has more than one bounding box, another bounding box will be shown on the image one by one

* Press the u,U,i,I,o,O,p,P keys to resize the bounding box

* Press the l/L key to move onto the next image (this saves a txt file of the previous image in the image directory)




