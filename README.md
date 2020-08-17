# Annotate_Images
Place all of the images to be annotated in a folder.
Keep the annotate.py file outside of this folder.



Run the following commands:
pip install requirements.txt

python annotate.py --dir *image_folder*



Instructions on how to label images

Left click on a point of the image, hold and drag the cursor to another point of the image and then release, a bounding box will be created in the region of interest.

Press the j/J,k/K keys to rotate the bounding box

Press the w/W,a/A,s/S,d/D keys to translate the bounding box

Press the h/H key to undo a bounding box

Press the c/C key to confirm a bounding box (you can not undo this box now)

Press the u,U,i,I,o,O,p,P keys to resize the bounding box

Press the l/L key to move onto the next image (this saves an xml file of the previous image in the image directory)




