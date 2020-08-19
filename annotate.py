# import the necessary packages
import argparse
import cv2
import numpy as np
import math
import os
import shutil
import xml.etree.ElementTree as ET
import pathlib


refPt = info =  []
rotated_pts = A = B = C = D = center = np.array([])
center  = theta = temp = counter = w = h =  width = height = 0
translate_x = 0


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def parse(xmlFile):
	# create element tree object 
	tree = ET.parse(xmlFile) 

	# get root element 
	root = tree.getroot() 

	# create empty list for news items 

	# iterate news items
	boxes = [] 
	for bndbox in root.findall('object/bndbox'):
		box = []

	
			 
	# iterate child elements of item 
		for child in bndbox: 
			box.append(int(float(child.text)))

		A = np.array([box[0], box[1]])
		B = np.array([box[2], box[1]])
		C = np.array([box[2], box[3]])
		D = np.array([box[0], box[3]])

		box = [A,B,C,D]
		boxes.append(box)

	return boxes



def convert_theta(theta):
	if theta<0:
		theta = math.pi + theta

	return theta * (180/math.pi)


def make_class(theta):
	rem = theta % 20

	if rem < 10:
		theta = int((theta / 10 * 10) - rem)
	else:
		theta = (int((theta + 10) / 10) * 10)

	return theta



def click_and_crop(box):
	# grab references to the global variables
	global rotated_pts, A , B , C , D , center, counter, w, h, theta, width, height, info, temp
	

	theta = 0
	
	
	counter+=1
	print(counter)
	
	# calculate the coordinates of all the four corners of the bounding box
	A = box[0]
	B = box[1]
	C = box[2]
	D = box[3]

	# calculate the coordinates of the center of the bounding box
	center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
	print(center)

	


    # calculate the width and height of the bounding box
	w = abs(A[0] - B[0])
	h = abs(A[1] - D[1]) 

    # display the bounding box on the image
	cv2.rectangle(image, (A[0],A[1]), (C[0],C[1]), (0, 0, 255), 3)
	cv2.imshow("image", image)
		
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dir", required=True, help="Path to the folder")
args = vars(ap.parse_args())

# list containing all the image names in the specified directory
image_names = os.listdir(args["dir"])
label_names = os.listdir(args["dir"])

filtered_image_names = []
filtered_label_names = []

for i in image_names:
	if i[-1]=='g':
		filtered_image_names.append(i)

for i in label_names:
	if i[-1]=='l':
		filtered_label_names.append(i)

image_names = filtered_image_names
label_names = filtered_label_names

for i in range(len(image_names)-1):
	if (image_names[i])[0:len(image_names[i])-6] == (image_names[i+1])[0:len(image_names[i+1])-6]:
		print('yes')


print(len(image_names))
print(len(label_names))

tuples = []

for i in range(len(image_names)):
	tuples.append((image_names[i], label_names[i]))



for img,anno in tuples:
	path_img = args["dir"] + "/" + img
	path_anno = args["dir"] + "/" + anno
	

	image = cv2.imread(path_img)
	height, width, channels = image.shape
	
	clone = image.copy()
	cv2.namedWindow("image")
	
	boxes = parse(path_anno)
	length = len(boxes)

	for box in boxes:
		click_and_crop(box)




		while True:
		# display the image and wait for a keypress
			cv2.imshow("image", image)
			key = cv2.waitKey(1) & 0xFF
			
			
		# if the 'h' key is pressed, remove the latest bounding box 
			if key == ord("h") or key == ord('H'):

				undo = True
				if counter>1:
					image = temp.copy()
				else: 
					image = clone.copy()

				
				A = B = C = D = center = np.array([])
				counter-=1
				
				if counter<0:
					counter = 0


		# if the 'l' key is pressed, move on to the next image
			elif key == ord("l") or key == ord('L'):

				s = ''
				fileName = img.replace('jpg','txt')
				f = open(fileName, "w")
				for x in info:
					for i in x:
						s += str(i) + " "
					s = s[0:len(s)-1] + "\n"
				s = s[0:len(s)-1]

				f.write(s)	
				f.close()	
				shutil.move(fileName, 'images')


				A = B = C = D = center = np.array([])
				counter = 0
				info = []
				break

		# if the 'j' key is pressed,  rotate the bounding box 0.01 radians counter clockwise
			elif key == ord("j") or key == ord('J'):
				
				theta+=0.01
				print(theta)
				
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()

				
				rotation = np.array([[math.cos(0.01),-1*math.sin(0.01)],[math.sin(0.01),math.cos(0.01)]])

				A = rotation@(A-center) + center
				B = rotation@(B-center) + center
				C = rotation@(C-center) + center
				D = rotation@(D-center) + center

				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				

				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)


	    # if the 'k' key is pressed,  rotate the bounding box 0.01 radians clockwise
			elif key == ord("k") or key == ord('K'):
	            
				theta-=0.01

				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation = np.array([[math.cos(-0.01),-1*math.sin(-0.01)],[math.sin(-0.01),math.cos(-0.01)]])

				A = rotation@(A-center) + center
				B = rotation@(B-center) + center
				C = rotation@(C-center) + center
				D = rotation@(D-center) + center

				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("u"):
				print(theta)
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				A = rotation1@(A-center) + center
				B = rotation1@(B-center) + center
				

				A[1]+=1
				B[1]+=1

				
				

				A = rotation2@(A-center) + center
				B = rotation2@(B-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				h-=1

				
				

				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("U"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				A = rotation1@(A-center) + center
				B = rotation1@(B-center) + center
				

				A[1]-=1
				B[1]-=1
				

				A = rotation2@(A-center) + center
				B = rotation2@(B-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				h+=1
				





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("i"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				
				C = rotation1@(C-center) + center
				D = rotation1@(D-center) + center

				C[1]-=1
				D[1]-=1
				

				
				C = rotation2@(C-center) + center
				D = rotation2@(D-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				h-=1


				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("I"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				
				C = rotation1@(C-center) + center
				D = rotation1@(D-center) + center

				C[1]+=1
				D[1]+=1
				

				
				C = rotation2@(C-center) + center
				D = rotation2@(D-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				h+=1





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("o"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				A = rotation1@(A-center) + center
				D = rotation1@(D-center) + center

				A[0]+=1
				D[0]+=1
				

				A = rotation2@(A-center) + center
				D = rotation2@(D-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				w-=1





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("O"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				A = rotation1@(A-center) + center
				D = rotation1@(D-center) + center

				A[0]-=1
				D[0]-=1
				

				A = rotation2@(A-center) + center
				D = rotation2@(D-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				w+=1





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("p"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				
				B = rotation1@(B-center) + center
				C = rotation1@(C-center) + center
				

				B[0]-=1
				C[0]-=1
				

				
				B = rotation2@(B-center) + center
				C = rotation2@(C-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				w-=1
				





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("P"):
	            
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()
			
				
				rotation1 =   np.array([[math.cos(-1*theta),-1*math.sin(-1*theta)],[math.sin(-1*theta),math.cos(-1*theta)]])
				rotation2 =   np.array([[math.cos(theta),-1*math.sin(theta)],[math.sin(theta),math.cos(theta)]])

				
				B = rotation1@(B-center) + center
				C = rotation1@(C-center) + center
				

				B[0]+=1
				C[0]+=1
				

				
				B = rotation2@(B-center) + center
				C = rotation2@(C-center) + center

				center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
				w+=1
				





				rotated_pts = np.array([A,B,C,D], np.int32)
				rotated_pts = rotated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[rotated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)







			elif key == ord("d") or key == ord('D'):
			
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()

				A[0] += 1
				B[0] += 1
				C[0] += 1
				D[0] += 1
				center[0] += 1

				

				translated_pts = np.array([A,B,C,D], np.int32)
				translated_pts = translated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[translated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("a") or key == ord('A'):
			
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()

				A[0] -= 1
				B[0] -= 1
				C[0] -= 1
				D[0] -= 1
				center[0] -= 1

				

				translated_pts = np.array([A,B,C,D], np.int32)
				translated_pts = translated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[translated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("s") or key == ord('S'):
			
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()

				A[1] += 1
				B[1] += 1
				C[1] += 1
				D[1] += 1
				center[1] += 1
				

				

				translated_pts = np.array([A,B,C,D], np.int32)
				translated_pts = translated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[translated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)

			elif key == ord("w") or key == ord('W'):
			
				if counter==1:
					image = clone.copy()

				else:
					image = temp.copy()

				A[1] -= 1
				B[1] -= 1
				C[1] -= 1
				D[1] -= 1
				center[1] -= 1

				

				translated_pts = np.array([A,B,C,D], np.int32)
				translated_pts = translated_pts.reshape((-1,1,2))

				
				cv2.polylines(image,[translated_pts],True,(0,0,255),3)
				cv2.imshow("image", image)


			elif key == ord("c") or key == ord('C'):
				temp = image.copy()
				theta = convert_theta(theta)
				print(theta)
				c = make_class(theta)
				class_label = int(((c-20)/20)+1)
				if class_label>8:
					class_label = 0
				z = [class_label, center[0]/width, center[1]/height, w/width, h/height]
				info.append(z)
                
				if length>1:
					length-=1
					break



  


cv2.destroyAllWindows()