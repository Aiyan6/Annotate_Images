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


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, rotated_pts, A , B , C , D , center, counter, w, h, theta, width, height, info, temp, undo
	key = cv2.waitKey(1) & 0xFF
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		theta = 0
		undo = False
		refPt.append((x, y))
		
	
		counter+=1
		print(counter)
		
		# calculate the coordinates of all the four corners of the bounding box
		A = np.array([refPt[0][0],refPt[0][1]])
		B = np.array([refPt[1][0],refPt[0][1]])
		C = np.array([refPt[1][0],refPt[1][1]])
		D = np.array([refPt[0][0],refPt[1][1]])

		# calculate the coordinates of the center of the bounding box
		center = np.array([(A[0]+C[0])/2, (A[1]+C[1])/2])
		print(center)

		


        # calculate the width and height of the bounding box
		w = abs(A[0] - B[0])
		h = abs(A[1] - D[1]) 

        # display the bounding box on the image
		cv2.rectangle(image, refPt[0], refPt[1], (0, 0, 255), 3)
		cv2.imshow("image", image)
		
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dir", required=True, help="Path to the folder")
args = vars(ap.parse_args())

# list containing all the image names in the specified directory
image_names = os.listdir(args["dir"])
for img in image_names:
	path = args["dir"] + "/" + img
	print(path)


	image = cv2.imread(path)
	height, width, channels = image.shape
	
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)

	if(cv2.setMouseCallback("image", click_and_crop)):
		print('Yes')


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

			
			
			fileName = img.replace('jpg','xml')
			root = ET.Element("annotation")
			folder = ET.SubElement(root, "folder").text = str(args['dir'])
			filename = ET.SubElement(root, "filename").text = img
			path = ET.SubElement(root, "path").text = str(pathlib.Path().absolute()) + '/' + args['dir'] + '/' + img
			source = ET.SubElement(root, "source")
			database = ET.SubElement(source, "database").text = 'Unknown'
			size = ET.SubElement(root, "size")
			width = ET.SubElement(size, "width").text = str(width)
			height = ET.SubElement(size, "height").text = str(height)
			depth = ET.SubElement(size, "depth").text = str(channels)
			segmented = ET.SubElement(root, "segmented").text = '0'
			object = ET.SubElement(root, "object")
			name = ET.SubElement(object, "name").text = 'bk'
			pose = ET.SubElement(object, "pose").text = 'Unspecified'
			truncated = ET.SubElement(object, "truncated").text = '0'
			difficult = ET.SubElement(object, "difficult").text = '0'			

			for bbox in info:

				
				bndbox = ET.SubElement(object, "bndbox")
				xmin = ET.SubElement(bndbox, "xmin").text = str(bbox['xmin'])
				ymin = ET.SubElement(bndbox, "ymin").text = str(bbox['ymin'])
				xmax = ET.SubElement(bndbox, "xmax").text = str(bbox['xmax'])
				ymax = ET.SubElement(bndbox, "ymax").text = str(bbox['ymax'])
				theta = ET.SubElement(bndbox, "theta").text = str(bbox['theta'])

        

			tree = ET.ElementTree(root)
			indent(root)
			tree.write(fileName, encoding="utf-8")

			shutil.move(fileName,args['dir'])
			




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
			if theta<0:
				theta = (2*math.pi) + theta
			z = {'width': width , 'height': height , 'xmin': A[0] , 'ymin': A[1] , 'xmax': C[0] , 'ymax': C[1], 'theta': theta}
			info.append(z)



  


cv2.destroyAllWindows()