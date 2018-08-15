from object_detection.utils import dataset_util
from lxml import etree
import glob
import cv2 as cv
import os
import random
DIR_img = 'data/xuelang*/*/'
DIR_xml = 'data/xuelang*/*/'
DIR_tag = 'data/crop/'
crop_size = 640
boxe43 = [[0, 640, 0, 640], [320, 960, 0, 640], [640, 1280, 0, 640], [960, 1600, 0, 640], [1280, 1920, 0, 640], [0, 640, 320, 960], [320, 960, 320, 960], [640, 1280, 320, 960], [960, 1600, 320, 960], [1280, 1920, 320, 960], [0, 640, 640, 1280], [320, 960, 640, 1280], [640, 1280, 640, 1280], [960, 1600, 640, 1280], [1280, 1920, 640, 1280], [0, 640, 960, 1600], [320, 960, 960, 1600], [640, 1280, 960, 1600], [960, 1600, 960, 1600], [1280, 1920, 960, 1600], [0, 640, 1280, 1920], [320, 960, 1280, 1920], [640, 1280, 1280, 1920], [960, 1600, 1280, 1920], [1280, 1920, 1280, 1920], [0, 640, 1600, 2240], [320, 960, 1600, 2240], [640, 1280, 1600, 2240], [960, 1600, 1600, 2240], [1280, 1920, 1600, 2240], [0, 640, 1920, 2560], [320, 960, 1920, 2560], [640, 1280, 1920, 2560], [960, 1600, 1920, 2560], [1280, 1920, 1920, 2560]]
#(xi,xa,yi,ya)
c = 0
c2 = 0
DIR_zc = 'data/xuelang/zc/'
DIR_yc = 'data/xuelang/yc/'
lll = []
def iou(box1, box2):
	if max(box1[0],box2[0])>min(box1[1],box2[1]):return False
	if max(box1[2],box2[2])>min(box1[3],box2[3]):return False
	return True

def listIou(box_list, boxa):
	if len(box_list)==0:return False
	for box in box_list:
		if iou(box,boxa):
			return True
	return False

def crop(xi,xa,yi,ya,X,Y):
	w = xa-xi
	h = ya-yi
	if w<crop_size :
		xi = max(0, xi-int((crop_size-w)/2))
		xa = min(X, xi+int((crop_size-w)/2))
		if xa-xi<crop_size:
			xi = max(0,  xi - (crop_size-xa+xi))
			xa = min(X,  xa + (crop_size-xa+xi))
	if h<crop_size :
		yi = max(0, yi-int((crop_size-w)/2))
		ya = min(Y, yi+int((crop_size-w)/2))
		if ya-yi<crop_size:
			yi = max(0,  yi - (crop_size-ya+yi))
			ya = min(Y,  ya + (crop_size-ya+yi))
	return (xi,xa,yi,ya)

def fun(path, imgpath):
	img = cv.imread(imgpath)
	line = open(path).read()
	xml = etree.fromstring(line)
	data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
	image_list = []
	box_list = []
	for obj in data['object']:
		box_list.append([int(obj['bndbox']['ymin']),int(obj['bndbox']['ymax']),int(obj['bndbox']['xmin']),int(obj['bndbox']['xmax'])])
	mark = False
	for i,box in enumerate(boxe43):
		_path = os.path.basename(imgpath).split('.jpg')[0] + "%02d"%i + '.jpg'
		if (listIou(box_list, box)):
			continue
			_path = os.path.join(DIR_yc, _path)
			cv.imwrite(_path, img[box[0]:box[1],box[2]:box[3]])
		else:
			if random.randint(0,100)<60:continue
			_path = os.path.join(DIR_zc, _path)
			cv.imwrite(_path, img[box[0]:box[1],box[2]:box[3]])
			

if __name__ == '__main__':
	if not  os.path.exists(DIR_zc):
		os.makedirs(DIR_zc)
	path_list = glob.glob(DIR_xml+'*xml')
	for xmlpath in path_list:
		imgpath = os.path.join(os.path.dirname(xmlpath), os.path.basename(xmlpath).split('.xml')[0]+'.jpg')
		imgCrop = fun(xmlpath, imgpath)
		#for i,img in enumerate(imgCrop):
		#	tagpath = os.path.join(DIR_tag, os.path.basename(xmlpath).split('.xml')[0]+str(i)+'.jpg')
		#	cv.imwrite(tagpath, img)
