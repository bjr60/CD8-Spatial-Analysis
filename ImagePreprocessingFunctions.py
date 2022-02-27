# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:33:39 2022

@author: bjr60
"""

#imports
import sys
import numpy as np
import slideio as sld
import os as os
from matplotlib import pyplot as plt
import cv2 as cv
from skimage import io, color, filters, morphology
from scipy.ndimage import morphology as scMorph
from sklearn.cluster import DBSCAN
import csv
import math
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn import mixture
import cv2
import random

sys.setrecursionlimit(10000)



#=================================================Functions============================================================
#----Class of functions to perform dfs to find number of cells, size of cells, and centeroid
# Only use if image and imageLAB exist or there will be an error
class markerDescriptions:
    def __init__(self):
        self.sizes = []
        self.centroids = []
        self.count = 0
        self.currentValue = 0
        self.RGBs = []
        self.LABs = []
    def describeCells(self, grid):
        self.currentSize = 0
        self.sizes = []
        
        self.currentLocationsx = []
        self.currentLocationsy = []
        self.centroids = []
        
        self.count = 0
        self.cellCount = 0
        
        self.currentRs = []
        self.currentGs = []
        self.currentBs = []
        
        self.currentLs = []
        self.currentAs = []
        self.currentB2s = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] > 1:
                    self.currentLocationsx = []
                    self.currentLocationsy = []
                    self.currentValue = grid[i][j]
                    self.currentSize = 0
                    self.currentRs = []
                    self.currentGs = []
                    self.currentBs = []
                    self.currentLs = []
                    self.currentAs = []
                    self.currentB2s = []
                    
                    self.dfs(grid,i,j)
                    
                    self.sizes.append(self.currentSize)
                    averagex = round(sum(self.currentLocationsx)/len(self.currentLocationsx))
                    averagey = round(sum(self.currentLocationsy)/len(self.currentLocationsy))
                    self.centroids.append([averagex,averagey])
                    averageR = 255.0*sum(self.currentRs)/len(self.currentRs)
                    averageG = 255.0*sum(self.currentGs)/len(self.currentGs)
                    averageB = 255.0*sum(self.currentBs)/len(self.currentBs)
                    averageL = sum(self.currentLs)/len(self.currentLs)
                    averageA = sum(self.currentAs)/len(self.currentAs)
                    averageB2 = sum(self.currentB2s)/len(self.currentB2s)
                    self.RGBs.append([averageR,averageG,averageB])
                    self.LABs.append([averageL,averageA,averageB2])
                    self.count+=1
        self.sizes = np.asarray(self.sizes)
    def dfs(self, grid, i ,j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j]!=self.currentValue:
            return
        grid[i][j] = 0
        self.currentSize += 1
        self.currentLocationsx.append(i)
        self.currentLocationsy.append(j)
        self.currentRs.append(image[i][j][0])
        self.currentGs.append(image[i][j][1])
        self.currentBs.append(image[i][j][2])
        self.currentLs.append(imageLAB[i][j][0])
        self.currentAs.append(imageLAB[i][j][1])
        self.currentBs.append(imageLAB[i][j][2])
        
        self.dfs(grid,i+1,j)
        self.dfs(grid,i-1,j)
        self.dfs(grid,i,j+1)
        self.dfs(grid,i,j-1)
        self.dfs(grid,i+1,j+1)
        self.dfs(grid,i-1,j+1)
        self.dfs(grid,i+1,j-1)
        self.dfs(grid,i-1,j-1)

def loadImage(filepath, file):
    #read image
    rootpath = 'D:/spatialAnalysis_cd8_ben/spatialAnalysis_cd8_ben'
    svsfile = '21137.svs'
    slide = sld.open_slide(os.path.join(rootpath,svsfile),'SVS');
    scene = slide.get_scene(0);
    infor = {'dim': np.double(scene.rect), 
             'channel_num': scene.num_channels, 
             'resolution': scene.resolution, 
             'magnification': scene.magnification}
    return scene
def selectImage(startX, startY, width = 1000, height = 1000, plotImage = False, title = ''):
    image = scene.read_block((startX, startY, width, height),(0,0),[])/255.0;
    if plotImage == True:
        #above line reads specific block of larger image and then divides by 255 to get 0-1 rgb values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.imshow(image)
        ax.axis('off')
        if title != '':
            plt.title(title)
        plt.savefig(savepath + '/' + currsavename + '/' + str(i) +'/' + 'Original Image.jpg')
        plt.show()
    return image

def createLABImage(image):
    """Change color space from RGB to Lab"""
    imageLAB = color.rgb2lab(image)
    imageA = imageLAB[:,:,1];#positive--redder; negative--greener
    imageB = imageLAB[:,:,2];#positive--yellower; negative--bluer
    #Try using positive red and negative blue to get masks for blue cells
    return imageLAB, imageA, imageB
    
def binaryMasks(imageA, imageB, plotCD8mask = False, plotBluemask = False):
    """We choose positive from images A and B (for browner stain);
    and negative from image B (for bluer image B stain)"""
    imageAB = imageA + imageB;
    imageAB[imageAB < 0] = 0; #less than 0 would be green and blue, make those 0
    thresh = filters.threshold_otsu(imageAB);
    binaryAB = imageAB > thresh; #1 if red/yellow is greater than threshold
    imageBneg = imageB;
    imageBneg[imageBneg>0] = 0
    thresh = filters.threshold_otsu(imageB)
    binaryBneg = imageBneg < thresh; #0 if less than threshold
    
    """clean image using median filter"""
    binaryABclean = filters.median(binaryAB,morphology.square(3)); #give all values same value as median to get rid of random 1s in groups of 0s or vice versa
    binaryBnegClean = filters.median(binaryBneg,morphology.square(3));
    #look into a multilevel otsu for thresholding, and use 2 threholds and choose the one that is lower
    
    
    binaryABm = morphology.erosion(binaryABclean); #erosion performed to try and seperate two overlapping cells (need better method)
    binaryBnegm = morphology.erosion(binaryBnegClean); #look up command for erosion
                          
    
    holesFilledCD8 = scMorph.binary_fill_holes(binaryABm)
    if plotCD8mask == True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.imshow(holesFilledCD8)
        ax.axis('off')
        plt.savefig(savepath + '/' + currsavename + '/' + str(i) +'/' + 'CD8 Binary Masks.jpg')
        plt.show()
    holesFilledBlue = scMorph.binary_fill_holes(binaryBnegm)
    if plotBluemask == True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.imshow(holesFilledBlue)
        ax.axis('off')
        plt.savefig(savepath + '/' + currsavename + '/' + str(i) +'/' + 'DAPI Binary Masks.jpg')
        plt.show()
    return holesFilledCD8, holesFilledBlue

def holesFilledToNum(holesFilled):
    i = 0
    j = 0
    newHolesFilled = [[0]*len(holesFilled) for col in range(len(holesFilled))]
    while i < len(newHolesFilled):
        j=0
        while j <len(newHolesFilled[0]):
            if holesFilled[i][j] == True:
                newHolesFilled[i][j] = 255
            else:
                newHolesFilled[i][j] = 0
            j+=1
        i+=1
    newHolesFilled = np.array(newHolesFilled).astype('uint8')
    return newHolesFilled

def watershed(newHolesFilled, image, noiseRemovalProcess = True):
    if noiseRemovalProcess == False:
        kernel = np.ones((3,3),np.uint8)
        # sure background area
        sure_bg = cv.dilate(newHolesFilled,kernel,iterations=5)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(newHolesFilled,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.2*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        tempImage = image*255.0
        tempImage = tempImage.astype('uint8')
        markers = cv2.watershed(tempImage,markers)
        tempImage[markers == -1] = [255,0,0]
        return markers, tempImage
    else:
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(newHolesFilled,cv.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=5)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.2*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        tempImage = image*255.0
        tempImage = tempImage.astype('uint8')
        markers = cv2.watershed(tempImage,markers)
        tempImage[markers == -1] = [255,0,0]
        return markers, tempImage
    
#Used to create copies of non numpy arrays    
def createCopy(image):
    temp = np.zeros(image.shape);
    for row in len(image):
        for column in len(image[0]):
            temp[row][column] = image[row][column]
    return temp

def appendToExcelSheet(sheetName, descriptions):
    with open(sheetName, 'a+', newline="") as f:
            writer = csv.writer(f)
            i = 0
            for row in descriptions.centroids:
                if descriptions.sizes[i] >= 50:
                    writer.writerow([row[0],row[1],descriptions.sizes[i],descriptions.RGBs[0],descriptions.RGBs[1],descriptions.RGBs[2],descriptions.LABs[0],descriptions.LABs[1],descriptions.LABs[2]])
                i+=1
        
#=================================================Functions============================================================
    
    
savepath = 'D:/spatialAnalysis_cd8_ben/spatialAnalysis_cd8_ben/segmentationResults' 
rootpath = 'D:/spatialAnalysis_cd8_ben/spatialAnalysis_cd8_ben/group1'
listOfImages = os.listdir(rootpath +'/')

for files in listOfImages:    
    scene = loadImage(rootpath, files)
    currsavename = files[:len(files)-4]
    for i in range(0,10):
        os.mkdir(savepath + '/' + currsavename + '/' + str(i) +'/')
        startX = random.randint(20000, 80000)
        startY = random.randint(35000,65000)
        image = selectImage(startX, startY, 1000, 1000, True, files + ' ' + str(startX) + ' ' + str(startY))
        imageLAB, imageA, imageB = createLABImage(image)
        holesFilledCD8, holesFilledBlue = binaryMasks(imageA, imageB, True, True)
        holesFilledCD8Num = holesFilledToNum(holesFilledCD8)
        holesFilledBlueNum = holesFilledToNum(holesFilledBlue)
        CD8Markers, CD8Watershed = watershed(holesFilledCD8Num, image)
        blueMarkers, blueWatershed = watershed(holesFilledBlueNum, image)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.imshow(CD8Markers)
        ax.axis('off')
        plt.savefig(savepath + '/' + currsavename + '/' + str(i) +'/'+ 'CD8 Watershed Segmentation.jpg')
        plt.show()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.imshow(blueMarkers)
        ax.axis('off')
        plt.savefig(savepath + '/' + currsavename + '/' + str(i) +'/'+ 'DAPI Watershed Segmentation.jpg')
        plt.show()

        
"""
copyMarker = CD8Markers.copy()
CD8Descriptions = markerDescriptions()
CD8Descriptions.describeCells(copyMarker)
appendToExcelSheet('watershedWholeImageTrial1.csv',CD8Descriptions)
copyMarker = blueMarkers.copy()
blueDescriptions = markerDescriptions()
blueDescriptions.describeCells(copyMarker)
appendToExcelSheet('watershedWholeImageTrial1.csv',blueDescriptions)
"""
