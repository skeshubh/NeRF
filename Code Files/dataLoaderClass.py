import numpy as np
import math
import cv2 as cv


class DataLoaderClass:
    def __init__(self):
        self.images = []
        self.cameraCenters = []
        self.poseMatrices = []
        self.focal = 0.0
        
    def getImages():
        filePath = "train-20221030T040052Z-001/train/r_"
        extension = ".png"
        images = []
        for i in range (0,100):
            fileName = filePath + str(i) + extension
            img = cv.imread(fileName)
            images.append(img)
        return images
    
    
    def getFocal():
        fileName = "train-20221030T040052Z-001/train/cameras.txt"
        file = open(fileName,"r")
        lines = file.readlines()
        lineNumber = 1
        focal = 0.0
        for line in lines:
            if(lineNumber >= 4):
                reading = line.split(" ")
                focalLength = float(reading[4])
                focal += focalLength
            lineNumber += 1
        focal /= 100
        return focal
    
    
    def getQuaternionsAndTranslations():
        fileName = "train-20221030T040052Z-001/train/images.txt"
        file = open(fileName,"r")
        lines = file.readlines()
        lineNumber = 1
        quaternions = []
        translations = []
        for line in lines:
            if(lineNumber >= 5 and (lineNumber%2) != 0):
                reading = line.split(" ")
                quat = np.array([float(reading[1]),float(reading[2]),float(reading[3]),float(reading[4])])
                quaternions.append(quat)
                trans = np.array([float(reading[5]),float(reading[6]),float(reading[7])])
                translations.append(trans)
            lineNumber += 1
        return quaternions,translations

    def getRotationalMatrices(quaternions):
        rotationalMatrices = []
        for i in range (len(quaternions)):
            q = quaternions[i]
            r00 = 2*(q[0]**2 + q[1]**2)-1
            r01 = 2*(q[1]*q[2] - q[0]*q[3])
            r02 = 2*(q[1]*q[3] + q[0]*q[2])
            r10 = 2*(q[1]*q[2] + q[0]*q[3])
            r11 = 2*(q[0]**2 + q[2]**2)-1
            r12 = 2*(q[2]*q[3] - q[0]*q[1])
            r20 = 2*(q[1]*q[3] - q[0]*q[2])
            r21 = 2*(q[2]*q[3] + q[0]*q[1])
            r22 = 2*(q[0]**2 + q[3]**2)-1
            rotationalMatrix = np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])
            rotationalMatrices.append(rotationalMatrix)
        return rotationalMatrices


    def getCameraCenters(rotationalMatrices,translations):
        cameraCenters = []
        for i in range(len(rotationalMatrices)):
            center = -1*np.dot(rotationalMatrices[i].T,translations[i])
            cameraCenters.append(center)
        return cameraCenters


    def getViewingAngles(cameraCenters):
        thetas = []
        phis = []
        for i in range(len(cameraCenters)):
            x,y,z = cameraCenters[i]
            theta = math.degrees(math.atan(y/x))
            phi = math.degrees(math.acos(z/(np.sqrt(x**2+y**2+z**2))))
            thetas.append(theta)
            phis.append(phi)
        return thetas,phis


    def getPoseMatrices(rotationalMatrices,translations):
        poseMatrices = []
        for i in range (len(rotationalMatrices)):
            poseMatrix = np.identity(4)
            poseMatrix[0] = np.append(rotationalMatrices[i][0],translations[i][0])
            poseMatrix[1] = np.append(rotationalMatrices[i][1],translations[i][1])
            poseMatrix[2] = np.append(rotationalMatrices[i][2],translations[i][2])
            poseMatrices.append(poseMatrix)
        return poseMatrices
    
    
    def getData(self):
        images = self.getImages()
        focal = self.getFocal()
        quaternions,translations = self.getQuaternionsAndTranslations()
        rotationalMatrices = self.getRotationalMatrices(quaternions)
        cameraCenters = self.getCameraCenters(rotationalMatrices, translations)
        thetas,phis = self.getViewingAngles(cameraCenters)
        poseMatrices = self.getPoseMatrices(rotationalMatrices, translations)
        self.images = images
        self.focal = focal
        self.cameraCenters = cameraCenters
        self.poseMatrices = poseMatrices
        return self.images, self.focal, self.cameraCenters, self.poseMatrices