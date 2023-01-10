import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt

from dataLoaderClass import DataLoaderClass


def getData():
    images,focal,cameraCenters,pose = DataLoaderClass().getData()
    return images,focal,cameraCenters,pose


def normalizeCoordinates(X):
    scaledUnit = 2/(np.max(X)-np.min(X))
    normalizedX = X*scaledUnit - np.min(X)*scaledUnit - 1
    return normalizedX


def cumprodExclusive(tensor):
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod,1,-1)
    cumprod[...,0] = 1.0
    return cumprod


def getBatches(tensor,chunksize):
    batches = [tensor[i : i+chunksize] for i in range (0,tensor.shape[0],chunksize)]
    return batches


def gammaFunction(L,p,normalise=False):
    if(normalise):
        normalizedp = F.normalize(p,p=1.0,dim=0)
    else:
        normalizedp = p
    frequency_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (L - 1),L,dtype=p.dtype,device=p.device)
    gammap = []
    for freq in frequency_bands:
        for func in [torch.sin,torch.cos]:
            gammap.append(func(normalizedp*freq))
    gammap = torch.cat(gammap,dim=-1)
    return gammap


def generateRays(H,W,focal,pose):
    meshX,meshY = torch.meshgrid(torch.arange(W,dtype=pose.dtype, device=pose.device),torch.arange(H,dtype=pose.dtype, device=pose.device))
    meshX = meshX.transpose(-1,-2)
    meshY = meshY.transpose(-1,-2)
    directions = torch.stack([((meshX-W*0.5)/focal),(-(meshY-H*0.5)/focal),(-torch.ones_like(meshX))],dim=-1)
    ray_d = torch.sum(directions[...,None,:]*pose[:3,:3],dim=-1)
    ray_o = pose[:3,-1].expand(ray_d.shape)
    return ray_o,ray_d


def generateQueryPoints(ray_o,ray_d,near,far,nPoints):
    depth = torch.linspace(near,far,nPoints).to(ray_o.device)
    queryPoints = torch.tensor(ray_o[...,None,:] + ray_d[...,None,:] * depth[...,:,None],dtype=torch.float32)
    return queryPoints, depth


def generateViewingAngles(pose):
    x,y,z = pose[:3,-1]
    theta = math.degrees(math.atan(y/x))
    phi = math.degrees(math.acos(z/(x**2+y**2+z**2)**0.5))
    x1 = math.cos(phi)*math.sin(theta)
    y1 = math.sin(phi)*math.sin(theta)
    z1 = math.cos(theta)
    return torch.tensor([x1,y1,z1],dtype=torch.float32)


def renderVolumeDensity(sigma,rgb,ray_o,depth):
    e_10 = torch.tensor([1e10],dtype = ray_o.dtype,device=ray_o.device)
    dists = torch.cat((depth[..., 1:] - depth[...,:-1],e_10.expand(depth[..., :1].shape)),dim=-1)
    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * cumprodExclusive(1.0 - alpha + 1e-10)
    rgbMap = (weights[...,None] * rgb).sum(dim=-2)
    depthMap = (weights * depth).sum(dim=-1)
    accMap = weights.sum(-1)
    return rgbMap, depthMap, accMap


def runNerfOnce(H,W,focal,pose,near,far,nPoints,chunks,model,pArgs,vArgs):
    ray_o,ray_d = generateRays(H, W, focal, pose)
    queryPoints, depth = generateQueryPoints(ray_o, ray_d, near, far, nPoints)
    flattenedQueryPoints = torch.Tensor(queryPoints.reshape((-1,3)))
    viewingAngles = generateViewingAngles(pose)
    gammaX = gammaFunction(pArgs, flattenedQueryPoints, True).double().to(ray_o.device)
    gammaD = gammaFunction(vArgs, viewingAngles, False).double().to(ray_o.device)
    sigma, rgb = model(gammaX,gammaD)
    sigmaShape = list(queryPoints.shape[:-1])
    rgbShape = list(queryPoints.shape[:-1]) + [3]
    sigma = torch.reshape(torch.flatten(sigma),sigmaShape)
    rgb = torch.reshape(torch.flatten(rgb),rgbShape)
    rgbMap, depthMap, accMap = renderVolumeDensity(sigma, rgb, ray_o, depth)
    return rgbMap,depthMap,accMap
    

class NeRFModel(nn.Module):
    def __init__(self):
        super(NeRFModel,self).__init__()
        self.linear1 = nn.Linear(60,256)
        self.linear2 = nn.Linear(256,256)
        self.linear3 = nn.Linear(256,256)
        self.linear4 = nn.Linear(256,256)
        self.linear5 = nn.Linear(316,256)
        self.linear6 = nn.Linear(256,256)
        self.linear7 = nn.Linear(256,256)
        self.linear8 = nn.Linear(256,256) 
        self.linearSigma = nn.Linear(256,1)
        self.linear9 = nn.Linear(280,256)
        self.linear10 = nn.Linear(256,128)
        self.linearRgb = nn.Linear(128,3)
        self.double()
    
    def forward(self,gammaX,gammaD):
        temp = torch.sigmoid(self.linear1(gammaX))
        temp = torch.sigmoid(self.linear2(temp))
        temp = torch.sigmoid(self.linear3(temp))
        temp = torch.sigmoid(self.linear4(temp))
        temp = torch.cat((gammaX,temp),dim=1)
        temp = torch.sigmoid(self.linear5(temp))
        temp = torch.sigmoid(self.linear6(temp))
        temp = torch.sigmoid(self.linear7(temp))
        sigma = torch.sigmoid(self.linear8(temp))
        temp1 = gammaD.repeat(sigma.shape[0],1)
        temp = torch.cat((sigma,temp1),dim=1)
        sigma = torch.relu(self.linearSigma(sigma))
        temp = torch.sigmoid(self.linear9(temp))
        temp = torch.relu(self.linear10(temp))
        rgb = torch.sigmoid(self.linearRgb(temp))
        return sigma,rgb        
        

def getImages():
    filePath = "train-20221030T040052Z-001/train/r_"
    extension = ".png"
    images = []
    for i in range (0,100):
        fileName = filePath + str(i) + extension
        img = cv.imread(fileName)
        img = img[0::2,0::2,...]
        images.append(img)
    images = np.array(images)
    return images


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


def getPoseMatrices(rotationalMatrices,translations):
    poseMatrices = []
    for i in range (len(rotationalMatrices)):
        poseMatrix = np.identity(4)
        poseMatrix[0] = np.append(rotationalMatrices[i][0],translations[i][0])
        poseMatrix[1] = np.append(rotationalMatrices[i][1],translations[i][1])
        poseMatrix[2] = np.append(rotationalMatrices[i][2],translations[i][2])
        poseMatrices.append(poseMatrix)
    poseMatrices = np.array(poseMatrices)
    return poseMatrices


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
    focal = np.array(focal)
    return focal


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quaternions,translations = getQuaternionsAndTranslations()
    rotationalMatrices = getRotationalMatrices(quaternions)
    
    images = torch.from_numpy(getImages())    
    poses = getPoseMatrices(rotationalMatrices, translations)
    poses = torch.from_numpy(poses).to(device)
    focal = getFocal()
    focal = torch.from_numpy(focal).to(device)
    H,W,_ = images[0].shape
    near = 2.0
    far = 6.0
    
    testimg, testpose = images[69],poses[69]
    testimg = testimg.to(device)
    testpose = testpose.to(device)
    
    pArgs = 10
    vArgs = 4
    nPoints = 16
    chunksize = 4096
    lr = 5e-4
    nIters = 50000
    nDisplay = 100
    model = NeRFModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for i in range(nIters):
        print("i=",i)
        targetImgID = np.random.randint(images.shape[0])
        targetImg = images[targetImgID].to(device)
        targetImg = targetImg.type(torch.DoubleTensor).to(device)
        targetPose = poses[targetImgID].to(device).to(device)
        rgbMap,depthMap,accMap = runNerfOnce(H, W, focal, targetPose, near, far, nPoints, chunksize, model, pArgs, vArgs)
        loss = F.mse_loss(rgbMap,targetImg)
        loss = loss.type(torch.DoubleTensor)
        print("loss=",loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
    rgbMap,depthMap,accMap = runNerfOnce(H, W, focal, testpose, near, far, nPoints, chunksize, model, pArgs, vArgs)
    final = rgbMap.cpu().detach().numpy()
    cv.imwrite("final.png",final)
    print("Done")
    
    
if __name__ == "__main__":
    main()
