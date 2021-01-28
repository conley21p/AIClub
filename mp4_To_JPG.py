# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:28:33 2021

@author: conle
"""

import cv2
import os

#Folder path with .mp4 video files
mask_mp4_path = "mask_videos"
non_mp4_path = "non_videos"

#Folder path with jpg files will be sent
mask_jpg_path = "mask_JPG"
non_jpg_path = "non_JPG"


#saves the frame of video as jpg in the folderPath folder
def getFrame(vid,folderPath,filename,sec,count):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    if hasFrames:
        cv2.imwrite(folderPath+"\\"+filename+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

def convert(fp,fileName,end):
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    vidcap = cv2.VideoCapture(fp+fileName)
    print(fileName)
    success = getFrame(vidcap,end,fileName,sec,count)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap,end,fileName,sec,count)
            
#runs the conversion for all mp4 files in the non_mp4_path folder
for filename in os.listdir(non_mp4_path):
    convert(non_mp4_path + "\\",filename, non_jpg_path)
#runs the conversion for all mp4 files in the mask_mp4_path folder#runs the conversion for all mp4 files in the non_mp4_path folder
for filename in os.listdir(mask_mp4_path):
    convert(mask_mp4_path + "\\",filename, mask_jpg_path)
        
#console output done when done
print("done")