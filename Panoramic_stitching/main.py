import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import imutils
'''
supported image pairs:
1 image/JishiBuildingLeft.jpg & image/JishiBuildingRight.jpg
2 image/DormitoryLeft.jpg & image/DormitoryRight.jpg
3 image/MountainLeft.jpg & image/MountainRight.jpg
4 image/NightScapeLeft.jpg & image/NightScapeRight.jpg
5 image/TjSlateLeft.jpg & image/TjSlateRight.jpg
'''
LeftImage="image/JishiBuildingLeft.jpg"
RightImage="image/JishiBuildingRight.jpg"
#Read images
img1=cv.imread(RightImage)
img2=cv.imread(LeftImage)

#Using SIFT to extract features
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
#display the keypoints extracted
img_featureExtraction1=cv.drawKeypoints(img1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_featureExtraction2=cv.drawKeypoints(img2,kp2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite("Save/feature-extraction-1.jpg",img_featureExtraction1)
cv.imwrite("Save/feature-extraction-2.jpg",img_featureExtraction2)
#renew the images
img1=cv.imread(RightImage)
img2=cv.imread(LeftImage)


#Feature matching with FLANN(Fast Library for Approximate Nearest Neighbors)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
#plt.imshow(img3,),plt.show()

#Ratio Test
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

#Get homography matrix with RANSAC
M=[]
mask=[]
MIN_MATCH_COUNT = 50
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,c = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
#renew image
img1=cv.imread(RightImage)
img2=cv.imread(LeftImage)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv.imwrite("Save/matched.jpg",img3)
print(M)
#renew the image
img1=cv.imread(RightImage)
img2=cv.imread(LeftImage)
OFFSET=np.array([[1.0,0,200],[0,1,200],[0,0,1]])
# Apply panorama correction
width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]

result=cv.warpPerspective(img1, M, (width, height))


result[0:img2.shape[0], 0:img2.shape[1]]=img2

# transform the panorama image to grayscale and threshold it
gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

# Finds contours from the binary image
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# get the maximum contour area
c = max(cnts, key=cv.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv.boundingRect(c)

# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(result)
plt.axis('off')
plt.show()
cv.imwrite("Save/result.jpg",result)


#Show the panoramic stitching result implemented in opencv library
img1=cv.imread(RightImage)
img2=cv.imread(LeftImage)
stitcher = cv.Stitcher.create(0)
result = stitcher.stitch((img1,img2))
cv.imwrite("Save/Opencv_result.jpg", result[1])