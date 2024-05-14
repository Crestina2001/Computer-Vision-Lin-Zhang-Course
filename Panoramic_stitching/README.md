# Installation and Running

You need to install:
opencv(My python is 3.8.8 and opencv is correspondent to it)
matplotlib
numpy
imutils(just use 'pip install imutils' to install it)

After the installation, just run 'python main.py' in cmd

More results are saved in the folder named 'Save'

To choose different pictures to stitch, you can revise the path of pictures in main.py

# Report

***5. (Programming) Get two images\*** **I*****1 and\*** **I*****2 of our campus and make sure that the\*** ***\**\*major parts of\*\**\*** **I*****1 and\*** **I*****2 are from the same physical plane. Stitch\*** **I*****1 and\*** **I*****2 together\*** ***\**\*to get a panorama view using scale-normalized LoG (or DoG) based interest point\*\** \**\*detector and SIFT descriptor. You can use OpenCV or VLFeat.\*\**\***

 

**First Step: Key points and Feature extraction with SIFT algorithm**

**Description:**

Different from features extracted by Harris edge detection, which is only invariant to rotation, SIFT edge detection is also invariant to scaling. With SIFT, we can extract all the key points and their characteristic scales, with which we can build image patches. Then for any region, we can build a histogram about the orientation of gradients of this area. To build a histogram, we compute gradients for all the points in the region, select the dominant orientation of gradients and rotate all the gradients so that the dominant orientation points to a fixed direction. We just separate the image patches mentioned above into 4*4 sub-patches, and draw a histogram with 8 bins on each of them. In this way, we get a 4*4*8=128 element descriptor, which is invariant to rotation and scaling, for each key point. In summary, we get key points and their descriptors from an image with SIFT algorithm.

**Effect:**

![img](file:///C:/Users/Forrest/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

![img](file:///C:/Users/Forrest/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

**Second Step: Feature Matching and Ratio Testing**

**Description:**

After the first step, we have acquired the key points and their descriptors of the two images, but we don’t know how the points in the first image match the points in the second one. A simple way to match points is to use brutal search to find the nearest one of each point. The distance of two points can be defined as the Euclidean distance between their descriptors. In the implementation, we adopt the FLANN to match points, which is a library for performing fast approximate nearest neighbor searches. And the ratio test is to drop the matches which are not similar enough so as to avoid mismatching.

**Effect:**

![img](file:///C:/Users/Forrest/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

**Third Step: Estimate the Homographic Matrix with RANSAC**

**Description:**

Firstly, randomly select a subset(at least four points) from key points and other points make up a complementary set. Secondly, fit a homographic matrix on them with Least Square Method. Thirdly, exert photographic transformation on every point in the complementary set with the homographic matrix to get transformed points. Fourthly, compute the distance between each transformed point and its matched points in the second image. If the distance is within a given threshold, count it. If the number of points counted exceeds another given threshold, return this homographic matrix and counted points. Otherwise, go back to the first step.

**Effect:**

M=

![img](file:///C:/Users/Forrest/AppData/Local/Temp/msohtmlclip1/01/clip_image007.png)

**Final Step: Stitch the two images**

**Description:**

Firstly, we transform the right image so that its key points overlap with their counterparts in the left one. Then we just paste the left image on the transformed right image to stitch them. Part of right image will be covered by the left one in this way of stitching.

**Effect:**

![img](file:///C:/Users/Forrest/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg)