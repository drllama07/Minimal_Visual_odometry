# Minimal Visual Odometry 

A super minimal visual odometry implementation in Python with [OpenCV](https://opencv.org/).


## Capabilities
- The project processes a video to estimate the camera trajectory and plots it in 3D space.
- The focal length of the camera used to film the video profoundly affects the accuracy of the result. Thus, try to provide an accurate focal length as the parameter  `focal` in `slam.py`.


## Used Libraries:

- [OPENCV](https://opencv.org/) for image processing and more.
- [Matplotlib](https://matplotlib.org/) for 3D plotting of camera trajectory
- [Numpy](https://numpy.org/) for matrix operations


## Implementation Details:

***NOTE:*** This project is heavily inspired by [TwitchSlam](https://github.com/geohot/twitchslam) by George Hotz and [Orb-Slam Monocular](https://github.com/raulmur/ORB_SLAM).

1. Feature Detection using `cv2.goodFeaturesToTrack`:
   -  This function is used to detect strong corners in the image, which serve as good points for tracking and feature matching.
2. KeyPoint Conversion and Description Computation:
   -  Convert detected points to `cv2.KeyPoint` objects and compute their descriptors using ORB.

3. Feature Matching:
   - Match features between consecutive frames using a feature matcher(FLANN).
   - This matches the features and keypoints between `frame1` and `frame2`.
4. Fundamental Matrix and Essential Matrix Calculation:
   - Calculate the Fundamental and Essential matrices to find the relative pose between frames.
   - Later on we will retrieve rotation and translation of camera from essential matrix.

5. Pose Recovery
   - Recover the relative pose (rotation and translation) between frames using the essential matrix.
   > **Note:** 
   Normally, you can use the `cv2.recoverPose` function provided by OpenCV to estimate the relative pose between frames, specifically the rotation matrix R and translation vector t. However, due to the nature of these estimations, the rotation has two possible solutions, only one of which is accurate. By manually engineering the rotation matrix R from the decomposed essential matrix, we achieve far more accurate results. 
   [Here's the refined approach based on a solution from the  OpenCV Q&A forum](https://answers.opencv.org/question/56588/opencv-30-recoverpose-wrong-results/)

   **More robust method**
   > ```python 
   > R = cv2.decomposeEssentialMat(E)
   > R_good = R[1]
   > 
    > if np.sum(R[1].diagonal()) < 0:
     >   R_good = R[0]
     > if t[2, 0]< 0:
     >   t *= -1
     > Rt = np.eye(4)
    >
    > Rt[:3, :3] = R_good
     > Rt[:3, 3] = t[:,0] 
    >```
   - ***Side Note:*** There is a weighted average method to reduce noise and smooth the trajectory estimation. You can adjust it for your needs.

     `avg_rt = ( (0.6 * Rt) + (0.4 * old_rt))`
6. The Plotting of The 3D Trajectory:
   - This script animates a 3D scatter plot to visualize camera points over frames using`matplotlib.animation.FuncAnimation`.

## Personal Code Review:
- While doing this, I have learned a lot about computer vision and algebraic methods used in computer vision.
  > ***Some of the resources I have used to learn these things:***
  >
  > [Epipolar Geometry by First Principles of Computer Vision](https://youtu.be/6kpBqfgSPRc?si=2Zc2HIInQ4UaEZUP)
  > 
  > [Estimating motion by First Principles of Computer Vision](https://youtu.be/JlOzyyhk1v0?si=e2aV-mieePhgmiaX)
  > 
  > [Calculating Depth by First Principles of Computer Vision](https://youtu.be/OYwm4VM6uNg?si=ENSPLz-kDiFH_Xq8)
  > 
  > [TwitchSlam](https://youtu.be/7Hlb8YX2-W8?si=OG7RyOBWNvi19uC7)
  > 


- This project contains some unfinished aspects like keypoint plotting and Slam like features.



For additional learning resources and tutorials, please stay tuned for updates.