import cv2 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

camera3d = np.array([0,0,0,1])


all_cam = np.empty((0, 3))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
    

ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)

def update_points(new_pts3d):
    global all_points
    # Append new points to the existing data
    all_points = np.vstack((all_points, new_pts3d))


# Function to update the plot
def update_camera(new_pts3d):
    global all_cam
    # Append new points to the existing data
    all_cam = np.vstack((all_cam, new_pts3d))

 
video = cv2.VideoCapture('test_videos/clean.mp4')

orb = cv2.ORB_create()

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


ret, frame1 = video.read()

f1_pose = np.eye(4)


W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


focal = 200
K = np.array([[focal,0,W//2],[0,focal,H//2],[0,0,1]])

def add_one(pts):
    return np.hstack([pts, np.ones((pts.shape[0],1))])


def triangulation(pose1, pose2, pts1, pts2):
    result = np.zeros((pts1.shape[0], 4))

    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)

    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

    for i, p in enumerate(zip(add_one(pts1), add_one(pts2))):


        A = np.zeros((4,4))

        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        

        _, _, vt = np.linalg.svd(A)

        result[i] = vt[3]

    return result

timer = 0 
old_rt = np.zeros((4,4))
while video.isOpened() and timer < 300 :
    ret, frame2 = video.read()
    timer += 1
    print("The frame -> ", timer)

    keypoints1 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), 2000, qualityLevel=0.07, minDistance=3)


    keypoints1 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in keypoints1]
    pts1, des1 = orb.compute(frame1, keypoints1)



    keypoints2 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), 2000, qualityLevel=0.07, minDistance=3)
    
    keypoints2 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in keypoints2]
    pts2, des2 = orb.compute(frame2, keypoints2)

 
    matches = flann.knnMatch(des1, des2, k=2 )

    knn = []
    uni1 = set()
    uni2 = set()
    for mn in matches:
     if len(mn) == 2:
         m ,n = mn
         if m.distance < 0.75*n.distance and m.queryIdx not in uni1 and m.trainIdx not in uni2:
           uni1.add(m.queryIdx)
           uni2.add(m.trainIdx)
           knn.append(m)
     
    
    if len(knn) < 5: 
         print("To little knns")
         continue
         
    pts1 = np.float32([pts1[m.queryIdx].pt for m in knn])
    pts2 = np.float32([pts2[m.trainIdx].pt for m in knn])     


    
    
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    E = K.T @ F @ K
    
    inliers = inliers.ravel().astype(bool)
    pts1_in = np.unique(pts1[inliers], axis=0)
    pts2_in = np.unique(pts2[inliers], axis=0)
    assert len(pts1_in)==len(pts2_in), "a problem with pts_ins"
    print(len(pts1_in))
    # Has some issues not reliable due to some reasons -> https://answers.opencv.org/question/56588/opencv-30-recoverpose-wrong-results/
    _, _, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

    R = cv2.decomposeEssentialMat(E)
    R_good = R[1]
    
    if np.sum(R[1].diagonal()) < 0:
        R_good = R[0]
    if t[2, 0]< 0:
        t *= -1
    Rt = np.eye(4)
    
    Rt[:3, :3] = R_good
 
    Rt[:3, 3] = t[:,0]
    
    avg_rt = ( (0.6 * Rt) + (0.4 * old_rt)) # for reducing the noise by using a momentum like approach
    f2_pose = np.dot(f1_pose,avg_rt)
    

    # These are for finding the points in real world
    # pts4d = triangulation(f1_pose, f2_pose,pts1[inliers], pts2[inliers])

    # pts4d /= pts4d[:, 3:]

    # good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    
    camera3d =np.dot(avg_rt, camera3d)
    old_rt = avg_rt
    # print(camera3d)
    
    
 
    update_camera(camera3d[:3])

    
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img =  cv2.drawKeypoints(gray, keypoints1, None, color=(0,255,0), flags=0)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
    f1_pose = f2_pose
    frame1 = frame2





scatter_cam = ax.scatter([], [], [], c='b', label='Camera Points')



# Set fixed axis limits
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

# Initialization function
def init():
    scatter_cam._offsets3d = ([], [], [])

    scatter_cam.set_array(np.array([]))

    return scatter_cam

# Update function for animation
# Update function for animation
def update(frame):
    if frame == 0:
        return scatter_cam  # Skip the first frame as it has no points
    
    # Extract data for current frame
    x_cam_data = all_cam[:frame, 2]
    y_cam_data = all_cam[:frame, 1]
    z_cam_data = all_cam[:frame, 0]
    

    
    # Update the scatter plot data
    scatter_cam._offsets3d = (x_cam_data, y_cam_data, z_cam_data)
    
    
    # Create an array for color mapping
    colors_cam = np.linspace(0, 1, frame)
    scatter_cam.set_array(colors_cam)
    
    
    return scatter_cam


# Create animation
ani = FuncAnimation(fig, update, frames=len(all_cam), init_func=init, blit=False, repeat=False)

# Show the plot
plt.show()




video.release()
cv2.destroyAllWindows()