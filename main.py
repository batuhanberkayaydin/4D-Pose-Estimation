"""

B.Berkay AYDIN
batuhanberkayaydin@gmail.com
15.01.2022 
pose_6DoF.py file

"""

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
Numpy file reader
Inputs : numpy file path
Outputs : numpy arrays
"""
def read_numpy_files(path_vr2d, path_vr3d):
    
    vr2d_coords = np.load(path_vr2d)  
    vr3d_coords = np.load(path_vr3d) 

    vr2d_coords = np.expand_dims(vr2d_coords, axis=0)
    vr3d_coords = np.expand_dims(vr3d_coords, axis=0)

    return vr2d_coords, vr3d_coords

"""
Not used orb matching because of the results. 
I used sift instead of orb to get better results.
"""
def orb_matching(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # Did not work correctly !
    # descriptor = cv2.xfeatures2d.BEBLID_create(0.001)
    # kp1 = sift_detector.detect(img1, None)
    # kp2 = sift_detector.detect(img2, None)
    # kp1, des1 = descriptor.compute(img1, kp1)
    # kp2, des2 = descriptor.compute(img2, kp2)

    brute_force_matcher = cv2.BFMatcher()
    matches = brute_force_matcher.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return points1, points2

def sift_matching(img1, img2):
    sift_detector = cv2.SIFT_create()

    kp1, des1 = sift_detector.detectAndCompute(img1,None)
    kp2, des2 = sift_detector.detectAndCompute(img2,None)

    # Did not work correctly !
    # descriptor = cv2.xfeatures2d.BEBLID_create(0.001)
    # kp1 = sift_detector.detect(img1, None)
    # kp2 = sift_detector.detect(img2, None)
    # kp1, des1 = descriptor.compute(img1, kp1)
    # kp2, des2 = descriptor.compute(img2, kp2)

    brute_force_matcher = cv2.BFMatcher()
    matches = brute_force_matcher.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.95*n.distance:
            good.append(m)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return points1, points2

"""
Calculates camera calibration matrix.
cx = 960 cy = 540 
Camera has no distortion.
Focal length is 100.
"""
def get_camera_parameters(vr2d_coords, vr3d_coords, image_size):

    cx = 960 
    cy = 540
    camera_focal_length = 100

    distortion_coefficients = np.array([[0],[0],[0],[0]], dtype=np.float32)

    camera_matrix = np.array([[camera_focal_length, 0,  cx],
                              [0, camera_focal_length,  cy],
                              [0,           0,           1]], dtype=np.float32)

    ret, new_camera_matrix, new_distortion_coefficients, rotation, translation = cv2.calibrateCamera(vr3d_coords, vr2d_coords,
                                     image_size, camera_matrix, distortion_coefficients, flags=(cv2.CALIB_USE_INTRINSIC_GUESS )) 

    return new_camera_matrix

def find_homography(points1, points2):

    H = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    
    return H


"""
Recovers the relative camera rotation and the translation from an estimated essential matrix and the corresponding points in two images
using cheirality check. Returns the number of inliers that pass the check.
For estimating essential matrix, i used keypoint matching and opencv's findEssentialMat function.
After that i converted rotation matrix to euler angles because of describing orientation of points.
"""
def estimate_rotation_and_translation(img1, img2, camera_matrix):

    points1, points2 = sift_matching(img1, img2) # Used SIFT for matching. I don't use SURF beacuse of licencing and ORB wont work correctly. 

    #H = find_homography(points1, points2)

    essential_matrix, mask_inliers = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC,  prob=0.999)

    points, rotation, translation, mask = cv2.recoverPose(essential_matrix, points1, points2, camera_matrix)

    # To Euler Angles referenced from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0

    euler_angles = np.array([x, y, z])
    

    return euler_angles, translation


"""
Main function that includes process steps.
"""
def calculate_6DoF(source_image, input_image):

    vr2d_coords, vr3d_coords = read_numpy_files("./inputs/vr2d.npy", "./inputs/vr3d.npy")

    camera_matrix = get_camera_parameters(vr2d_coords, vr3d_coords, image_size1)

    rotation_matrix, translation_matrix = estimate_rotation_and_translation(source_image, input_image, camera_matrix)

    origin_point = np.float32([0, 0, 0])
    projected_point, _ = cv2.projectPoints(origin_point, rotation_matrix, 
                                       translation_matrix, camera_matrix, None)

    (h, w) = input_image.shape[:2]
    input_image = cv2.arrowedLine(input_image, (w//2, h//2), (int(projected_point[0][0][0]), int(projected_point[0][0][1])), (255,0,0), thickness=3)
    

    return rotation_matrix, translation_matrix, input_image



if __name__ == "__main__":


    if not os.path.exists('./outputs'):
        os.makedirs('outputs')

    img1 = cv2.imread("./inputs/img1.png")
    img2 = cv2.imread("./inputs/img2.png")
    img3 = cv2.imread("./inputs/img3.png")

    image_size1 = (img1.shape[1],img1.shape[0])
    image_size2 = (img2.shape[1],img2.shape[0])
    image_size3 = (img3.shape[1],img3.shape[0])

    r1, t1, image1_6DoF = calculate_6DoF(img1, img2) #img1 to img2
    r2, t2, image2_6DoF = calculate_6DoF(img1, img3) #img1 to img3 


    print ("Img1 to img2 : ")
    print ("Rotation  : \n ", r1)
    print("Translation : \n ", t1)
    
    print ("Img1 to img3 : ")
    print ("Rotation  : \n ", r2)
    print("Translation : \n ", t2)

    f = open("./outputs/results.txt", "w")
    f.write("Img1 to img2 : ")
    f.write("\n")
    f.write("Rotation : ")
    f.write(str(r1))
    f.write("\n")
    f.write("Translation : ")
    f.write(str(t1))
    f.write("\n")
    f.write("Img1 to img3 : ")
    f.write("\n")
    f.write("Rotation : ")
    f.write(str(r2))
    f.write("\n")
    f.write("Translation : ")
    f.write(str(t2))
    f.write("\n")
    f.close()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(0, 0, 0, c='r', marker='o')
    ax.scatter(t1[0], t1[1], t1[2], c='g', marker='o')
    ax.scatter(t2[0], t2[1], t2[2], c='b', marker='o')

    plt.savefig('./outputs/6DoF_Plotting.png')
    plt.show()

    numpy_vertical = np.vstack((img1, image1_6DoF, image2_6DoF))
    numpy_horizontal = np.hstack((img1, image1_6DoF, image2_6DoF))

    cv2.imwrite("./outputs/vertical_result.png", numpy_vertical)
    cv2.imwrite("./outputs/horizontal_result.png", numpy_horizontal)
