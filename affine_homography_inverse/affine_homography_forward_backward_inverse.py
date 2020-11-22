"""
Original skate video: https://www.youtube.com/watch?v=2S_U1pnLE-M

https://www.youtube.com/watch?v=MkcUgPhOlP8&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=28

https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/

draw keypoints: https://www.youtube.com/watch?v=USl5BHFq2H4

This result confirms that Affine from frame1 to frame2 is inverse to the Affine from frame2 to frame1

"""

import cv2
import numpy as np

#######################################################################################################################
def findMatchesBetweenImages(image_1, image_2, num_matches):

    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findAffine(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.

    # initiate two emtpy lists
    image_1_pts, image_2_pts = [], []

    for match in matches:
        # Get the matching keypoints for each of the images
        pt1 = image_1_kp[match.queryIdx].pt
        pt2 = image_2_kp[match.trainIdx].pt

        image_1_pts.append(pt1)
        image_2_pts.append(pt2)

    image_1_points = np.float64(image_1_pts).reshape(-1, 1, 2)
    image_2_points = np.float64(image_2_pts).reshape(-1, 1, 2)

    # Compute Affine
    M, _ = cv2.estimateAffine2D(image_1_points, image_2_points, cv2.RANSAC)

    M_aff = np.eye(3)

    M_aff[:2,:] = M

    return M_aff


def findHomography(image_1_kp, image_2_kp, matches):

    # initiate two emtpy lists
    image_1_pts, image_2_pts = [], []

    for match in matches:
        # Get the matching keypoints for each of the images
        pt1 = image_1_kp[match.queryIdx].pt
        pt2 = image_2_kp[match.trainIdx].pt

        image_1_pts.append(pt1)
        image_2_pts.append(pt2)

    image_1_points = np.float64(image_1_pts).reshape(-1, 1, 2)
    image_2_points = np.float64(image_2_pts).reshape(-1, 1, 2)

    # Compute homography (since our objects are planar) using RANSAC to reject outliers
    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC,
                                          5.0)  # transform img2 to img1's space

    return homography.astype(np.float64)


#######################################################################################################################

def main():
    # frame1 and frame2 were obtained by splitting the original video with ffmpeg
    # you can also read directly from the video
    frame1 = cv2.imread('frame0001.png')
    frame2 = cv2.imread('frame0002.png')

    # backward
    image_2_kp, image_1_kp, matches = findMatchesBetweenImages(frame2, frame1, num_matches = 100)

    homography_21 = findHomography(image_2_kp, image_1_kp, matches)
    # Inverse
    homography_21_inv = np.linalg.inv(homography_21)

    M_aff_21 = findAffine(image_2_kp, image_1_kp, matches)
    # Inverse
    M_aff_21_inv = np.linalg.inv(M_aff_21)

    # forward
    image_1_kp, image_2_kp, matches = findMatchesBetweenImages(frame1, frame2, num_matches = 100)

    homography_12 = findHomography(image_1_kp, image_2_kp, matches)

    M_aff_12 = findAffine(image_1_kp, image_2_kp, matches)

    print(f'\nHomography_21:\n{homography_21}.')

    print(f'\nHomography_21_inv:\n{homography_21_inv}.')
    print(f'\nhomography_12:\n{homography_12}.')

    print(f'\nAffine_21:\n{M_aff_21}.')

    print(f'\nAffine_21_inv:\n{M_aff_21_inv}.')
    print(f'\nAffine_12:\n{M_aff_12}.')

    img_feature1 = cv2.drawKeypoints(frame1, image_1_kp, None)
    img_feature2 = cv2.drawKeypoints(frame2, image_2_kp, None)

    cv2.imshow("Frame 1", img_feature1)
    cv2.imshow("Frame 2", img_feature2)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    cv2.imwrite("frame0001_with_features.png", img_feature1)
    cv2.imwrite("frame0002_with_features.png", img_feature2)

if __name__ == "__main__":
    main()