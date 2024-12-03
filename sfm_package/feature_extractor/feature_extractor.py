import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, method, match_thresh, dy_thresh, sift_params = None):
        self.method = method
        self.match_thresh = match_thresh
        self.dy_thresh = dy_thresh
        
        if method == "sift":
            
            if sift_params is None:
                self.extractor = cv2.SIFT_create()
            else:
                contrast_thresh, edge_thresh, n_octave = sift_params
                self.extractor = cv2.SIFT_create(contrastThreshold=contrast_thresh, edgeThreshold=edge_thresh, nOctaveLayers=n_octave)
            self.matcher = cv2.NORM_L2
                
        elif method == "orb":
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.NORM_HAMMING
        else:
            raise ValueError("Invalid method")
        
    def extract_img_features(self, img, mask = None):
        kp, des = self.extractor.detectAndCompute(img, mask)
        return kp, des
    
    def match_features(self, des1, des2):
        bf = cv2.BFMatcher(self.matcher, crossCheck = True)
        matches = bf.match(des1, des2)
        return matches
    
    def identify_points(self, kp1, kp2, matches):
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return points1, points2
        
    def fundamental_threshold(self, points1, points2):
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, self.match_thresh)
        mask = mask.ravel() == 1
        return mask
    
    def dy_threshold(self, points1, points2):
        dy = points1[:, 1] - points2[:, 1]
        mask = np.abs(dy) < self.dy_thresh
        return mask
    
    def filter_matches(self, matches, points1, points2):
        f_mask = self.fundamental_threshold(points1, points2)#.astype(bool).ravel()  # Convert to 1D boolean array
        dy_mask = self.dy_threshold(points1, points2)#.astype(bool)  # Also boolean
        
        combined_mask = np.logical_and(f_mask, dy_mask)

        # Filter matches and points based on combined mask
        filtered_matches = [matches[i] for i in range(len(matches)) if combined_mask[i]]
        points1_inliers = points1[combined_mask]
        points2_inliers = points2[combined_mask]
        
        return filtered_matches, points1_inliers, points2_inliers
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matched_img