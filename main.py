import os
import cv2
from sfm_package.camera.camera import Camera
from sfm_package.camera.cam_data_extractor import CameraDataExtractor
from sfm_package.camera.image_extractor import ImageExtractor

from sfm_package.feature_extractor.feature_extractor import FeatureExtractor

def main():
    SEABED_DIR = os.path.join(os.path.dirname(__file__), 'sfm_package/data_files/seabed_images')
    CAM_CALIB = os.path.join(os.path.dirname(__file__), 'sfm_package/data_files/calib_stereo_diver.pkl')
    POSE_DATA = os.path.join(os.path.dirname(__file__), 'sfm_package/data_files/camera_pose_data.pkl')
    
    # load camera calibration data
    cam_data_extractor = CameraDataExtractor(CAM_CALIB, POSE_DATA)
    K, D, names = cam_data_extractor.extract()
    img_extractor = ImageExtractor(SEABED_DIR, names)
    images = img_extractor.extract_images()
    
    photo_camera = Camera(K, D)
    
    for i, image in enumerate(images):
        photo_camera.store_frame(image)
    
    method = "sift"
    match_thresh = 0.3
    dy_thresh = 35
    sift_params = (0.002, 2, 12) # contrast threshold, edge threshold, number of octaves
    # sift_params = None
    feature_extractor = FeatureExtractor(method, match_thresh, dy_thresh, sift_params)
    prev_kp, prev_des = feature_extractor.extract_img_features(photo_camera.frames[0])
    for i in range(1, len(photo_camera.frames)):
        kp, des = feature_extractor.extract_img_features(photo_camera.frames[i])
        matches = feature_extractor.match_features(prev_des, des)
    
        pts1, pts2 = feature_extractor.identify_points(prev_kp, kp, matches)
    
        match_inliers, pts1_inliers, pts2_inliers = feature_extractor.filter_matches(matches, pts1, pts2)
        
        
    

if __name__ == "__main__":
    main()