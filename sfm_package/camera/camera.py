import cv2

class Camera:
    def __init__(self, K_matrix, dist_matrix):
        # instrinsics and distortion
        self.K_matrix = K_matrix
        self.dist_matrix = dist_matrix
        # images stored in camera
        self.frames = []

    def undistort_image(self, image):
        return cv2.undistort(image, self.K_matrix, self.dist_matrix)
    
    def store_frame(self, frame):
        undistorted_frame = self.undistort_image(frame)
        self.frames.append(undistorted_frame)