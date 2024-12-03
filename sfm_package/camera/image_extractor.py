import cv2

class ImageExtractor:
    def __init__(self, img_path, image_names):
        self.img_path = img_path
        self.image_names = image_names
        
    def extract_images(self):
        images = []
        for image_name in self.image_names:
            image = cv2.imread(self.img_path + "/" + image_name)
            # convert image to rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images