import pickle as pk

class CameraDataExtractor:
    def __init__(self, data_path, img_name_dir):
        self.data_path = data_path
        self.img_name_dir = img_name_dir
        
    def extract(self):
        with open(self.data_path, 'rb') as f:
            data = pk.load(f)
        with open(self.img_name_dir, 'rb') as f:
            img_names = pk.load(f)    
            
        K = data['Kl']
        D = data['Dl']
        
        names = img_names['filenames_left']
        
        return K, D, names