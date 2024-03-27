from PIL import Image

class ImageLoader:
    def load_image(self, filepath,mode='RGB'):
        # Load the image
        try:
            self.image = Image.open(filepath)
        except IOError:
            print(f"Error: Cannot open image {filepath}")
            return None
        
        # Convert the image to the desired format
        if mode.upper() == 'RGBA':
            self.image = self.image.convert('RGBA')
        elif mode.upper() == 'RGB':
            self.image = self.image.convert('RGB')
        elif mode.upper() == 'GRAY':
            self.image = self.image.convert('L')  # 'L' mode is for grayscale
        else:
            print(f"Error: Unknown mode {mode}")
        return self.image