from PIL import Image, ImageEnhance
import os
from IPython.display import display

def enhance_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(factor)
    
    #display(image)
    #display(image_enhanced)
    return image_enhanced

def save_image(image, filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    image.save(filepath+filename+".jpg")
