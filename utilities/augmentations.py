from PIL import Image, ImageEnhance
from IPython.display import display

def enhance_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(factor)
    
    #display(image)
    #display(image_enhanced)
    return image_enhanced