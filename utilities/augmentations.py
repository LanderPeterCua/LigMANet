from PIL import Image, ImageEnhance

def enhance_contrast(image, factor):
  enhancer = ImageEnhance.Contrast(image)
  image_enhanced = enhancer.enhance(factor)
  
  return image_enhanced