from rembg import remove
from PIL import Image

input_image = Image.open('testing.png')
output = remove(input_image)
output.save('testing_output.png')
