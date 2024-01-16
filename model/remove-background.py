from rembg import remove
from PIL import Image

input_image = Image.open('random.png')
output = remove(input_image)
output.save('random_output.PNG')
