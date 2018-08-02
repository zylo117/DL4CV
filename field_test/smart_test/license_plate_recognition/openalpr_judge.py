import requests
import base64
import json

# Sample image file is available at http://plates.openalpr.com/ea7the.jpg
IMAGE_PATH = './test/0111.jpg'
SECRET_KEY = 'sk_DEMODEMODEMODEMODEMODEMO'

with open(IMAGE_PATH, 'rb') as image_file:
    img_base64 = base64.b64encode(image_file.read())

url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=cn&secret_key=%s' % (SECRET_KEY)
r = requests.post(url, data=img_base64)
r = r.json()
js = json.dumps(r, indent=2)
print(js)
