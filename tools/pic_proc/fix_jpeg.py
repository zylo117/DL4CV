from PIL import Image

def gif2jpeg(gif_file):
    f = open(gif_file, "rb")
    if len(f.readline()) > 0:
        f.close()
        img = Image.open(gif_file)
        img = img.convert("RGB")
        img.save(gif_file)
        return True
    else:
        return False

if __name__ == "__main__":
    file = "/Volumes/OSX_Data/Github/DL4CV/dataset/cats.vs.dogs_simple/cats/666.jpg"
    gif2jpeg(file)