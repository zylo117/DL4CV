import sys


def jpgfix(name):
    sig = b"\xFF\xD8\xFF\xDB"
    with open(name, "rb") as fd:
        # fd.seek(len(sig), 0)
        jpg = fd.read()
        pos = jpg.find(sig)
        if pos < 0:
            print("Lost signature")
            jpg = sig + jpg
        else:
            jpg = jpg[pos:]
    with open(name, "wb") as fd:
        fd.seek(0, 0)
        print("size is:", len(jpg))
        fd.write(jpg)
        fd.close()

if __name__ == "__main__":
    try:
        while True:
            jpgfix(sys.argv[1])
    except:
        print("Done")