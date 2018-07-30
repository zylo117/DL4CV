import itertools


def rect_overlapping(rect1, rect2):
    x01, y01, x02, y02 = rect1
    x11, y11, x12, y12 = rect2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def rect_coincide(rect1, rect2):
    x01, y01, x02, y02 = rect1
    x11, y11, x12, y12 = rect2
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)

    if rect_overlapping(rect1, rect2):
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        coincide = intersection / (area1 + area2 - intersection)
        return coincide, area1, area2
    else:
        return 0, area1, area2


def combination(iterable, num):
    result = []
    for c in itertools.combinations(iterable, num):
        result.append(c)
    return result


if __name__ == '__main__':
    com = combination([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], 2)
    print(com)
