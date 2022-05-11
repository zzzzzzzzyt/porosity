import numpy as np
import os
import cv2 as cv

dirPath = 'oldPhoto'
newDir = 'fillPhoto'

if not os.path.exists(newDir):
    os.makedirs(newDir)

order = [int(i.strip(".png")) for i in os.listdir(dirPath) if i.endswith(".png")]
jpgList = [f"{i}.png" for i in sorted(order)]  # 直接读取可能非顺序帧

for i, png in enumerate(jpgList):
    old = dirPath + f'/{png}'
    img = cv.imread(old)
    imGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imGray[imGray < 100] = 0
    imGray[imGray >= 100] = 255
    mask = 255 - imGray
    marker = np.zeros_like(imGray)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255
    marker_0 = marker.copy()
    SE = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imGray
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    filling = cv.morphologyEx(filling, cv.MORPH_OPEN, kernel, 1)
    contours, hierarchy = cv.findContours(filling, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (212, 255, 127), 1)
    new = newDir + f'/{png}'
    cv.imwrite(new, img)
    print(f'{i + 1} / {len(jpgList)}')
















