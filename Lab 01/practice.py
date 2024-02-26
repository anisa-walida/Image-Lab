import numpy as np
import cv2

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img_bordered = cv2.copyMakeBorder(src=img, top=25, bottom=25, left=25, right=25, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('grayscaled image', img)
cv2.imshow('bordered image', img_bordered)

out = np.zeros((512, 512))  # , dtype=np.uint8)
kernel = (1 / 273) * np.array([[1, 4, 7, 4, 1],
                               [4, 16, 26, 16, 4],
                               [7, 26, 41, 26, 7],
                               [4, 16, 26, 16, 4],
                               [1, 4, 7, 4, 1]])

n = int(kernel.shape[0] / 2)
for x in range(n,img.shape[0]-n):
    for y in range(n,img.shape[1]-n):
        res = 0
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):  
                g = kernel.item(i, j)
                f = img.item(x - i, y - j)
                res += g*f

        out[x, y] = res

print(out)
cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
print(out)
cv2.imshow('normalized output image', out)

cv2.waitKey(0)
cv2.destroyAllWindows()
