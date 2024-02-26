import numpy as np 
import cv2 
import math
    
def convolution(img,kernel,c_x,c_y):
    n = kernel.shape[0] // 2
    m = kernel.shape[1] // 2
    padd_top = c_x
    padd_bottom = kernel.shape[0]-c_x -1
    padd_left = c_y
    padd_right = kernel.shape[1]-c_y -1
   
    img_bordered = cv2.copyMakeBorder(src=img,top= padd_top, bottom= padd_bottom, left= padd_left,right= padd_right, borderType=cv2.BORDER_CONSTANT)
    out = img_bordered.copy()  # , dtype=np.uint8)
    
    

   
    for x in range(c_x,img.shape[0]-padd_bottom-n):
        for y in range(c_y,img.shape[1]-padd_right-m):
            res = 0
            for i in range(-n, n + 1):
                for j in range(-m, m + 1):  
                    g = kernel[i+n, j+m]
                    f = img_bordered[x - i, y - j]
                    res += g*f

            out[x, y] = res
            
    print(out)
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    print(f"normalized {out}")
    # crop image to original image
    out = out[c_x: -padd_bottom, c_y:-padd_right]
    cv2.imshow('Input', img_bordered)
    cv2.imshow('Output Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out
        
 

def gaussian_kernel(sigma_x,sigma_y):
    k_sizex = int (5*sigma_x)
    k_sizey = int (5*sigma_y)
    if(k_sizex % 2 == 0):
        k_sizex+=1
    if(k_sizey % 2 == 0):
        k_sizey+=1    
    #normalization constant
    norm = 1/( 2*3.141592*sigma_x*sigma_y)
    
    gaussian = np.zeros((k_sizex,k_sizey),np.float32)
    
    
    for x in range(k_sizex):
        for y in range(k_sizey):
            px = (x**2)/(sigma_x**2)
            py =  (y**2)/(sigma_y**2)
            p = (px + py )/2
            p = math.exp(-p)
            gaussian[x,y]= p*norm
    
    print(gaussian)
    
    return gaussian


def mean_kernel(row,col):
   
    meann = (1 / (row * col)) * np.ones((row,col), dtype=np.uint8)

    print(meann)
    return meann
def laplacian_kernel():
    laplacian = np.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]],np.float32)
    print(laplacian)
    return laplacian
def sobel_kernel():
    blur = np.array([[1],
                  [2],
                  [1]], dtype=np.float32)

    derivative = np.array([1, 0, -1], dtype=np.float32)

    h_sobel_kernel = blur * derivative
    return h_sobel_kernel
kernel = mean_kernel(3,3)
print("1. Gausian Filer")
print("2. Mean Filer")
print("3. Laplacian Filer")
print("4. Sobel Filer")
choice_f =int(input("Enter Filter: "))
print("1. Grayscale")
print("2. RGB")

choice_i = int(input("Enter Image type: "))

c_x = int(input("Enter center(x) :"))
c_y = int(input("Enter center(y) :"))
if(choice_f==1):
   sigma_x = float(input("Enter sigma_x : "))
   sigma_y = float(input("Enter sigma_y : "))
   kernel = gaussian_kernel(sigma_x, sigma_y)
elif(choice_f==2):
    row = int(input("Enter row :"))
    col = int(input("Enter col: "))
    kernel = mean_kernel(row,col) 
elif(choice_f==3):
   kernel = laplacian_kernel()  
else:
  kernel = sobel_kernel() 

  
if(choice_i==1):
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    convolution(img,kernel,c_x,c_y)
       
else:
    
    img = cv2.imread('Lena.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.imshow('input image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    b, g, r = cv2.split(img)

    b_new = convolution(b,kernel,c_x,c_y)  
    g_new = convolution(g,kernel,c_x,c_y)  
    r_new = convolution(r,kernel,c_x,c_y)  
      
    merged = cv2.merge((b_new, g_new, r_new))
    cv2.imshow('input image', img)
    cv2.imshow("merged", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (b_hsv , g_hsv , r_hsv) = cv2.split(hsv_image)
    b_newhsv = convolution(b_hsv,kernel,c_x,c_y)  
    g_newhsv = convolution(g_hsv,kernel,c_x,c_y)  
    r_newhsv = convolution(r_hsv,kernel,c_x,c_y)
    merged_hsv = cv2.merge((b_newhsv, g_newhsv, r_newhsv))
    cv2.imshow('input image', img)
    cv2.imshow("merged_hsv", merged_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    diff= merged-merged_hsv
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
#convolution(img,mean_kernel(),1,1)  

