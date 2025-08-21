# Image-Enhancement-and-Manipulation-Using-OpenCV-in-Python

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** VARNIKA.P  
- **Register Number:** 212223240170

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
bg_image = cv2.imread('Eagle_in_Flight.jpg',0)
```

#### 2. Print the image width, height & Channel.
```python
bg_image.shape
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(bg_image,cmap = 'gray')
plt.axis('off')
plt.title('BGR Image')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight2.jpg',bg_image)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
bgr_image = cv2.imread('Eagle_in_Flight.jpg')
rgb_color_img = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(rgb_color_img)
plt.axis('off')
plt.title('RGB Image')
plt.show()
rgb_color_img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
crop_img = rgb_color_img[25:410,200:545]
plt.imshow(crop_img)
plt.axis('off')
plt.title('Crop Image')
plt.show()
crop_img.shape
```

#### 8. Resize the image up by a factor of 2x.
```python
resize = cv2.resize(crop_img,None,fx=20,fy=20,interpolation=cv2.INTER_LINEAR)
resize.shape
```

#### 9. Flip the cropped/resized image horizontally.
```python
flip_img = cv2.flip(crop_img,1)
plt.imshow(flip_img)
plt.title('Flipped Image')
plt.axis('off')
plt.show()

img_eagle_flip_hor = cv2.flip(crop_img, 1)
img_eagle_flip_ver = cv2.flip(crop_img, 0)
img_eagle_flip_both = cv2.flip(crop_img, -1)

plt.figure(figsize = [18, 14])
plt.subplot(141); plt.imshow(img_eagle_flip_hor)
plt.title('Horizontal Flip')
plt.subplot(142); plt.imshow(img_eagle_flip_ver)
plt.title('Vertical Flip')
plt.subplot(143); plt.imshow(img_eagle_flip_both)
plt.title('Both Flipped')
plt.subplot(144); plt.imshow(crop_img)
plt.title('Original');
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
apollo_img = cv2.imread('Apollo-11-launch.jpg')
plt.imshow(apollo_img)
plt.axis('off')
plt.title('Apollo-11-launch')
plt.show()
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
# YOUR CODE HERE: use putText()
text_img = apollo_img.copy()
f_scale = 2
f_color = (255,255,255)
f_thickness = 3
text_img = cv2.putText(text_img, text, (315,715), font_face, f_scale, f_color, f_thickness, cv2.LINE_AA)

plt.imshow(text_img[:,:,::-1])
plt.title('Apollo-11-launch with description')
plt.axis('off')
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rectangle_img = apollo_img.copy()
rectangle_img = cv2.rectangle(rectangle_img, (515,55), (815,635), (225, 205, 255), thickness = 10, lineType = cv2.LINE_8)

```

#### 13. Display the final annotated image.
```python
plt.imshow(rectangle_img[:, :, ::-1])
plt.title('Apollo-11-launch with tower and rocket annotated - RGB')
plt.axis('off')
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img_boy = cv2.imread('boy.jpg', cv2.IMREAD_COLOR)
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
# matrix_ones = 
matrix = np.ones(img_boy.shape, dtype = 'uint8') * 45
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
bright_img = cv2.add(img_boy, matrix)
dark_img = cv2.subtract(img_boy, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize = [50,10])
plt.subplot(131); plt.imshow(dark_img[:, :, ::-1]); plt.title('Darker')
plt.subplot(132); plt.imshow(img_boy[:, :, ::-1]); plt.title('Original')
plt.subplot(133); plt.imshow(bright_img[:, :, ::-1]); plt.title('Brighter')
```

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = 
matrix2 = 
# img_higher1 = 
# img_higher2 = 
matrix1 = np.ones(img_boy.shape) * 0.75
matrix2 = np.ones(img_boy.shape) * 1.28

low  = np.uint8(cv2.multiply(np.float64(img_boy), matrix1))
high = np.uint8(cv2.multiply(np.float64(img_boy), matrix2))
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize = [14,8])
plt.subplot(131); plt.imshow(low[:, :, ::-1]); plt.title('Lower Contrast')
plt.subplot(132); plt.imshow(img_boy[:, :, ::-1]); plt.title('Original')
plt.subplot(133); plt.imshow(high[:, :, ::-1]); plt.title('Higher Contrast')
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(img_boy)

plt.figure(figsize = [20, 10])
plt.subplot(141); plt.imshow(r); plt.title('Red Channel')
plt.subplot(142); plt.imshow(g); plt.title('Green Channel')
plt.subplot(143); plt.imshow(b); plt.title('Blue Channel')

Merge = cv2.merge((r, g, b))

plt.subplot(144)
plt.imshow(Merge)
plt.title('Merged Output')
```

#### 21. Merged the R, G, B , displays along with the original image
```python
plt.figure(figsize = [18,14])
plt.subplot(121); plt.imshow(Merge); plt.axis('off'); plt.title('Merged Output')
plt.subplot(122); plt.imshow(img_boy[:,:,::-1]); plt.axis('off'); plt.title('Original Input')
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(img_boy, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

plt.figure(figsize = [18, 9])
plt.subplot(141); plt.imshow(h); plt.title('H Channel')
plt.subplot(142); plt.imshow(s); plt.title('S Channel')
plt.subplot(143); plt.imshow(v); plt.title('V Channel')

merge = cv2.merge([h,s,v])
merge_rgb = cv2.cvtColor(merge, cv2.COLOR_HSV2RGB)

plt.subplot(144); plt.imshow(merge_rgb); plt.title('Merged')
```
#### 23. Merged the H, S, V, displays along with original image.
```python
plt.figure(figsize = [17,11])
plt.subplot(121); plt.imshow(merge_rgb); plt.axis('off'); plt.title('Merged Output')
plt.subplot(122); plt.imshow(img_boy[:,:,::-1]); plt.axis('off'); plt.title('Original Input')
```

## Output:
- **i)** Read and Display an Image.  
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

