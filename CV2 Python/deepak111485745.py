import cv2


# Read an image file and return
def read_image(name):
    return cv2.imread(name, cv2.IMREAD_COLOR)


# show an image on window and wait for the key event
def show(image):
    cv2.imshow('output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Add a scalar value to each pixel of the image
def add(image, scalar):
    return cv2.add(image, scalar)


# Subtract a scalar value from each pixel of the image
def subtract(image, scalar):
    return cv2.subtract(image, scalar)


# Divide image with a scalar value
def divide(image, scalar):
    return cv2.divide(image, scalar)


# Multiply image with a scalar value
def multiply(image, scalar):
    return cv2.multiply(image, scalar)


# Resize a image by given factor
def resize_image(image, factor):
    return cv2.resize(
        image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


img = read_image('horse.jpeg')
img_add = add(img, 50)
img_subtract = subtract(img, 50)
img_multiply = multiply(img, 2)
img_divide = divide(img, 2)
img_resize = resize_image(img, float(1) / 2)
cv2.imshow('Original', img)
cv2.imshow('Addition', img_add)
cv2.imshow('Subtraction', img_subtract)
cv2.imshow('Multiplication', img_multiply)
cv2.imshow('Divide', img_divide)
cv2.imshow('Resize', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
