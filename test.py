import cv2
import matplotlib.pyplot as plt
import dlib
from real_time_wearing_mask import detect_mouth

path = r".\mask_img\6.png"
# path = r"C:\Users\User\Desktop\mugshot_2.jpg"
img_path = r"C:\Users\User\Desktop\woman.jpg"

# ----Dlib init
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img = cv2.imread(path,cv2.IMREAD_UNCHANGED)

print(img.shape)






#----read omage
img_human = cv2.imread(img_path)
x_min, x_max, y_min, y_max, size = detect_mouth(img_human, detector, predictor)
if size is not None:
    img_mouth_part = img_human[y_min:y_min + size[1], x_min:x_min + size[0]]

    #----mask process
    img_mask_resized = cv2.resize(img,size)
    img_alpha_ch = img_mask_resized[:, :, 3]

    _, mask_item = cv2.threshold(img_alpha_ch, 220, 255, cv2.THRESH_BINARY)
    mask_human = cv2.bitwise_not(mask_item)

    img_mouth_part = cv2.bitwise_and(img_mouth_part,img_mouth_part,mask=mask_human)
    img_mask_part = cv2.bitwise_and(img_mask_resized[:,:,:3],img_mask_resized[:,:,:3],mask=mask_item)

    dst = cv2.add(img_mouth_part,img_mask_part)

    cv2.imwrite('test.jpg', dst)




plt.imshow(dst,cmap='gray')

plt.show()