import cv2,os
import numpy as np
import time
import dlib


def video_init(is_2_write=False, save_path=None):
    writer = None
    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # default 480
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # default 640

    if is_2_write is True:
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap, height, width, writer

def detect_mouth(img,detector,predictor):
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    size = None

    faces, scores, idx = detector.run(img, 0)
    if len(faces):
        for k, d in enumerate(faces):
            x = []
            y = []
            height = d.bottom() - d.top()
            width = d.right() - d.left()

            shape = predictor(img, d)

            #----get the mouth
            for i in range(48, 68):
                x.append(shape.part(i).x)
                y.append(shape.part(i).y)
            height_part = height // 3
            width_part = width//3

            y_max = min((max(y) + height_part),img.shape[0])
            y_min = max((min(y) - height_part),0)
            x_max = min((max(x) + width_part),img.shape[1])
            x_min = max((min(x) - width_part),0)


            size = ((x_max-x_min),(y_max-y_min))

    return x_min, x_max, y_min, y_max, size


def Remenber_the_mask():
    # ----var
    frame_count = 0
    FPS = "Initialing"
    no_face_str = "No faces detected"
    mask_img_dir = r"./mask_img"


    # ----video streaming init
    cap, height, width, writer = video_init(is_2_write=False)

    # ----Dlib init
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    #----read mask image paths
    mask_paths = [file.path for file in os.scandir(mask_img_dir) if file.name.split(".")[-1] == 'png']
    mask_quantity = len(mask_paths)
    mask_num = np.random.randint(mask_quantity)
    img_mask_ori = cv2.imread(mask_paths[mask_num],cv2.IMREAD_UNCHANGED)

    # ----video nonstop capture
    while (cap.isOpened()):

        # ----get image
        ret, img = cap.read()

        if ret is True:
            #----mouth detection
            x_min, x_max, y_min, y_max, size = detect_mouth(img, detector, predictor)
            if size is not None:
                #----mask png file process
                img_mask = cv2.resize(img_mask_ori, size)
                img_mask_bgr = img_mask[:, :, :3]
                img_alpha_ch = img_mask[:, :, 3]
                _, item_mask = cv2.threshold(img_alpha_ch, 220, 255, cv2.THRESH_BINARY)
                human_mask = cv2.bitwise_not(item_mask)

                roi = img[y_min:y_min + size[1], x_min:x_min + size[0]]

                img_item_part = cv2.bitwise_and(img_mask_bgr, img_mask_bgr, mask=item_mask)
                img_human_part = cv2.bitwise_and(roi, roi, mask=human_mask)
                dst = cv2.add(img_human_part, img_item_part)
                img[y_min: y_min + size[1], x_min:x_min + size[0]] = dst

            # ----no faces detected
            else:
                cv2.putText(img, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # ----FPS count and claculation
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 10:
                FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                frame_count = 0

            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # ----image display
            cv2.imshow("demo by JohnnyAI", img)

            # ----key press detection
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                mask_num = np.random.randint(mask_quantity)
                img_mask_ori = cv2.imread(mask_paths[mask_num], cv2.IMREAD_UNCHANGED)

        else:
            print("get image failed")
            break

    # ----release
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Remenber_the_mask()
