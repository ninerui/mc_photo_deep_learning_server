import sys

import cv2
import numpy as np

if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        output_path = args.pop(-1)
        img_path_list = args
        # assert len(args) == 2
        # img_path_list, output_path = args
        # img_path_list = img_path_list.split(',')
        img_count = len(img_path_list)
        assert 0 <= img_count <= 6
        if img_count == 1:
            pass
        elif img_count == 2:
            img_0 = cv2.imread(img_path_list[0])
            img_1 = cv2.imread(img_path_list[1])
            ratio_0 = img_0.shape[0] / img_0.shape[1]
            ratio_1 = img_1.shape[0] / img_1.shape[1]
            if (ratio_0 >= 1.0) and ((ratio_0 - 1.0) > (1.0 - ratio_1)):
                img_0 = cv2.resize(img_0, (1005, 501))
                img_1 = cv2.resize(img_1, (1005, 501))
                a = np.zeros((3, 1005, 3), np.uint8)
                a.fill(255)
                img = np.hstack((img_0, a, img_1))

            else:
                img_0 = cv2.resize(img_0, (501, 1005))
                img_1 = cv2.resize(img_1, (501, 1005))
                a = np.zeros((1005, 3, 3), np.uint8)
                a.fill(255)
                img = np.hstack((img_0, a, img_1))
        elif img_count == 3:
            img_0 = cv2.imread(img_path_list[0])
            img_1 = cv2.imread(img_path_list[1])
            img_2 = cv2.imread(img_path_list[2])

            img_1 = cv2.resize(img_1, (501, 501))
            img_2 = cv2.resize(img_2, (501, 501))
            a = np.zeros((501, 3, 3), np.uint8)
            a.fill(255)
            img = np.hstack((img_1, a, img_2))

            img_0 = cv2.resize(img_0, (1005, 501))
            a = np.zeros((3, 1005, 3), np.uint8)
            a.fill(255)
            img = np.vstack((img_0, a, img))
        elif img_count == 4:
            img_0 = cv2.imread(img_path_list[0])
            img_1 = cv2.imread(img_path_list[1])
            img_2 = cv2.imread(img_path_list[2])
            img_3 = cv2.imread(img_path_list[3])

            img_0 = cv2.resize(img_0, (501, 501))
            img_1 = cv2.resize(img_1, (501, 501))
            img_2 = cv2.resize(img_2, (501, 501))
            img_3 = cv2.resize(img_3, (501, 501))
            a = np.zeros((501, 3, 3), np.uint8)
            a.fill(255)
            img_0_1 = np.hstack((img_0, a, img_1))
            img_2_3 = np.hstack((img_2, a, img_3))
            a = np.zeros((3, 1005, 3), np.uint8)
            a.fill(255)
            img = np.vstack((img_0_1, a, img_2_3))
        elif img_count == 5:
            img_0 = cv2.imread(img_path_list[0])
            img_1 = cv2.imread(img_path_list[1])
            img_2 = cv2.imread(img_path_list[2])
            img_3 = cv2.imread(img_path_list[3])

            img_0 = cv2.resize(img_0, (501, 332))
            img_1 = cv2.resize(img_1, (501, 332))
            img_2 = cv2.resize(img_2, (501, 332))
            img_3 = cv2.resize(img_3, (501, 332))
            a = np.zeros((332, 3, 3), np.uint8)
            a.fill(255)
            img_0_1 = np.hstack((img_0, a, img_1))
            img_2_3 = np.hstack((img_2, a, img_3))
            a = np.zeros((3, 1005, 3), np.uint8)
            a.fill(255)
            img = np.vstack((img_0_1, a, img_2_3))

            img_4 = cv2.imread(img_path_list[4])
            img_4 = cv2.resize(img_4, (1005, 501))
            a = np.zeros((3, 1005, 3), np.uint8)
            a.fill(255)
            img = np.vstack((img_4, a, img))
        elif img_count == 6:
            img_0 = cv2.imread(img_path_list[0])
            img_1 = cv2.imread(img_path_list[1])
            img_2 = cv2.imread(img_path_list[2])
            img_3 = cv2.imread(img_path_list[3])
            img_4 = cv2.imread(img_path_list[4])
            img_5 = cv2.imread(img_path_list[5])

            img_0 = cv2.resize(img_0, (332, 332))
            img_1 = cv2.resize(img_1, (332, 332))
            img_2 = cv2.resize(img_2, (332, 332))
            a = np.zeros((332, 3, 3), np.uint8)
            a.fill(255)
            img_0_1_2 = np.hstack((img_0, a, img_1, a, img_2))

            img_3 = cv2.resize(img_3, (332, 332))
            img_4 = cv2.resize(img_4, (332, 332))
            a = np.zeros((3, 332, 3), np.uint8)
            a.fill(255)
            img_3_4 = np.vstack((img_3, a, img_4))

            img_5 = cv2.resize(img_5, (667, 667))
            a = np.zeros((667, 3, 3), np.uint8)
            a.fill(255)
            img_345 = np.hstack((img_3_4, a, img_5))

            a = np.zeros((3, 1002, 3), np.uint8)
            a.fill(255)
            img = np.vstack((img_0_1_2, a, img_345))
        cv2.imwrite(output_path, img)
        sys.stdout.write('生成图片成功: {}'.format(output_path))
    except:
        sys.stdout.write('生成图片失败')
