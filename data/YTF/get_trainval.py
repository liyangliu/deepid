import pandas as pd
import numpy as np
import Image
import os

files_det = pd.read_csv('DlibSeetaDet.txt', sep = ' ')
files_ali = pd.read_csv('DlibSeetaAli.txt', sep = ' ')
# files_det = pd.read_csv('Seeta+DlibSeetaDet.txt', sep = ' ')
# files_ali = pd.read_csv('Seeta+DlibSeetaAli.txt', sep = ' ')
sorted_det = files_det.sort(columns = 'img_id')
# sorted_det.to_csv('DlibSeetaDetSorted.txt', sep = ' ', index = False)
sorted_ali = files_ali.sort(columns = 'img_id')
# sorted_ali.to_csv('DlibSeetaAliSorted.txt', sep = ' ', index = False)

person_before = sorted_det.iloc[0, 0].split('/')[1]
num_persons = 1
cnt_person = []
cnt_person.append(0)
cnt = 0
# cnt_img = 0
cnt_txt = 0
images_person = []
images_person_ori = []

crop_root = 'aligned_crop_DB/'
ftrain = open(r'train.txt', 'w')
fval = open(r'val.txt', 'w')
ftest = open(r'test.txt', 'w')

ftrainali = open(r'train_ali.txt', 'w')
fvalali = open(r'val_ali.txt', 'w')
ftestali = open(r'test_ali.txt', 'w')

for i in range(sorted_det['img_id'].count()):
    cnt += 1
    image = sorted_det.iloc[i, 0]
    write_name = '/'.join(image.split('/')[1:])
    write_name = crop_root + write_name
    person = image.split('/')[1]

    if person != person_before:
        num_images_person = cnt_person[-1]
        assert(num_images_person == len(images_person))
        idx = int(np.sum(cnt_person[:-1]))

        for j in range(num_images_person):
            left = sorted_det.iloc[j + idx, 1]
            top = sorted_det.iloc[j + idx, 2]
            width = sorted_det.iloc[j + idx, 3]
            height = sorted_det.iloc[j + idx, 4]

            # img = Image.open(images_person_ori[j])
            # img = np.array(img)
            # img = img[top : top + height, left : left + width, :]
            # img = Image.fromarray(img)
            # if not os.path.exists(os.path.dirname(images_person[j])):
                # os.makedirs(os.path.dirname(images_person[j]))
                # img.save(images_person[j])
                # cnt_img += 1
            # else:
                # img.save(images_person[j])
                # cnt_img += 1

            le_x = sorted_ali.iloc[j + idx, 1]
            le_y = sorted_ali.iloc[j + idx, 2]
            re_x = sorted_ali.iloc[j + idx, 3]
            re_y = sorted_ali.iloc[j + idx, 4]
            n_x = sorted_ali.iloc[j + idx, 5]
            n_y = sorted_ali.iloc[j + idx, 6]
            lm_x = sorted_ali.iloc[j + idx, 7]
            lm_y = sorted_ali.iloc[j + idx, 8]
            rm_x = sorted_ali.iloc[j + idx, 9]
            rm_y = sorted_ali.iloc[j + idx, 10]

            le_x_crop = le_x - left
            le_y_crop = le_y - top
            re_x_crop = re_x - left
            re_y_crop = re_y - top
            n_x_crop = n_x - left
            n_y_crop = n_y - top
            lm_x_crop = lm_x - left
            lm_y_crop = lm_y - top
            rm_x_crop = rm_x - left
            rm_y_crop = rm_y - top

            le_x_crop = max(le_x_crop, 0); le_x_crop = min(le_x_crop, width - 1)
            le_y_crop = max(le_y_crop, 0); le_y_crop = min(le_y_crop, height - 1)
            re_x_crop = max(re_x_crop, 0); re_x_crop = min(re_x_crop, width - 1)
            re_y_crop = max(re_y_crop, 0); re_y_crop = min(re_y_crop, height - 1)
            n_x_crop = max(n_x_crop, 0); n_x_crop = min(n_x_crop, width - 1)
            n_y_crop = max(n_y_crop, 0); n_y_crop = min(n_y_crop, height - 1)
            lm_x_crop = max(lm_x_crop, 0); lm_x_crop = min(lm_x_crop, width - 1)
            lm_y_crop = max(lm_y_crop, 0); lm_y_crop = min(lm_y_crop, height - 1)
            rm_x_crop = max(rm_x_crop, 0); rm_x_crop = min(rm_x_crop, width - 1)
            rm_y_crop = max(rm_y_crop, 0); rm_y_crop = min(rm_y_crop, height - 1)

            # assert(le_x_crop >= 0 and le_x_crop < width)
            # assert(le_y_crop >= 0 and le_y_crop < height)
            # assert(re_x_crop >= 0 and re_x_crop < width)
            # assert(re_y_crop >= 0 and re_y_crop < height)
            # assert(n_x_crop >= 0 and n_x_crop < width)
            # assert(n_y_crop >= 0 and n_y_crop < height)
            # assert(lm_x_crop >= 0 and lm_x_crop < width)
            # assert(lm_y_crop >= 0 and lm_y_crop < height)
            # assert(rm_x_crop >= 0 and rm_x_crop < width)
            # assert(rm_y_crop >= 0 and rm_y_crop < height)

            r = np.random.uniform()
            if r < 0.8:
                cnt_txt += 1
                ftrain.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
                ftrainali.write(images_person[j] + ' ' +
                                str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                                str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                                str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                                str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                                str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
            elif r < 0.9:
                cnt_txt += 1
                fval.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
                fvalali.write(images_person[j] + ' ' +
                                str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                                str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                                str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                                str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                                str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
            else:
                cnt_txt += 1
                ftest.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
                ftestali.write(images_person[j] + ' ' +
                                str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                                str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                                str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                                str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                                str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
        assert(j + 1 + idx == np.sum(cnt_person))

        images_person = []
        images_person_ori = []
        images_person.append(write_name)
        images_person_ori.append(image)
        num_persons += 1
        person_before = person
        cnt_person.append(1)
    else:
        cnt_person[num_persons - 1] += 1
        images_person.append(write_name)
        images_person_ori.append(image)
    if (cnt % 1000 == 0):
        print str(cnt), "images processed..."

num_images_person = cnt_person[-1]
assert(num_images_person == len(images_person))
idx = int(np.sum(cnt_person[:-1]))

for j in range(num_images_person):
    left = sorted_det.iloc[j + idx, 1]
    top = sorted_det.iloc[j + idx, 2]
    width = sorted_det.iloc[j + idx, 3]
    height = sorted_det.iloc[j + idx, 4]

    # img = Image.open(images_person_ori[j])
    # img = np.array(img)
    # img = img[top : top + height, left : left + width, :]
    # img = Image.fromarray(img)
    # if not os.path.exists(os.path.dirname(images_person[j])):
        # os.makedirs(os.path.dirname(images_person[j]))
        # img.save(images_person[j])
        # cnt_img += 1
    # else:
        # img.save(images_person[j])
        # cnt_img += 1

    le_x = sorted_ali.iloc[j + idx, 1]
    le_y = sorted_ali.iloc[j + idx, 2]
    re_x = sorted_ali.iloc[j + idx, 3]
    re_y = sorted_ali.iloc[j + idx, 4]
    n_x = sorted_ali.iloc[j + idx, 5]
    n_y = sorted_ali.iloc[j + idx, 6]
    lm_x = sorted_ali.iloc[j + idx, 7]
    lm_y = sorted_ali.iloc[j + idx, 8]
    rm_x = sorted_ali.iloc[j + idx, 9]
    rm_y = sorted_ali.iloc[j + idx, 10]

    le_x_crop = le_x - left
    le_y_crop = le_y - top
    re_x_crop = re_x - left
    re_y_crop = re_y - top
    n_x_crop = n_x - left
    n_y_crop = n_y - top
    lm_x_crop = lm_x - left
    lm_y_crop = lm_y - top
    rm_x_crop = rm_x - left
    rm_y_crop = rm_y - top

    le_x_crop = max(le_x_crop, 0); le_x_crop = min(le_x_crop, width - 1)
    le_y_crop = max(le_y_crop, 0); le_y_crop = min(le_y_crop, height - 1)
    re_x_crop = max(re_x_crop, 0); re_x_crop = min(re_x_crop, width - 1)
    re_y_crop = max(re_y_crop, 0); re_y_crop = min(re_y_crop, height - 1)
    n_x_crop = max(n_x_crop, 0); n_x_crop = min(n_x_crop, width - 1)
    n_y_crop = max(n_y_crop, 0); n_y_crop = min(n_y_crop, height - 1)
    lm_x_crop = max(lm_x_crop, 0); lm_x_crop = min(lm_x_crop, width - 1)
    lm_y_crop = max(lm_y_crop, 0); lm_y_crop = min(lm_y_crop, height - 1)
    rm_x_crop = max(rm_x_crop, 0); rm_x_crop = min(rm_x_crop, width - 1)
    rm_y_crop = max(rm_y_crop, 0); rm_y_crop = min(rm_y_crop, height - 1)

    # assert(le_x_crop >= 0 and le_x_crop < width)
    # assert(le_y_crop >= 0 and le_y_crop < height)
    # assert(re_x_crop >= 0 and re_x_crop < width)
    # assert(re_y_crop >= 0 and re_y_crop < height)
    # assert(n_x_crop >= 0 and n_x_crop < width)
    # assert(n_y_crop >= 0 and n_y_crop < height)
    # assert(lm_x_crop >= 0 and lm_x_crop < width)
    # assert(lm_y_crop >= 0 and lm_y_crop < height)
    # assert(rm_x_crop >= 0 and rm_x_crop < width)
    # assert(rm_y_crop >= 0 and rm_y_crop < height)

    r = np.random.uniform()
    if r < 0.8:
        cnt_txt += 1
        ftrain.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
        ftrainali.write(images_person[j] + ' ' +
                        str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                        str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                        str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                        str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                        str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
    elif r < 0.9:
        cnt_txt += 1
        fval.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
        fvalali.write(images_person[j] + ' ' +
                        str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                        str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                        str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                        str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                        str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
    else:
        cnt_txt += 1
        ftest.write(images_person[j] + ' ' + str(num_persons - 1) + '\n')
        ftestali.write(images_person[j] + ' ' +
                        str(le_x_crop) + ' ' + str(le_y_crop) + ' ' +
                        str(re_x_crop) + ' ' + str(re_y_crop) + ' ' +
                        str(n_x_crop) + ' ' + str(n_y_crop) + ' ' +
                        str(lm_x_crop) + ' ' + str(lm_y_crop) + ' ' +
                        str(rm_x_crop) + ' ' + str(rm_y_crop) + '\n')
assert(j + 1 + idx == np.sum(cnt_person))

assert(np.sum(cnt_person) == cnt)
assert(cnt == sorted_det['img_id'].count())
# assert(cnt_img == cnt)
assert(cnt_txt == cnt)
print str(cnt), "images processed..."

ftrain.close()
fval.close()
ftest.close()

ftrainali.close()
fvalali.close()
ftestali.close()
