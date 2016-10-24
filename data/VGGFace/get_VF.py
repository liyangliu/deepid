import urllib
import os

lists_root = 'vgg_face_dataset/files'
VF_root = 'VF/'
persons = os.listdir(lists_root)
cnt = 0

for person in persons:
    person_name = person[:-4]
    url_list_file = open(lists_root + '/' + person, 'r')
    url_list = url_list_file.readlines()
    for url_line in url_list:
        url = url_line.split(' ')[1]
        img_name = url_line.split(' ')[0]
        write_name = VF_root + person_name + '/' + img_name + '.jpg'
        if not os.path.exists(os.path.dirname(write_name)):
            os.makedirs(os.path.dirname(write_name))
        urllib.urlretrieve(url, write_name)
        cnt += 1
        if (cnt % 10 == 0):
            print cnt, 'images downloaded...'
print cnt, 'images downloaded...'
