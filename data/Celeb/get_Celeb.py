import base64
import os

cnt = 0
for line in open('MsCelebV1-Faces-Aligned.tsv'):
    line = line.split('\t')
    person = line[0]
    img = line[-1]
    img = base64.b64decode(img)
    rank = line[1]
    faceID = line[4]
    img_name = 'Celeb/' + person + '/' + rank + '-' + faceID + '.jpg'
    dir_name = os.path.dirname(img_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_img = open(img_name, 'wb')
    f_img.write(img)
    f_img.close()
    cnt += 1
    if (cnt % 1000 == 0):
        print cnt, 'images written...'
print cnt, 'images written...'
