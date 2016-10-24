import threading
import urllib
import os
import Image

def downLoadImg (person, lists_root, err_root, cor_root, VF_root):
    person_name = person[:-4]
    url_list_file = open(lists_root + '/' + person, 'r')
    url_list = url_list_file.readlines()
    err = 0
    f_err = open(err_root + '/' + person ,'w')
    f_cor = open(cor_root + '/' + person ,'w')
    cnt = 0
    url = url_list[0].split(' ')[1]
    img_name = url_list[0].split(' ')[0]
    write_name = VF_root + person_name + '/' + img_name + url[-4:]
    os.makedirs(os.path.dirname(write_name))
    for url_line in url_list:
        url = url_line.split(' ')[1]
        img_name = url_line.split(' ')[0]
        write_name = VF_root + person_name + '/' + img_name + url[-4:]
        # if not os.path.exists(os.path.dirname(write_name)):
            # os.makedirs(os.path.dirname(write_name))
        try:
            urllib.urlretrieve(url, write_name)
            try:
                Image.open(write_name)
                f_cor.write(url_line)
            except IOError:
                err += 1
                f_err.write(url_line)
        except IOError:
            f_err.write(url_line)
            err += 1
        cnt += 1
        if (cnt % 10 == 0):
            print person_name, 'has downloaded', cnt, 'images...'
            print person_name, 'occurs', err, 'errors'
    url_list_file.close()
    f_err.close()
    f_cor.close()

def main():
    print 'start downloading...'
    threadpool=[]
    lists_root = 'vgg_face_dataset/files'
    err_root = 'vgg_face_dataset/err'
    cor_root = 'vgg_face_dataset/cor'
    VF_root = 'VF/'
    persons = os.listdir(lists_root)
    for person in persons:
        th = threading.Thread(target = downLoadImg, args= (person, lists_root, err_root, cor_root, VF_root))
        threadpool.append(th)

    for th in threadpool:
        th.start()

    for th in threadpool :
        threading.Thread.join(th)

    print 'all Done'


if __name__ == '__main__':
    main()
