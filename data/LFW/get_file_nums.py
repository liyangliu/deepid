import os

data_root = 'lfw'
persons = os.listdir(data_root)
fn = open(r'file_nums.txt', 'w')
for person in persons:
    files = os.listdir(data_root + '/' + person)
    fn.write(person + ' ' + str(len(files)) + '\n')
fn.close()

# data_root = 'aligned_crop_DB'
# persons = os.listdir(data_root)
# fn = open(r'file_names_crop.txt', 'w')
# fp = open(r'person_names_crop.txt', 'w')
# for person in persons:
    # fp.write(person + '\n')
    # dirs = os.listdir(data_root + '/' + person)
    # num_dirs = len(dirs)
    # for subdir in dirs:
        # files = os.listdir(data_root + '/' + person + '/' + subdir)
        # for img_file in files:
            # fn.write(data_root + '/' + person + '/' + subdir + '/' +
                     # img_file + '\n')
# fn.close()
# fp.close()
