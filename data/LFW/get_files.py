import os

data_root = 'LFWA'
persons = os.listdir(data_root)
fn = open(r'file_names.txt', 'w')
# fp = open(r'person_names.txt', 'w')
for person in persons:
    # fp.write(person + '\n')
    files = os.listdir(data_root + '/' + person)
    for img_file in files:
        fn.write(data_root + '/' + person + '/' +
                    img_file + '\n')
fn.close()
# fp.close()

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
