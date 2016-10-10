cnt = 0
for line in open('MsCelebV1-Faces-Aligned.tsv'):
    cnt += 1
    if (cnt % 10000 == 0):
        print cnt, 'images seen...'
print cnt, 'images ...'
