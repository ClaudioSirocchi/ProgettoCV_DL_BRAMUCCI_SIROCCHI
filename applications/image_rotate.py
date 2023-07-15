import glob
import os
import cv2

def rotate(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for filename in glob.iglob(source + '/**/*.png', recursive=True):
        src = cv2.imread(filename)
        image = cv2.rotate(src, cv2.ROTATE_180)
        filename= filename.replace(source + '/', '')
        info = filename.split('/')
        print(f'paziente {info[0]}, immagine {info[1]}')

        if not os.path.exists(destination + '/' + info[0]):
            os.makedirs(destination + '/' + info[0])

        """
        with open(destination + '/' + filename, 'w') as f:
            f.write(f'paziente {info[0]}, immagine {info[1]}')
            f.write('\n')
        """

        cv2.imwrite(destination + '/' + filename, image)




def main():
    rotate('../data/babypose/images/val2017', '../data/babypose/images/val2017_rotated')


if __name__ == '__main__':
    main()
