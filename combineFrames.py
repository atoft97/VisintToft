import cv2
from os import listdir, rename
numbers = [5]

for number in numbers:
    paddedNumber = str(number).zfill(5)

    startPath = "Video"+ paddedNumber +"_Ano/images_int"

    files = listdir(startPath)

    #print(files)

    #rename("frame_000000.PNGima","frame_000000.PNG")

    for filename in files:
        #print(filename[:-3])
        #rename(filename, filename[:-3])

        #rename("data/Video"+ paddedNumber +"_Ano/images_int/" + filename, "data/Video"+ paddedNumber +"_Ano/images_int/" + filename[:-3])
        #rename("data/Video"+ paddedNumber +"_Ano/images_int/" + filename, "data/Video"+ paddedNumber +"_Ano/images_int/" + filename[:-3])
        #rename("data/Video"+ paddedNumber +"_Ano/images_range/" + filename, "data/Video"+ paddedNumber +"_Ano/images_range/" + filename[:-3])

        img1 = cv2.imread("Video"+ paddedNumber +"_Ano/images_int/" + filename, 0)
        img2 = cv2.imread("Video"+ paddedNumber +"_Ano/images_ambi/" + filename, 0)
        img3 = cv2.imread("Video"+ paddedNumber +"_Ano/images_range/" + filename, 0)

        img = cv2.merge((img1, img2, img3))

        print(filename)
        cv2.imwrite("Video"+ paddedNumber +"_Ano/combined/" + filename, img)