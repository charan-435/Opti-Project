import cv2 as cv
import os
input_folder='no'
output_folder='no_enhance'


os.makedirs(output_folder,exist_ok=True)



clahe=cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg','.png','.jpeg')):

        img_path=os.path.join(input_folder,filename)
        img=cv.imread(img_path,cv.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        enhanced = clahe.apply(img)

        output_path=os.path.join(output_folder,filename)
        cv.imwrite(output_path,enhanced)

        print(f"Enhanced image saved to: {output_path}")
        