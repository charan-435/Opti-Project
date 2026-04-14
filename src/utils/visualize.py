import cv2
import matplotlib.pyplot as plt
import os

def show_imgs(img_list, titles=None):
    # little helper to see what we are doing
    n = len(img_list)
    plt.figure(figsize=(n*4, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(img_list[i], cmap='gray')
        if titles: plt.title(titles[i])
        plt.axis('off')
    plt.show()

# some test logic
if __name__ == "__main__":
    p = "data/processed/yes"
    if os.path.exists(p):
        f = os.listdir(p)[0]
        img = cv2.imread(os.path.join(p, f), 0)
        show_imgs([img], ["test yes"])
