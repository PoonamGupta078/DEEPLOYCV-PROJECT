import os
import matplotlib.pyplot as plt
from PIL import Image
def show(name,n,m,i,Title):
    plt.subplot(n,m,i)
    plt.imshow(name, cmap="gray")
    plt.title(Title)
    plt.axis('off')
    plt.show()

path=[]

img_title=[]
x=1
image_path=r'C:\Users\HP\OneDrive\Desktop\doremon images'
for img in os.listdir(image_path):
    image=os.path.join(image_path,img)
    path.append(image)
    img_title.append("image"+str(x))
    x=x+1

plt.figure(figsize=(10,10))
#for name,i,Title in zip(path,range(1,5),img_title):
img=Image.open(path[0])
img=img.convert("RGBA")
show(img,1,2,1,img_title[0])
Img=Image.open(path[1])
IMG=Img.convert("RGBA")
show(Img,1,2,2,img_title[1])