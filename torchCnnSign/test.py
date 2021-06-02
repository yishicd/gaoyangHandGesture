import cv2
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import trochmodle


if __name__ == '__main__':

    for ind, img in enumerate(os.listdir('dataset/hands')):
        print(ind)
        print('dataset/hands'+'/'+str(ind))
        img = cv2.imread('dataset/hands'+'/'+str(ind)+'.jpg',cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)/255


        img = np.expand_dims(img, axis=0)

        print('手指', int(ind / 10))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = torch.tensor(img)

        img = img.to(torch.float32)
        newmoo = trochmodle.HandsNetModel()
        retC = torch.load('model_01.pt')
        newmoo.load_state_dict(retC['net'])

        output = newmoo(img)
        print("结果", output)
        break

