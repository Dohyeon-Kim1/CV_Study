import numpy as np
import torch
import torch.nn as nn
import cv2 as cv 
import matplotlib.pyplot as plt
#import winsound

class MLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hid1 = nn.Sequential(nn.Linear(784, 1024), nn.ReLU())
        self.hid2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.hid3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.out = nn.Linear(512,10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.hid1(x)
        x = self.hid2(x)
        x = self.hid3(x)
        x = self.out(x)
        return x

model = MLP()
model.load_state_dict(torch.load("dmlp_trained.pth"))
model.eval()

def reset():
    global img
       
    img = np.ones((200,520,3), dtype=np.uint8) * 255
    for i in range(5):
        cv.rectangle(img, (10+i*100,50), (10+(i+1)*100,150), (0,0,255))
    cv.putText(img, 'e:erase s:show r:recognition q:quit', (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1)

def grab_numerals():
    numerals=[]
    for i in range(5):
        roi = img[51:149, 11+i*100:9+(i+1)*100,0]
        roi = 255 - cv.resize(roi,(28,28), interpolation=cv.INTER_CUBIC)
        numerals.append(roi)  
    numerals = np.array(numerals)
    return numerals

def show():
    numerals = grab_numerals()
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def recognition():
    numerals = grab_numerals()
    numerals = numerals.reshape(5,784)
    numerals = numerals.astype(np.float32) / 255.0
    numerals = torch.FloatTensor(numerals)
    with torch.no_grad():
        res = model(numerals) # 신경망 모델로 예측
    class_id = torch.argmax(res, axis=1)
    for i in range(5):
        cv.putText(img, str(class_id[i].item()), (50+i*100,180), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    #winsound.Beep(1000,500)    
        
BrushSiz=4
LColor=(0,0,0)

def writing(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSiz, LColor, -1) 
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON: 
        cv.circle(img, (x,y), BrushSiz, LColor, -1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing', writing)

while(True):
    cv.imshow('Writing', img)
    key=cv.waitKey(1)
    if key==ord('e'):
        reset()
    elif key==ord('s'):
        show()        
    elif key==ord('r'):
        recognition()
    elif key==ord('q'):
        break
    
cv.destroyAllWindows()
