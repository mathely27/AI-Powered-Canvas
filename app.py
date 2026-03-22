import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

if not os.path.exists('hand_landmarker.task'):
    print('ERROR')
    exit()

bo = python.BaseOptions(model_asset_path='hand_landmarker.task')
op = vision.HandLandmarkerOptions(base_options=bo,num_hands=2,min_hand_detection_confidence=0.4,min_hand_presence_confidence=0.4,min_tracking_confidence=0.4,running_mode=vision.RunningMode.VIDEO)
det = vision.HandLandmarker.create_from_options(op)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,540)

cvs = None
pp = {'Left':(0,0),'Right':(0,0)}
sp = {'Left':(0,0),'Right':(0,0)}

glow = (128,128,0)
neon = (255,255,0)
white = (255,255,255)

HC = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

sm = 0.5
st = time.time()
print('Magic Canvas Started!')

while True:
    ok,fr = cap.read()
    if not ok:
        break
    fr = cv2.flip(fr,1)
    h,w,_ = fr.shape
    if cvs is None:
        cvs = np.zeros((h,w,3),dtype=np.uint8)
    bg = np.zeros((h,w,3),dtype=np.uint8)
    rgb = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    mi = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    ts = int((time.time()-st)*1000)
    res = det.detect_for_video(mi,ts)
    hf = {'Left':False,'Right':False}
    mt = ''
    if res.hand_landmarks:
        for i,hl in enumerate(res.hand_landmarks):
            ht = res.handedness[i][0].category_name
            hf[ht] = True
            for cn in HC:
                x1,y1 = int(hl[cn[0]].x*w),int(hl[cn[0]].y*h)
                x2,y2 = int(hl[cn[1]].x*w),int(hl[cn[1]].y*h)
                cv2.line(bg,(x1,y1),(x2,y2),glow,2,cv2.LINE_AA)
                cv2.line(bg,(x1,y1),(x2,y2),neon,1,cv2.LINE_AA)
            ix,iy = int(hl[8].x*w),int(hl[8].y*h)
            ox,oy = sp[ht]
            if ox==0 and oy==0:
                ox,oy = ix,iy
            nx = int(sm*ix+(1-sm)*ox)
            ny = int(sm*iy+(1-sm)*oy)
            sp[ht] = (nx,ny)
            pmx,pmy = int(hl[9].x*w),int(hl[9].y*h)
            iu = hl[8].y<hl[6].y
            mu = hl[12].y<hl[10].y
            ru = hl[16].y<hl[14].y
            pu = hl[20].y<hl[18].y
            palm = iu and mu and ru and pu
            fi = iu and not mu and not ru and not pu
            px,py = pp[ht]
            if fi:
                mt = 'DRAW'
                cv2.circle(bg,(nx,ny),4,glow,cv2.FILLED)
                cv2.circle(bg,(nx,ny),2,white,cv2.FILLED)
                if px==0 and py==0:
                    px,py = nx,ny
                cv2.line(cvs,(px,py),(nx,ny),white,2,cv2.LINE_AA)
                pp[ht] = (nx,ny)
            elif palm:
                mt = 'ERASE'
                cv2.circle(cvs,(pmx,pmy),35,(0,0,0),cv2.FILLED)
                pp[ht] = (0,0)
            else:
                pp[ht] = (0,0)
    for ht in ['Left','Right']:
        if not hf[ht]:
            pp[ht] = (0,0)
            sp[ht] = (0,0)
    out = cv2.add(bg,cvs)
    if mt:
        cv2.putText(out,mt,(15,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,neon,1,cv2.LINE_AA)
    cv2.putText(out,'Index=Draw | Palm=Erase | C=Clear | Q=Quit',(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(60,60,60),1,cv2.LINE_AA)
    cv2.imshow('Magic Canvas',out)
    k = cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
    elif k==ord('c'):
        cvs = np.zeros((h,w,3),dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
print('Done!')
