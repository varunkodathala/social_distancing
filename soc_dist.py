import cv2
import torch
import torchvision
import torchvision.transforms as T 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


video_path = '/Users/varun/Documents/Deep_Learning/Learn_PYTORCH/ped_detect/video_org.mp4'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

(model.eval())

labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(image, threshold):

      transform = T.Compose([T.ToTensor()]) 
    
      img = transform(image) 
        
      pred = model([img])

      pred_class = [labels[i] for i in list(pred[0]['labels'].numpy())] 

      bb = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
     
      prob = list(pred[0]['scores'].detach().numpy())

      prob_t = [prob.index(x) for x in prob if x > threshold][-1]

      bb = bb[:prob_t+1]

      pred_class = pred_class[:prob_t+1]

      return bb, pred_class


def detect_person(image,response_map,threshold):
    cx = []
    cy = []
    response_map = np.zeros((900,1280,3), np.uint8)
    boxes, pred_cls = get_prediction(image,0.5)
    for i in range(len(boxes)):
        if(pred_cls[i]=='person'):
            cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=2)
            x = (int(np.mean(boxes[i][0])))
            y = (int(np.mean(boxes[i][1])))
            cx.append(x)
            cy.append(y)
            response_map[x-5:x+5,y-5:y+5] = (0,255,0)

    response_map = dist_measure(cx,cy,response_map)
    
    


    return image,response_map


def dist_measure(cx,cy,response_map):
    red_map =[]
    yellow_map = []
    yellow_count = 0
    red_count = 0
    all_count = 0
    all_map = []
    response_map = response_map

    for i in range(len(cx)):
        dist = []
        si = []
        di = []
        ndi = []
        for j in range(len(cx)):
            dist.append(((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)**0.5)
        status,d_c,nd_c = check_dist(dist)

        for k in range(len(status)):

            if(status[k]=='danger'):
                di.append(k)
            if(status[k] == 'near_danger'):
                ndi.append(k)
            if(status[k] == 'safe'):
                si.append(k)
    
        for m in range(len(ndi)):
            response_map[cx[i]-5:cx[i]+5,cy[i]-5:cy[i]+5] = (0,255,255)
            response_map[cx[ndi[m]]-5:cx[ndi[m]]+5,cy[ndi[m]]-5:cy[ndi[m]]+5] = (0,255,255)
            

        for m in range(len(di)):
            response_map[cx[i]-5:cx[i]+5,cy[i]-5:cy[i]+5] = (0,0,255)
            response_map[cx[di[m]]-5:cx[di[m]]+5,cy[di[m]]-5:cy[di[m]]+5] = (0,0,255) 
 


    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(response_map,f"Response MAP",(360,150), font, 1,(255,255,255),6,cv2.LINE_AA)

    cv2.putText(response_map,f"Total Count: {len(cx)}",(100,850), font, 1,(255,0,0),6,cv2.LINE_AA)
        
    return response_map

        

def check_dist(dist):

    status = []
    d_c = 0
    nd_c = 0

    for i in range(len(dist)):

        if(dist[i]>0 and dist[i]<28):

            status.append('danger')
            d_c+=1
        
        elif(dist[i]>35 and dist[i]<56):

            status.append('near_danger')
            nd_c +=1
        
        if(dist[i]>=85):

            status.append('safe')
    
    return status,d_c,nd_c


response_map = np.zeros((900,1280,3), np.uint8)



cam = cv2.VideoCapture(video_path)
count = 0

while(True):
    ret,image = cam.read()
    imag,response_map = detect_person(image,response_map,0.5)
    count += 50
    cam.set(1, count)
    try:
        cv2.imshow('video',cv2.resize(imag,(512,512)))
        roi_img = response_map[100:,100:800,:]
        response = cv2.resize(roi_img,(255,255))
        cv2.imshow('response',response)
    except Exception as e:
        break    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()