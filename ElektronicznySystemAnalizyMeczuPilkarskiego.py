from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadImages
from pathlib import Path
import torch 
import cv2
import numpy as np
import time as t
import yolov5
from strongsort.strong_sort import StrongSORT
video_path = 'mecze/Cam2/Cam2_Mecz1'
image_size = 640
config = 'osnet_x0_25_market1501.pt'
model = 'yolov5s.pt'
confidence = 0.5
device = 'cpu'
strongsort_list=[]
augment = False
view_image = True
model = yolov5.load(model, device=device)
model.conf = confidence

dataset = LoadImages(Path(video_path), image_size)
strongsort_list.append(StrongSORT(model_weights=Path("osnet_x0_25_market1501.pt"), device= 'cpu', fp16=False))
outs = [None]
last_id = {}
all_distance = {}
distance_const = 154
help1 = 1
help_id = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Test.avi', fourcc, 20.0, (1920,1080))
for path, im, im0s, vid_cap, s in dataset:

    with open('id.txt', 'a') as i:
        i.write(f"\nKlatka: {help1} ")
    print("Klatka: " + str(help1))
    help1 += 1
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /=255
    if len(im.shape) ==3:
        im=im[None]
    prediction = model(im, size = image_size, augment = augment)
    prediction = non_max_suppression(prediction, conf_thres=model.conf, iou_thres=model.iou, classes=model.classes, agnostic=model.agnostic)

    for i, det in enumerate(prediction):

        annotator = Annotator(im0s, line_width=2, example=str(model.names))

        if len(det):
            det[:,:4]= scale_boxes(im.shape[2:], det[:,:4], im0s.shape).round()
            xywh=xyxy2xywh(det[:,0:4]).cpu().detach().numpy()
            conf = det[:,4].cpu().detach().numpy()
            clas = det[:, 5].cpu().detach().numpy()
            outs[i] = strongsort_list[i].update(det,im0s)
            if len(outs[i]) > 0:

                    for j, (outs1, conf1) in enumerate(zip(outs[i], conf)):
                        boxes = outs1[0:4]
                        id = outs1[4]
                        clas = outs1[5]
                        if view_image:
                            clas2 = int(clas)
                            id = str(id)
                            label = label = "%s %.2f" % (model.names[int(clas)], conf1)
                            annotator.box_label(boxes, id, color=colors(clas2, True))
                            help_id.append(id)

                        if id not in last_id.keys():
                            last_id[id] = (boxes[0], boxes[1])
                            all_distance[id] = 0.0
                        else:
                            distance = np.sqrt(np.power((boxes[0] - last_id[id][0]), 2) + np.power((boxes[1]- last_id[id][1]), 2)) / distance_const
                            if distance >0.01:
                                last_id[id] = (boxes[0], boxes[1])
                                all_distance[id] += distance
                                print("ID: " + str(id) + " .Dystans: " + str(all_distance), "m.")
                                with open('distance.txt', 'w') as d:
                                    pass
                                with open('distance.txt', 'w') as f:
                                    for id in last_id:
                                        f.write(f"ID: {id}: Dystans: {all_distance[id]:.2f} m.\n")
                                        
                                
    im0 = annotator.result()
    if view_image:
        im0 = cv2.resize(im0, (1920,1080), interpolation=cv2.INTER_AREA)
        cv2.imshow(str(path), im0)
        cv2.waitKey(1)
        out.write(im0)
    with open('id.txt', 'a') as i:
        for id in help_id:
            i.write(f"ID: {id} ")
            print("ID: " + str(id) + " ")
    help_id = []
