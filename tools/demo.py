import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import time
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

import shutil

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):
    # Create txt folder
    txt_result_path = os.path.join(opt.save_dir,'text')
    if os.path.exists(txt_result_path):
      shutil.rmtree(txt_result_path)  # delete dir
    os.mkdir(txt_result_path)

# =========================================
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16


    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    # Set Dataloader
    source = opt.source
    list_folder = os.listdir(source)

    # run every single folder
    for folder in list_folder:
      print('===== ', folder)
      opt.source = os.path.join(source, folder)
      name_folder_run = str(opt.source).split('/')[-1]
      
      # Get speech from txt 
      ls = os.listdir(opt.source)
      for i in ls:
        if '.txt' in i:
          file_speed = open(os.path.join(opt.source, i), 'r')
          for item in file_speed:
            data =  item.split(',')
            idx = data[0]
            speed = data[-1]
            dict_speed.update({idx: speed})
      

      split = str(opt.save_dir).split('/')
      if len(split) > 2: # inference/output/trafficjam_1602_12/trafficjam_1602_32/trafficjam_1602_20/trafficjam_1602_23
        opt.save_dir = os.path.join(split[0], split[1])
    
      opt.save_dir = os.path.join(opt.save_dir, name_folder_run)
      os.makedirs(opt.save_dir)  # make new dir

      txt = open(os.path.join(opt.save_dir, name_folder_run + '.txt'), 'w')
      txt2 = open(os.path.join(txt_result_path,name_folder_run + '.txt'), 'w')

      if opt.source.isnumeric():
          cudnn.benchmark = True  # set True to speed up constant image size inference
          dataset = LoadStreams(opt.source, img_size=opt.img_size)
          bs = len(dataset)  # batch_size
      else:
          dataset = LoadImages(opt.source, img_size=opt.img_size)
          bs = 1  # batch_size


      # Get names and colors
      names = model.module.names if hasattr(model, 'module') else model.names
      colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


      # Run inference
      t0 = time.time()


      vid_path, vid_writer = None, None
      img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
      _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
      model.eval()

      inf_time = AverageMeter()
      nms_time = AverageMeter()
      
      for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
          start = time.time()
          img = transform(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          if img.ndimension() == 3:
              img = img.unsqueeze(0)
          # Inference
          t1 = time_synchronized()
          det_out, da_seg_out,ll_seg_out= model(img)
          t2 = time_synchronized()

          inf_out, _ = det_out
          inf_time.update(t2-t1,img.size(0))

          # Apply NMS
          t3 = time_synchronized()
          det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
          t4 = time_synchronized()

          nms_time.update(t4-t3,img.size(0))
          det=det_pred[0]

          save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")


          _, _, height, width = img.shape
          h,w,_=img_det.shape
          pad_w, pad_h = shapes[1][1]
          pad_w = int(pad_w)
          pad_h = int(pad_h)
          ratio = shapes[1][0][1]

          da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
          da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
          _, da_seg_mask = torch.max(da_seg_mask, 1)
          da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
          # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

          
          ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
          ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
          _, ll_seg_mask = torch.max(ll_seg_mask, 1)
          ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
          # Lane line post-processing
          # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
          #ll_seg_mask = connect_lane(ll_seg_mask)
          img_det, segment_img = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
          # img_det = show_seg_result(img_det,  ll_seg_mask, _, _, is_demo=True)


          if len(det):
              det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
              for *xyxy,conf,cls in reversed(det):
                  label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                  plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
          
          if dataset.mode == 'images':
              infer_time = time.time()-start
              cv2.imwrite(save_path,img_det)
              # cv2.imwrite(save_path,vanishing(ll_seg_mask,img_det))
              # countour
              contour, area = run_contours(segment_img, infer_time)
              path = save_path.replace('.jpg', '_seg.jpg')
              cv2.imwrite(path,contour)
              speed = dict_speed[get_nameImg(path)].replace('\n','')
              txt.writelines(f'{path}|{area}|{speed}|\n')
              txt2.writelines(f'{path}|{area}|{speed}|\n')

          elif dataset.mode == 'video':
              if vid_path != save_path:  # new video
                  vid_path = save_path
                  if isinstance(vid_writer, cv2.VideoWriter):
                      vid_writer.release()  # release previous video writer

                  fourcc = 'mp4v'  # output video codec
                  fps = vid_cap.get(cv2.CAP_PROP_FPS)
                  h,w,_=img_det.shape
                  vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
              vid_writer.write(img_det)
          
          else:
              cv2.imshow('image', img_det)
              cv2.waitKey(1)  # 1 millisecond

      print('Results saved to %s' % Path(opt.save_dir))
      print('Done. (%.3fs)' % (time.time() - t0))
      print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))

    
    # Run predict congestion
    congestion = predict_trafficCongestion(txt_result_path)


def predict_trafficCongestion(path_file):
    congestion = False
    files = os.listdir(path_file)
    for f in files:
        txt = pd.read_csv(os.path.join(path_file, files), header=None)
        name,dt,vt,_ = str(txt.values[0]).split('|')
        dts = []
        vts = []
        for line in txt.values:
            name,dt,vt,_ = str(line).split('|')
            dts.append(dt)
            vts.append(vt)

        l_ketxe = []
        for  i in range(1,4):
            dt = dts[-i]
            vt = vts[-i]
            if float(dt) < 100000 and float(vt) <20 :
                l_ketxe.append(-1)
            else:
                l_ketxe.append(1)

        if sum(l_ketxe) < 0:
            congestion = True
            print(f'{f} ket xe')
        else:
            print(f'{f} khong ket xe')

    return congestion


def get_nameImg(path):
    name = str(path).replace('_seg.jpg', '').split('/')[-1]
    return name


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def run_contours(img, time):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 20, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    draw = cv2.drawContours(img, contours, -1, (255,0,0), 2)

    bboxes = []
    center_box=[]
    total = 0
    count = 0
    for i, contour in enumerate(contours):
      M = cv2.moments(contour)
      area = cv2.contourArea(contour)
      total += area

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    image = cv2.putText(draw, str(total), org, font, 
                      fontScale, color, thickness, cv2.LINE_AA)
    
    return draw, total

def vanishing(ll_seg_mask, img_det ):
    ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
    ll_seg_mask = np.expand_dims(ll_seg_mask, axis=0)
    ll_seg_mask = np.transpose(ll_seg_mask, (1, 2, 0))
    ll_seg_mask = np.uint8(ll_seg_mask*255)

    map2 = np.zeros((2000,2500),dtype=np.uint8)
    map3 = np.zeros((img_det.shape[0],img_det.shape[1]),dtype=np.uint8)
    ll_seg_mask = cv2.erode(ll_seg_mask, (5,5), iterations=10)
    edges = cv2.Canny(ll_seg_mask,0,255)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0)
    if lines is not None: 
        for line in lines:
            rho,theta=line[0]
            theta1=theta*180/np.pi
            cv2.circle(map2,(int(rho)+1000,int(theta1*10)),15,[255,10,25],6)
        contours,_ = cv2.findContours(map2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        bboxes = []
        center_box=[]
        for cnt in contours:
            a = cv2.contourArea(cnt)
            #print(a)
            if a > 1200:
              count +=1
              x1,y1,w1,h1 = cv2.boundingRect(cnt)
              bboxes.append([x1,y1,w1,h1])
              cv2.rectangle(map2,(x1,y1),(x1+w1,y1+h1),(255,255,0),4)
              center_box.append((x1+w1/2,y1+h1/2))
        #print(center_box)
        line_final = []
        for center in center_box:
            rho = center[0]-1000
            theta = (center[1]/10)*np.pi/180
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            line_final.append((pt1,pt2))
            cv2.line(img_det, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        va_points = []
        for i in range(len(line_final)):
          for j in range(len(line_final)-i):
              if i != j+i:
                x_va,y_va = line_intersection(line_final[i],line_final[j+i])
                va_points.append((x_va,y_va))
                #print(x_va,y_va)
                cv2.circle(img_det,(int(x_va),int(y_va)),15,[255,10,25],6)
                cv2.circle(map3,(int(x_va),int(y_va)),15,[255,10,25],6)
        contours2,_ = cv2.findContours(map3,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boxes_va = []
        for cnt_va in contours2:
          x2,y2,w2,h2 = cv2.boundingRect(cnt_va)
          cv2.rectangle(map3,(x2,y2),(x2+w2,y2+h2),(255,255,0),4)
          boxes_va.append([x2,y2,w2,h2])
        max_point = 0

        candidate = None
        for box_va in boxes_va:
            num_va = 0
            for vapoint in va_points:
              if vapoint[0]>box_va[0] and vapoint[0]< (box_va[0] + box_va[2]) and vapoint[1]>box_va[1] and vapoint[1]<(box_va[1] + box_va[3]):

                  num_va +=1 
            if num_va > max_point and num_va > 1:
              max_point = num_va
              candidate = (box_va[0]+box_va[2]/2,box_va[1]+box_va[3]/2)
        if candidate is not None:
          print(candidate)
          cv2.circle(img_det,(int(candidate[0]),int(candidate[1])),15,[0,255,25],6)

    return img_det    


dict_speed = {}
txt=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
      
    with torch.no_grad():
        detect(cfg,opt)
