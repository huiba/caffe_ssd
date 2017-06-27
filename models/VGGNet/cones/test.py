# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import _init_paths
import matplotlib.pyplot as plt
import numpy as np
import datetime
import caffe
import os
import cv2

# parameters
#dataFolderTest = '/home/andras/data/datasets/kar/measured/angle'
#dataFolderTest = '/home/andras/data/datasets/kar/cones_08.04'
dataFolderTest = '/home/ubuntu/data/cones/val/images'

modelDef = '/home/ubuntu/efs/ssd/caffe/models/VGGNet/cones/SSD_300x300/deploy.prototxt'
#modelDef = 'deploy.prototxt'
#modelParams = '04.04_augm/snapshot_iter_9000.caffemodel'
modelParams = '/home/ubuntu/efs/ssd/caffe/models/VGGNet/cones/SSD_300x300/VGG_cones_SSD_300x300_iter_300.caffemodel'
#threshold = 2.2

resX = 300
#resX = 1280
resY = 300
#resY = 804


import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def write_bboxes(im, imagename, bboxArray, scoreArray, classArray):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(bboxArray)):
        bbox = bboxArray[i]
        score = scoreArray[i]
        class_name = classArray[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(str(class_name), score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #ax.set_title(('Detections with '
    #              'p(obj | box) >= {:.1f}').format(logo_threshold),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('/home/andras/resultimages/' + imagename)
    plt.close()

def draw_boxes(im, imagename, bboxArray):
    for bbox in bboxArray:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite('/home/andras/data/datasets/kar/videos/vid_second_out/' + imagename.split('.')[0] + '.jpg', im)



caffe.set_mode_gpu()
caffe.set_device(0)


### step 1: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
net = caffe.Net(modelDef, modelParams, caffe.TEST)

### step 2: test net with dynamically created data
print('{:s} - Testing'.format(str(datetime.datetime.now()).split('.')[0]))

dummy = np.array([[[[1]]]]).astype(np.float32)

out = ''
for subdir, dirs, files in os.walk(dataFolderTest):
    for f in files:
        filename = os.path.join(subdir, f)
        print filename
        testimg = cv2.imread(filename)

        downscaled = testimg[550: testimg.shape[0], :]
        height, width = downscaled.shape[:2]
        xScale = float(width) / float(resX)
        yScale = float(height) / float(resY)
        downscaled = cv2.resize(downscaled, (resX, resY), interpolation = cv2.INTER_CUBIC)
        downscaled = cv2.cvtColor(downscaled, cv2.COLOR_BGR2RGB)
        downscaled = downscaled - np.array([104, 117, 123])
        downscaled = downscaled.swapaxes(0,2).swapaxes(1,2)
        img = np.array([downscaled]).astype(np.float32)
        net.blobs['data'].reshape(*img.shape)
        net.blobs['data'].data[...] = img
        #net.set_input_arrays(img, dummy)
        net.forward()

        # accuracy test for verification
        bb = net.blobs['detection_out'].data
        print bb
        system.exit(0)
        #print bb
        #indexes = np.where(net.blobs['coverage'].data > 0)
        # FOR MEASURED CONES: & (bb[0, :, 3] > 160)
        mask = np.any(bb > 0, axis=2) & (((bb[0, :, 3]-bb[0, :, 1]) < 140)) #& (bb[0, :, 1] > 300) & (bb[0, :, 4] > threshold)
        #&((bb[0, :, 0] < 160) | (bb[0, :, 3] < 350) | ((bb[0, :, 0] > 870) & (bb[0, :, 3] < 450))) & (bb[0, :, 3] > 220) & (bb[0, :, 1] > 170)
        
        bb = bb[mask]
        mask = py_cpu_nms(bb, 0.7)
        #mask = 
        boxes = bb[mask, :4]
        dir = subdir.split('/')[-1]
        for box in boxes:
            #out += dir + '/' + f + ',' + str(int(box[0])) + ',' + str(int(box[1])) + ',' + str(int(box[2])) + ',' + str(int(box[3])) + '\n'
            box[0] = int(round((float(box[0]) * xScale)))
            box[1] = int(round((float(box[1]) * yScale))) + 550
            box[2] = int(round((float(box[2]) * xScale)))
            box[3] = int(round((float(box[3]) * yScale))) + 550

        scores = bb[mask, 4]
        #testimg = testimg.swapaxes(1,2).swapaxes(0,2)
        #testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
        #text = ["cone" for i in range(len(scores))]
        #write_bboxes(testimg, f, boxes, scores, text)
        draw_boxes(testimg, f, boxes)
        
#with open('/home/andras/measuredconebboxes.txt', 'w') as f:
#    f.write(out)
