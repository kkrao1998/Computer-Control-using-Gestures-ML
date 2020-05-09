import sys
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import copy
import cv2 
import numpy as np
import pyautogui

cap=cv2.VideoCapture(0)

def removeBG(hsv):
    fgmask=bgM.apply(hsv,learningRate=0)
    kern=np.ones((9,9),np.uint8)
    fgmask = cv2.erode(fgmask, kern, iterations=1)
    res = cv2.bitwise_and(hsv, hsv, mask=fgmask)
    return res

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    resul = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            resul = human_string
    return resul, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0
    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 0
    consecutive = 0
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            x1, y1, x2, y2 = 400, 200, 800, 400
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            if i == 17:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                print(res,score)
                if res == 'U'or res == 'u':
                    print('up')
                    pyautogui.click()
                    pyautogui.scroll(-20)
                elif res == 'N' or res == 'n':
                    print('down')
                    pyautogui.click()
                    pyautogui.scroll(20)
                elif res == 'Y' or res == 'y':
                    print('Zoom out')
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('-')
                    pyautogui.keyUp('ctrl')
                elif res == 'A' or res == 'a' or res == 'S' or res =='s':
                    print('Zoom in')
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('+')
                    pyautogui.keyUp('ctrl')
                elif res == 'G' or res == 'g':
                    print('Left')
                    pyautogui.press('left')
                elif res == 'H' or res == 'h':
                    print('Right')
                    pyautogui.press('right')
                elif res == 'B' or res == 'b':
                    print('click')
                    pyautogui.click()
                elif res == 'V' or res == 'v' or res == 'K' or res == 'k':
                    print('Double Click')
                    pyautogui.doubleClick()
                else:
                    # lower_blue = np.array([100,100,85])  # for blue range value
                    # upper_blue = np.array([210,255,255])
                    # color range for cursor 
                    t_end = time.time() +  15 #for 15 second s
                    while time.time() < t_end:
                        _, frame = cap.read()
                        frame=cv2.flip(frame,1)
                        cv2.rectangle(frame,(400,400),(800,100),(0,255,0),0)
                        frame=frame[200:400,400:800]
                        panfree=np.copy(frame)
                        #Bgr to Hsv
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        #cv2.imshow('main frame', frame)


                        bgM = cv2.createBackgroundSubtractorMOG2(0,82)


                        img=removeBG(hsv)
                        #cv2.imshow('fg_mask', img)

                        #color range for cursor
                        """lower_blue = np.array([100,90,100]) 
                        upper_blue = np.array([245,255,255])"""
                        #color range for cursor 

                        

                        lower_blue = np.array([100,100,85]) 
                        upper_blue = np.array([210,255,255])    
                        #track blue cursor   
                        mask = cv2.inRange(img, lower_blue, upper_blue) 
                        resul = cv2.bitwise_and(frame,frame, mask= mask)
                        #cv2.imshow('mask',mask) 
                        #cv2.imshow('resul',resul)
                        _, thresh1 = cv2.threshold(mask,90, 255, cv2.THRESH_BINARY)
                        _, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                        max_area_blue=-1
                        area=0
                        for i in range(len(contours)):
                            cnt=contours[i]
                            area = cv2.contourArea(cnt)
                            if(area>max_area_blue):
                                #print("blue area",area)
                                max_area_blue=area
                                ci=i
                                cnt=contours[ci]
                        if area>5:
                            a,b,w,h = cv2.boundingRect(cnt)
                            axb,byb,w,h = cv2.boundingRect(cnt)
                            cv2.rectangle(frame,(a,b),(a+w,b+h),(0,0,255),2)
                            #print(a,b)
                            if a!=0 and b!=0:
                                ox=240
                                nx=1366
                                a=a*nx/ox
                                oy=198
                                ny=768
                                b=b*ny/oy
                                pyautogui.moveTo(a,b)



                    
                
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,255,0), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)
            if a == 27:             # when esc is pressed
                break
            # if a & 0xff== ord('k'): # when `k` is pressed
            #     break
input("Press Enter to Continue")
#time.sleep(5)
# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()