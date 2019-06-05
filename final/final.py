#This script defines a GUI interface and load the trained model and do prediction work
# import system module
import logging
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
import numpy as np
import time
import sys
import threading
# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget,QProgressBar
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets

IMG_SIZE = 28
input_size = 28
batch_size = 1
num_batch = 1
resultlist = ['stage1','stage2','stage3','stage4','stage5']
predict_sum = 0
counter = 0

#create UI interface
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        
        #set the overall window size
        Form.resize(1010, 900)
        
        #set the position and size of the video frame and start/stop button
        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setObjectName("image_label")
        self.image_label.move(60,370)
        self.image_label.resize(400,400)
        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.control_bt.resize(200,50)
        self.control_bt.move(160,800)
        
        #label for the progress bar
        project_label1 = QtWidgets.QLabel(Form)
        project_label1.setText("Total Progress")
        project_label1.move(440,20)
        
        #label for the video frame
        project_label2 = QtWidgets.QLabel(Form)
        project_label2.setText("Video Frame")
        project_label2.move(210,350)
        
        #label for the current stage
        project_label3 = QtWidgets.QLabel(Form)
        project_label3.setText("Your Current Stage")
        project_label3.move(670,350)
        
        #five pictures and corresponding checkbox or label
        self.pix1 = QPixmap('stage1.jpg')
        self.lb1 = QtWidgets.QLabel(Form)
        self.lb1.setGeometry(10,100,190,190)
        self.lb1.setStyleSheet("border: 2px solid red")
        self.lb1.setPixmap(self.pix1)
        self.lb1.setScaledContents(True)
        step_label = QtWidgets.QLabel(Form)
        step_label.setText("Step0:Empty Box")
        step_label.move(45,300)
        
        self.pix2 = QPixmap('stage2.jpg')
        self.lb12 = QtWidgets.QLabel(Form)
        self.lb12.setGeometry(210,100,190,190)
        self.lb12.setStyleSheet("border: 2px solid red")
        self.lb12.setPixmap(self.pix2)
        self.lb12.setScaledContents(True)
        self.step2_label = QtWidgets.QCheckBox("Step1:Install Part1",Form)  
        self.step2_label.move(220,300)
        
        self.pix3 = QPixmap('stage3.jpg')
        self.lb13 = QtWidgets.QLabel(Form)
        self.lb13.setGeometry(410,100,190,190)
        self.lb13.setStyleSheet("border: 2px solid red")
        self.lb13.setPixmap(self.pix3)
        self.lb13.setScaledContents(True)
        self.step3_label = QtWidgets.QCheckBox("Step2:Install Part2",Form)  
        self.step3_label.move(420,300)
        
        self.pix4 = QPixmap('stage4.jpg')
        self.lb14 = QtWidgets.QLabel(Form)
        self.lb14.setGeometry(610,100,190,190)
        self.lb14.setStyleSheet("border: 2px solid red")
        self.lb14.setPixmap(self.pix4)
        self.lb14.setScaledContents(True)
        self.step4_label = QtWidgets.QCheckBox("Step3:Install Part3",Form)  
        self.step4_label.move(620,300)
        
        self.pix5 = QPixmap('stage5.jpg')
        self.lb15 = QtWidgets.QLabel(Form)
        self.lb15.setGeometry(810,100,190,190)
        self.lb15.setStyleSheet("border: 2px solid red")
        self.lb15.setPixmap(self.pix5)
        self.lb15.setScaledContents(True)
        self.step5_label = QtWidgets.QCheckBox("Step4:Install Part4",Form)  
        self.step5_label.move(820,300)

        #set the current stage position and size
        self.lb = QtWidgets.QLabel(Form)
        self.lb.setGeometry(550,370,400,400)
        self.lb.setPixmap(self.pix1)
        self.lb.setScaledContents(True)
        
        #define the progress bar
        self.pbar = QProgressBar(Form)
        self.pbar.setGeometry(60, 40, 900, 30)
        self.pbar.setValue(0)
        self.retranslateUi(Form)
   
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        self.image_label.setText(_translate("Form", " "))
        self.control_bt.setText(_translate("Form", "Start"))
        


class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

    # view camera
    def viewCam(self):
        global counter
        global predict_sum
        # read image in BGR format
        ret, image = self.cap.read()
        frame = image
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        self.ui.image_label.setScaledContents(True)
        
        try:
            training_data = []
            for i in range(1):
                cv2.waitKey(20)
                new_array = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, 0])
            X = []
            y = []

            for features, label in training_data:
                X.append(features)
                y.append(label)
            X_train = np.transpose(X, (0, 3, 1, 2))
            
            X_train_set_as_float = X_train.astype('float32')
            X_train_set_norm = X_train_set_as_float[:] / 255.0
            
            data = X_train_set_norm
            label = np.array(y)
        
            input_size = 28
            batch_size = 1
            num_batch = 1
            eval_iter = mx.io.NDArrayIter(data, label, batch_size)
            
            # load model
            sym, arg_params, aux_params = mx.model.load_checkpoint("mynet", 10) # load with net name and epoch num
            mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=["data"], label_names=label) # label can be empty
            mod.bind(for_training=False, data_shapes=[("data", (batch_size,3,input_size, input_size))]) # data shape, 1 x 2 vector for one test data record
            mod.set_params(arg_params, aux_params, allow_missing=True)
            
            # predict
            predict_stress = mod.predict(eval_iter, num_batch)
#            kk = np.argmax(predict_stress, axis = 1)
            kk = int(np.argmax(predict_stress, axis = 1).asnumpy())
            
            #make a filter to improve the performance, the filter analyze results of each ten predictions
            if (counter < 10):
                counter = counter + 1 
                predict_sum = predict_sum + kk
            else:
                if(predict_sum < 3):
                    self.ui.pbar.setValue(100)
                    self.ui.step5_label.setChecked(True)
                    self.ui.step4_label.setChecked(True)
                    self.ui.step3_label.setChecked(True)
                    self.ui.step2_label.setChecked(True)
                    self.ui.lb.setPixmap(self.ui.pix5)
                    self.ui.lb.setScaledContents(True)

                if(predict_sum >= 7 and predict_sum < 13):
                    self.ui.pbar.setValue(75)
                    self.ui.step5_label.setChecked(False)
                    self.ui.step4_label.setChecked(True)
                    self.ui.step3_label.setChecked(True)
                    self.ui.step2_label.setChecked(True)
                    self.ui.lb.setPixmap(self.ui.pix4)
                    self.ui.lb.setScaledContents(True)
                 
                if(predict_sum >= 15 and predict_sum < 23):
                    self.ui.pbar.setValue(50)
                    self.ui.step5_label.setChecked(False)
                    self.ui.step4_label.setChecked(False)
                    self.ui.step3_label.setChecked(True)
                    self.ui.step2_label.setChecked(True)
                    self.ui.lb.setPixmap(self.ui.pix3)
                    self.ui.lb.setScaledContents(True)
               
                if(predict_sum >= 23 and predict_sum < 33):
                    self.ui.pbar.setValue(25)
                    self.ui.step5_label.setChecked(False)
                    self.ui.step4_label.setChecked(False)
                    self.ui.step3_label.setChecked(False)
                    self.ui.step2_label.setChecked(True) 
                    self.ui.lb.setPixmap(self.ui.pix2)
                    self.ui.lb.setScaledContents(True)
                   
                if(predict_sum >= 37):
                    self.ui.pbar.setValue(0)
                    self.ui.step5_label.setChecked(False)
                    self.ui.step4_label.setChecked(False)
                    self.ui.step3_label.setChecked(False)
                    self.ui.step2_label.setChecked(False)
                    self.ui.lb.setPixmap(self.ui.pix1)
                    self.ui.lb.setScaledContents(True)                  

                counter = 0
                predict_sum = 0
#            print (kk)
#            print (resultlist[int(np.argmax(predict_stress, axis = 1).asnumpy())]) # you can transfer to numpy array
            i = i+1
            
            #this time should be modified if there is delay
#            time.sleep(0.1)    #webcam
            time.sleep(0.02)   #phone cam
            

        except Exception as e:
            pass
        
    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
#            self.cap = cv2.VideoCapture(0)   #webcam
            self.cap = cv2.VideoCapture('http://192.168.188.33:8080/video')   #10FPS 640*480 small delay  phone cam
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")
            
# create a thread to protect the program shut down
def guiThread():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
    
# main function    
if __name__ == '__main__':
    added_thread = threading.Thread(target = guiThread)
    added_thread.start()
