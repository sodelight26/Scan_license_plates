import cv2 
import numpy as np 
import pytesseract 

from connect import mydb

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from datetime import date, datetime
from tracker import * 

        
#เรียกใช้ฟังก์ชั่น Tracke
Tracker = EuclideanDistTracker() 
cap = cv2.VideoCapture(0)
current_time = datetime.now().time()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
#การสร้างหน้าต่างสำหรับการปรับค่า Traeshold         
def empty(a):
        pass

cv2.namedWindow("Parameter") 
cv2.resizeWindow("Parameter", 640, 240) 
cv2.createTrackbar("Threshold1","Parameter",150,255,empty) 
cv2.createTrackbar("Threshold2","Parameter",255,255,empty) 

#สร้างตัวเก็บนับจำนวน
totalcount = []

while True: 
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bilateralFilter(gray, 10, 10, 20) 
        threshold1 = cv2.getTrackbarPos("Threshold1","Parameter") 
        threshold2 = cv2.getTrackbarPos("Threshold2","Parameter") 
        edged = cv2.Canny(gray, threshold1, threshold2) 

        imgcontour = frame.copy()
        
        contours, hierarchy  = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        detection = [] 
        for cnt in contours: 
                area = cv2.contourArea(cnt) 
                if area> 1200: 
                        cv2.drawContours(imgcontour, [cnt], -1,(255,0,255), 3) 

                        peri = cv2.arcLength(cnt, True) 
                        appox = cv2.approxPolyDP(cnt, 0.02*peri, True)                     
                        
                        if len(appox) == 4: 
                                screenCnt = appox 
                                x, y, w, h, = cv2.boundingRect(appox)
                                cv2.rectangle(imgcontour,(x,y), (x + w, y + h), (0,255,0), 5) 
                                detection.append([x, y, w, h]) 

                                # Masking คือการ mask ส่วนที่เราต้องการ 
                                mask = np.zeros(gray.shape,np.uint8) 
                                cv2.drawContours(mask,[screenCnt],0,255,-1,) 
                                cv2.bitwise_and(frame,frame,mask=mask) 
                                
                                # Now crop ตัดภาพ
                                (x, y) = np.where(mask == 255)  
                                (topx, topy) = (np.min(x), np.min(y))  
                                (bottomx, bottomy) = (np.max(x), np.max(y)) 
                                Cropped = gray[topx:bottomx+1, topy:bottomy+1] 
                                cv2.imshow("Crop", Cropped) 
                                text = pytesseract.image_to_string(Cropped, config='--psm 7 -c tessedit_char_whitelist=0123456789' , lang='tha+eng') 
                                #อ่านภาพบนตัวแปร Cropped และเก็บภาพลงในตัวแปร text 
                                
        #line กำหนดพิกัดเส้นsensor
        limit = [145, 0, 145, 480]
                            
        #box tracking สร้างตัวแปร box_id ไว้เก็บไอดีของวัตถุที่ตรวจจับ
        box_ids = Tracker.update(detection)

       
        cv2.line(imgcontour, (limit[0], limit[1]), (limit[2], limit[3]), (0,0,255), 3)
        for box_id in box_ids: 
                x, y, w, h, id = box_id 
                cv2.putText(imgcontour, str(id), (x,y - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)          
                cv2.putText(imgcontour,"Number plate is :" + str(text),(x,y - 50),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2) #แสดงเลขป้ายทะเบียนที่อ่านได้
                print(text) 

                cx,cy = x+w//2, y+h//2 #กำหนดจุกึ่งกลางของป้ายทะเบียน
                cv2.circle(imgcontour,(cx, cy), 5,(255,0,255),cv2.FILLED) 

                if limit[0] -20 <cx< limit[0] +20 and limit[1] <cy< limit[3]: 
                        if totalcount.count(id) == 0: 
                                totalcount.append(id) 
                                if text: 
                                        mycursor = mydb.cursor() 
                                        current_date = date.today() 
                                        current_time = datetime.now().time() 
                                        sql = "INSERT INTO car_records (license_plate_image, license_plate_text, date, time, total_cars) VALUES (%s, %s, %s, %s, %s)" #คำสั่ง SQL ที่ใช้ในการเพิ่มข้อมูล
                                        
                                        ret, buffer = cv2.imencode('.jpg', frame[y:y+h, x:x+w]) 
                                        img_str = buffer.tobytes()
                                        val = (img_str, text, current_date, current_time, len(totalcount)) 
                                        mycursor.execute(sql, val) 
                                        mydb.commit() 
                                        print("Record inserted successfully.")
        
        cv2.putText(imgcontour, f'CarCount :  {len(totalcount)}', (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)
        
        
        cv2.imshow("cam1", imgcontour) 
        cv2.imshow("cam2", edged)
        key = cv2.waitKey(1) 
        if key == 27: 
                break



cap.release() #คืนค่า ram 
cv2.destroyAllWindows() #ปิด Window ทุกอัน