import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image ,ImageDraw
import pytesseract
from openpyxl import load_workbook
import xlwings as xw

def pdf_to_image(pdf_path,page):
    images = convert_from_path(pdf_path)
    return np.array(images[page-1])

def distance(box1, box2):
    # Compute the distance between the centers of two boxes
    return np.sqrt((box1[0] + box1[2]/2 - box2[0] - box2[2]/2)**2 + (box1[1] + box1[3]/2 - box2[1] - box2[3]/2)**2)

def preprocess_image(image_org,ratio):
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, ratio, 255, cv2.THRESH_BINARY)
    return thresh

def detect_checkboxes(thresh,img):

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 40 and w <= 50:  # Check if side length is at least 18 pixels
                bounding_boxes.append((x, y, w, h))

    # Filter boxes based on distance
    filtered_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        keep = True
        for other_box in bounding_boxes:
            if distance(box, other_box) < 20:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,255), 2)

    output = img.copy()
    return output, filtered_boxes

def extract_text_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
    #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
    return text

def update_excel_sheet(path,inputs):
    start_time = time.time()
    print ('start_time...')
    app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
    wb = app.books.open(path)
    ws = wb.sheets['Basic- Auto']
    for cell in inputs:
        ws.range(cell).value = inputs[cell]
    # Save and close
    wb.save()
    wb.close()
    app.quit()
    end_time = time.time()
    print ('end_time')
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.2f} seconds")
    print(f"Updated Excel sheet '{path}' with :")
    for cell in inputs:    
        print(f"value '{inputs[cell]}' in cell '{cell}'.")


if __name__ == "__main__":
    pdf_path = (r'guide\FOR_UPWORK_#1.pdf')
    excel_path = (r'guide\WORKING.xlsm')
    page=19
    #***********************************valus that can be tweaked************************************************
    #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
    thresh_check_ratio=180
    thresh_radio_ratio=120
    thresh_OCR_ratio=140

    image_org = pdf_to_image(pdf_path,page)
    thresh_check = preprocess_image(image_org,thresh_check_ratio)
    thresh_radio = preprocess_image(image_org,thresh_radio_ratio)
    output_image, boxes = detect_checkboxes(thresh_check,image_org)



    img_for_OCR= preprocess_image(image_org,thresh_OCR_ratio)
    Image.fromarray(img_for_OCR).show()

    x, y, w, h = boxes[2]
    x1,y1,x2,y2 =x, y, x+w, y+h



    cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    box=x1+2,y1+2,x2-2,y2-2
    urinary_text=extract_text_from_roi(img_for_OCR,box)
    

    excel_inputs={}

    print('urinary_text:',urinary_text)
    urinary_number=int (urinary_text)
    
    urinary_mapping = {
    0: "No Incontinance",
    1: "Incontinent",
    2: "Foley"
}

    urinary=urinary_mapping.get(urinary_number, "Invalid urinary_number")
    
    print(f"urinary: {urinary}")


    if urinary != "Invalid urinary_number":
        excel_inputs['F65']=urinary


    Image.fromarray(output_image).show()
    #******************updating the excel file*************************************
    update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py