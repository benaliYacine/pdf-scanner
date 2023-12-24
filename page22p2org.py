import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from openpyxl import load_workbook
import xlwings as xw

from difflib import SequenceMatcher

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
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28:  # Check if side length is at least 18 pixels
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

    # Draw the filtered boxes on the image
    squares = []
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,255), 2)
        
        center_x = x + w // 2
        center_y = y + h // 2
        diameter =( w // 2 )+3  # or h, since it's approximately a square
        squares.append((center_x, center_y, diameter))
    output = img.copy()
    return output, squares

def detect_filled_button(thresh, circles,image_org, filled_threshold_ratio):
    output = image_org.copy()
    filled_buttons = []

    for (x, y, r) in circles:
        roi = thresh[y-r:y+r, x-r:x+r]
        black_pixel_count = np.sum(roi == 0)
        total_pixel_count = np.pi * r * r
        if black_pixel_count / total_pixel_count > filled_threshold_ratio:
            cv2.circle(output, (x, y), r-2, (255, 0, 0), 2)
            filled_buttons.append((x, y, r))

    return filled_buttons,output

def detect_word_location(img,word,length):
    # Step 2: Use OCR to detect the word "Address" and get its bounding boxes
    length=(length-128)//3
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r=4
        if word in line:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)-r, int(x2)+r, int(y2)+r

            bounding_boxes.append((x2+128+5, y1-20, x2+length+128-5, y2+7))
            bounding_boxes.append((x2+length+128+5, y1-20, x2+length*2+128-5, y2+7))
            bounding_boxes.append((x2+length*2+128+5, y1-20, x2+length*3+128-5, y2+7))

    return bounding_boxes

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

def join_strings(string_list):
    if not string_list:
        return ""
    elif len(string_list) == 1:
        return string_list[0]
    elif len(string_list) == 2:
        return " and ".join(string_list)
    return ", ".join(string_list[:-1]) + ", and " + string_list[-1]



#page 22 p2 methods
def format_site_texts(site1_text, site2_text, site3_text):
    # Function to handle special replacements
    def handle_special_replacements(text):
        # if text == "BLE’s" or text == "BLE":
        if text == "BLE’s":
            return "HIP AREA"
        return text

    # Apply special replacements and filter out the empty sites
    sites = [handle_special_replacements(site).upper() for site in [site1_text, site2_text, site3_text] if site]

    # Use the join_strings function to format the text
    formatted_text = join_strings(sites)

    return formatted_text

if __name__ == "__main__":

    pdf_path = ('FOR_UPWORK_#1.pdf')
    excel_path = ('WORKING.xlsm')
    page=22
    image_org = pdf_to_image(pdf_path,page)

    thresh_OCR_ratio=150
    img_for_OCR= preprocess_image(image_org,thresh_OCR_ratio)
    Image.fromarray(img_for_OCR).show()
    Location_ROI=detect_word_location(img_for_OCR,'Location',670)
    x1,y1,x2,y2=Location_ROI[0]
    cv2.rectangle(image_org, (x1, y1), (x2, y2), (255, 0, 0), 2)
    site1_text=extract_text_from_roi(img_for_OCR,Location_ROI[0])
    print('site1:',site1_text)

    x1,y1,x2,y2=Location_ROI[1]
    cv2.rectangle(image_org, (x1, y1), (x2, y2), (0, 255, 0), 2)
    site2_text=extract_text_from_roi(img_for_OCR,Location_ROI[1])
    print('site2:',site2_text)

    x1,y1,x2,y2=Location_ROI[2]
    cv2.rectangle(image_org, (x1, y1), (x2, y2), (0, 0, 255), 2)
    site3_text=extract_text_from_roi(img_for_OCR,Location_ROI[2])
    print('site3:',site3_text)


    PAIN_ASSESSMENT = format_site_texts(site1_text, site2_text, site3_text)

    # Update the Excel sheet
    excel_inputs = {}
    print(f"PAIN ASSESSMENT: {PAIN_ASSESSMENT}")
    excel_inputs['H20'] = PAIN_ASSESSMENT


    Image.fromarray(image_org).show()

    #******************updating the excel file*************************************
    update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py