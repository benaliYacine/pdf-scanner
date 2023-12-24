import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from openpyxl import load_workbook
import xlwings as xw

from difflib import SequenceMatcher

#section methods
def detect_black_bars_and_extract_roi(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel_length = 150 # You can adjust this value
    min_height = 20
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    horizontal_img = cv2.erode(thresh, horizontal_kernel, iterations=3)
    horizontal_img = cv2.dilate(horizontal_img, horizontal_kernel, iterations=3)
    contours, _ = cv2.findContours(horizontal_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > kernel_length and h > min_height:
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 0, 255), 2)
            rois.append(((x, y, x+w, y+h), img_np[y:y+h, x:x+w]))
    return img_np, rois

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def recognize_text_and_highlight_bars(img_np, rois, target_text):
    similarity_threshold = 0.7
    for (x1, y1, x2, y2), roi in rois:
        recognized_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        print(recognized_text)
        if similar(recognized_text.upper(), target_text) > similarity_threshold:
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_np

def detect_and_highlight_rectangle(img_np, rois, target_text):
    similarity_threshold = 0.7
    green_bar_coords = None
    for (x1, y1, x2, y2), roi in rois:
        recognized_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        if similar(recognized_text.upper(), target_text) > similarity_threshold:
            green_bar_coords = (x1, y2, x2, y2)  # Using the bottom y-coordinate of the green bar
            break

    if green_bar_coords:
        next_bar_y = img_np.shape[0]  # Set to the bottom of the image as an initial value
        for (x1, y1, x2, y2), _ in rois:
            if y1 > green_bar_coords[1] and y1 < next_bar_y:
                next_bar_y = y1
        section_coords= (green_bar_coords[0], green_bar_coords[1],green_bar_coords[2], next_bar_y)
        
        cv2.rectangle(img_np,  (green_bar_coords[0], green_bar_coords[1]),(green_bar_coords[2], next_bar_y), (255, 0, 0), 2)
    return img_np,section_coords

def extract_needed_section(pdf_path,page):
    img_np = pdf_to_image(pdf_path,page)
    detected_img, rois = detect_black_bars_and_extract_roi(img_np)
    highlighted_img = recognize_text_and_highlight_bars(detected_img, rois, "FALL RISK ASSESSMENT")
    final_img,section_coords = detect_and_highlight_rectangle(highlighted_img, rois, "FALL RISK ASSESSMENT")
    Image.fromarray(final_img).show()
    # If no section was detected, return.
    if not section_coords:
        print("Target section not detected.")
        return
    
    # Step 2: Crop the image to the detected section.
    x1, y1, x2, y2 = section_coords
    cropped_image = final_img[y1:y2, x1:x2]
    Image.fromarray(cropped_image).show()
    return cropped_image


#check boxe methods
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
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r=4
        if word in line:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)-r, int(x2)+r, int(y2)+r
            bounding_boxes.append((x2+12, y1, x2+length, y2))

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

if __name__ == "__main__":

    pdf_path = "sections\FOR UPWORK #1.pdf"
    page=18
    needed_section_image=extract_needed_section(pdf_path,page)  


    thresh_OCR_ratio=120
    img_for_OCR= preprocess_image(needed_section_image,thresh_OCR_ratio)
    Image.fromarray(img_for_OCR).show()
    total_ROI=detect_word_location(img_for_OCR,'TOTAL',80)
    x1,y1,x2,y2=total_ROI[0]
    cv2.rectangle(needed_section_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    total_text=extract_text_from_roi(img_for_OCR,total_ROI[0])
    print('TOTAL:',total_text)
    excel_path = (r'guide\WORKING.xlsm')
    excel_inputs={}

    if int (total_text) not in [0,1,2,3,4,5,6,7,8,9,10]:

        #***********************************values that can be tweaked************************************************
        #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
        thresh_check_ratio=180
        thresh_radio_ratio=120
        thresh_OCR_ratio=200
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_check_ratio=0.6


        image_org = needed_section_image
        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        thresh_radio = preprocess_image(image_org,thresh_radio_ratio)
        output_image, squares = detect_checkboxes(thresh_check,image_org)
        filled_check_buttons,output_image = detect_filled_button(thresh_radio,squares,output_image,filled_check_ratio)
        img_for_OCR= preprocess_image(image_org,thresh_OCR_ratio)
        img_for_OCR2= preprocess_image(image_org,140)
        if filled_check_buttons:
            filled_count = len(filled_check_buttons)
            print(filled_count)
            total=filled_count


        else:
            print("No filled button detected.")
            total=0
        Image.fromarray(output_image).show()
    else:
        total=int (total_text)


    print(f"FALL RISKS TOTAL: {total}")
    excel_inputs['F65']=total
    #******************updating the excel file*************************************
    update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py