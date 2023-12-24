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

def extract_text_roi(image_org, filled_buttons,x_ratio,y_ratio):
    output = image_org.copy()
    rois_coordinates=[]
    for x, y, r in filled_buttons:#ani dayerha twila bezaf lakhaterch ocr ye9ra ri lkelma lewla li le9raha tema
        roi_x_start = x + r+50
        roi_x_end = x + x_ratio * r  # This can be adjusted based on expected text length
        roi_y_start = y - r- y_ratio
        roi_y_end = y + r+ y_ratio
        cv2.rectangle(output, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 0, 255), 2)
        rois_coordinates.append((roi_x_start, roi_y_start, roi_x_end, roi_y_end))
    return output, rois_coordinates

def extract_text_from_roi_checks(image_org, roi_coordinates):
    detected_options=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        detected_options.append(first_word)
    print('detected_options: ',detected_options)
    return detected_options

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

# page 22 methods
def filter_numbers(detected_list):
    # Create a list of numbers 1-9 in various formats
    numbers = [str(i) for i in range(1, 10)]
    numbers_with_period = [str(i) + '.' for i in range(1, 10)]
    numbers_with_comma_space = [str(i) + ',' for i in range(1, 10)]

    # Combine all the formats into one list
    all_valid_formats = numbers + numbers_with_period + numbers_with_comma_space
    
    return [item for item in detected_list if item in all_valid_formats]

def map_numbers_to_cells(filtered_numbers):
    # Mapping of numbers to Excel cells
    number_to_cell_map = {
        '1': 'F14','1.': 'F14','1,': 'F14',
        '2': 'F15','2.': 'F15','2,': 'F15',
        '3': 'F16','3.': 'F16','3,': 'F16',
        '4': 'F17','4.': 'F17','4,': 'F17',
        '5': 'F18','5.': 'F18','5,': 'F18',
        '6': 'F19','6.': 'F19','6,': 'F19',
        '7': 'F20','7.': 'F20','7,': 'F20',
        '8': 'F21','8.': 'F21','8,': 'F21',
        '9': 'F22','9.': 'F22','9,': 'F22'
    }
    
    # Create a dictionary with cell address as key and number (without dot) as value
    excel_inputs = {number_to_cell_map[num]: num[:-1] for num in filtered_numbers}
    return excel_inputs

if __name__ == "__main__":
    pdf_path = ('FOR_UPWORK_#1.pdf')
    excel_path = ('WORKING.xlsm')
    page=22
    #***********************************valus that can be tweaked************************************************
    #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
    thresh_check_ratio=180
    thresh_radio_ratio=120
    thresh_OCR_ratio=150

    filled_check_ratio=0.6
    #ratio ta3 tol w 3ard el ROIs ta3 koul button
    x_check_ratio=9
    y_check_ratio=12

    image_org = pdf_to_image(pdf_path,page)
    thresh_check = preprocess_image(image_org,thresh_check_ratio)
    thresh_radio = preprocess_image(image_org,thresh_radio_ratio)
    output_image, squares = detect_checkboxes(thresh_check,image_org)

    filled_check_buttons,output_image = detect_filled_button(thresh_radio,squares,output_image,filled_check_ratio)

    img_for_OCR= preprocess_image(image_org,thresh_OCR_ratio)
    Image.fromarray(img_for_OCR).show()
    img_for_OCR2= preprocess_image(image_org,140)
    excel_inputs={}
    if  filled_check_buttons:
        output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
        print('********************************check_text*************************************************************')
        detected_check_text= extract_text_from_roi_checks(img_for_OCR, roi_check_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma


        filtered_numbers = filter_numbers(detected_check_text)
        print(filtered_numbers)
        excel_inputs = map_numbers_to_cells(filtered_numbers)
        print(excel_inputs)


    else:
        print("No filled button detected.")
    Image.fromarray(output_image).show()
    #******************updating the excel file*************************************
    update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py