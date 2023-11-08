import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image ,ImageDraw, ImageEnhance
import pytesseract
from openpyxl import load_workbook
import xlwings as xw
from fuzzywuzzy import fuzz #You're trying to find the location of a phrase in an image using OCR (Optical Character Recognition) with the pytesseract library. Due to the inaccuracies in OCR, you may get slightly incorrect readings of the phrase. One way to handle this is to use fuzzy string matching to detect phrases that are "close enough" to the target phrase.
from math import atan2, degrees
import os
import tempfile
from io import BytesIO
import PyPDF2
import fitz
import re
from bs4 import BeautifulSoup
def pdf_to_image(pdf_path, page, zoom=200 / 72):
    # Load the PDF
    doc = fitz.open(pdf_path)
    
    # Get the desired page
    page = doc.load_page(page - 1)
    
    # Create a matrix for zooming
    mat = fitz.Matrix(zoom, zoom)
    
    # Get the pixmap using the matrix for higher resolution
    pix = page.get_pixmap(matrix=mat)

    if pix.alpha:
        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)  # RGBA
    else:
        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)  # RGB
    
    return image_np

def pdf_to_image_test(pdf_path,page):
    """
    Flatten the input PDF and convert a given page to an image.
    """
    # Read the PDF
    with open(pdf_path, "rb") as input_file:
        reader = PyPDF2.PdfReader(input_file)
        writer = PyPDF2.PdfWriter()

        # Flatten the 10th page
        page_content = reader.pages[9]  # 0-based index, so 9 refers to the 10th page
        page_content.merge_page(page_content)
        writer.add_page(page_content)
        
        # Save the flattened content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            writer.write(temp_file)
            temp_file_path = temp_file.name
        
    # Convert the temporary PDF to an image
    images = convert_from_path(temp_file_path)
    
    # Optionally: remove the temporary file
    os.remove(temp_file_path)
    
    return np.array(images[0])

def correct_skew(image):
    print('start tilt correc')
    # Convert the image to grayscale for line detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the Hough Line Transform method to detect lines in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use the Probabilistic Hough Line Transform method for better line detection
    min_line_length = 600
    max_line_gap = 40
    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    angles = []

    for line in lines_p:
        for x1, y1, x2, y2 in line:
            # Calculate the angle in radians and convert to degrees
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if angle<10 and angle>-10:
                angles.append(angle)

    # Average out the angles to get the tilt
    tilt_angle = np.mean(angles)

    print(tilt_angle)
    # Rotate the image to correct the tilt
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), tilt_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    print('end tilt correc')
    return rotated_image

def distance(box1, box2):
    # Compute the distance between the centers of two boxes
    return np.sqrt((box1[0] + box1[2]/2 - box2[0] - box2[2]/2)**2 + (box1[1] + box1[3]/2 - box2[1] - box2[3]/2)**2)

def preprocess_image(image_org,ratio):
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, ratio, 255, cv2.THRESH_BINARY)
    return thresh

def detect_checkboxes(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28 and y>950 and y<1100:  # Check if side length is at least 18 pixels
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
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)

        center_x = x + w // 2
        center_y = y + h // 2
        # diameter =( w // 2 ) -2 # or h, since it's approximately a square  #kkanet ( w // 2 ) wkanet temchi m3a kamel les pdf li semouhoum for upwork
        #moraha seyit nrod diameter static tema daymen nafs el valeur mechi 3la hsab wach ydetecti l program wmchat bien 
        diameter =11
        # diameter=diameter-3#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        squares.append((center_x, center_y, diameter))
    
    return output, squares
#page24
def detect_radio_buttons(thresh, image_org):
    circles = cv2.HoughCircles(
        # thresh, cv2.HOUGH_GRADIENT, dp=1.35, minDist=30, param1=50, param2=25, minRadius=8, maxRadius=11
        # thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=50, param2=25, minRadius=20, maxRadius=30
        #**********************************************************************************************
        # thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=27, minRadius=9, maxRadius=15
        thresh, cv2.HOUGH_GRADIENT, dp=0.1, minDist=20, param1=50, param2=19, minRadius=9, maxRadius=15# kanet dp=0.1 w kanet temchi m3a kamel les pdf li smouhoum for upwork
    )
    output = image_org.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles=[]
        for i in range(circles.shape[0]):
            #radit r=10 daymen mechi 3la hsab cha ydetecti l code kima l checkboxes radithom diameter=11 haka wlat khir fel detection ta3 filled buttons pareceque daymen nafs l size men9bel kan kayen li ydetectihom kbar kayen li sghar tema hadak el ratio s3ib bach nhadedou ida 0.5 wela 0.6 welaa....
            x, y, r = circles[i]
            if (y>950 and y<1100 or y>750 and y<820) and x<1000:
                valid_circles.append((x, y, 10))
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    return output, valid_circles
#page24
def detect_radio_buttons2(thresh, image_org):
    circles = cv2.HoughCircles(
        # thresh, cv2.HOUGH_GRADIENT, dp=1.35, minDist=30, param1=50, param2=25, minRadius=8, maxRadius=11
        # thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=50, param2=25, minRadius=20, maxRadius=30
        #**********************************************************************************************
        # thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=27, minRadius=9, maxRadius=15
        thresh, cv2.HOUGH_GRADIENT, dp=0.1, minDist=20, param1=50, param2=19, minRadius=6, maxRadius=13# kanet dp=0.1 w kanet temchi m3a kamel les pdf li smouhoum for upwork
    )
    output = image_org.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles=[]
        for i in range(circles.shape[0]):
            #radit r=10 daymen mechi 3la hsab cha ydetecti l code kima l checkboxes radithom diameter=11 haka wlat khir fel detection ta3 filled buttons pareceque daymen nafs l size men9bel kan kayen li ydetectihom kbar kayen li sghar tema hadak el ratio s3ib bach nhadedou ida 0.5 wela 0.6 welaa....
            x, y, r = circles[i]
            if y>1200 and y<1600:
                valid_circles.append((x, y, 9))
                cv2.circle(output, (x, y), 9, (0, 255, 0), 2)
    return output, valid_circles
#page24
def detect_filled_button(thresh, circles,image_org, filled_threshold_ratio):
    output = image_org.copy()
    filled_buttons = []

    for (x, y, r) in circles:
        r=r-4#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023 kanet r=r-4 w kanet temchi m3a kamel li semouhoum for upwork
        roi = thresh[y-r:y+r, x-r:x+r]
        black_pixel_count = np.sum(roi == 0)
        total_pixel_count = np.pi * r * r
        if black_pixel_count / total_pixel_count > filled_threshold_ratio:
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            filled_buttons.append((x, y, r+4))

    return filled_buttons,output
#page24
def detect_word_location(img, word, length, threshold=80):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            bounding_boxes.append((x2, y1-10, x1+length, y2+5))

    return bounding_boxes
#page24
def detect_word_location2(img, word, length, threshold=60):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)-r, int(x2)+r, int(y2)+r#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            if x1>600 and x1<900 and y1>700 and y1<900:
                bounding_boxes.append((x1, y1, x2, y2))
    return bounding_boxes

def detect_word_location_old_method(img,word,length):
    # Step 2: Use OCR to detect the word "Address" and get its bounding boxes
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r=4
        if word in line:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)
            bounding_boxes.append((x2+2, y1, x2+length, y2))

    return bounding_boxes

def detect_phrase_location(img, phrase, length, threshold=85):
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    words = phrase.split()
    bounding_boxes = []

    lines = hocr_data.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Use fuzzy matching instead of exact matching
        if fuzz.partial_ratio(words[0], line) >= threshold:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            current_box = [x1, y1, x2, y2]

            all_words_found = True
            for word in words[1:]:
                i += 1
                if i < len(lines) and fuzz.partial_ratio(word, lines[i]) >= threshold:
                    x1, y1, x2, y2 = lines[i].split('bbox ')[1].split(';')[0].split()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Expand the bounding box to include this word
                    current_box[2] = max(current_box[2], x2)
                    current_box[3] = max(current_box[3], y2)
                else:
                    all_words_found = False
                    break

            if all_words_found:
                x1,y1,x2,y2=current_box
                current_box=(x2, y1-7, x2+length, y2)#kima dert m3a Other welit nehseb l x2 men x1 w nzid valeur kima hna 161 parceque l x2 wlat taghlat 
                bounding_boxes.append(tuple(current_box))
        i += 1

    return bounding_boxes

def detect_phrase_location_old_method(img,phrase,length):
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    words = phrase.split()
    bounding_boxes = []

    lines = hocr_data.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if the first word is in the line
        if words[0] in line:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            current_box = [x1, y1, x2, y2]

            all_words_found = True
            for word in words[1:]:
                i += 1
                if i < len(lines) and word in lines[i]:
                    x1, y1, x2, y2 = lines[i].split('bbox ')[1].split(';')[0].split()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Expand the bounding box to include this word
                    current_box[2] = max(current_box[2], x2)
                    current_box[3] = max(current_box[3], y2)
                else:
                    all_words_found = False
                    break

            if all_words_found:
                x1,y1,x2,y2=current_box
                current_box=(x2+2, y1-7, x2+length, y2)
                bounding_boxes.append(tuple(current_box))
        i += 1

    return bounding_boxes
#page24
def extract_text_roi(image_org, filled_buttons,x_ratio,y_ratio):
    output = image_org.copy()
    rois_coordinates=[]
    for x, y, r in filled_buttons:#ani dayerha twila bezaf lakhaterch ocr ye9ra ri lkelma lewla li le9raha tema
        r=r-4
        roi_x_start = x + r+7#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/8/2023
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
        if not(first_word=='Inability' and 'cope'==text.split(' ')[2]):
            detected_options.append(first_word)
    print('detected_options: ',detected_options)
    return detected_options
#page24
def extract_text_from_roi_radios(image_org, roi_coordinates):
    detected_options=[]
    detected_yes_no={}
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
    
        if fuzz.partial_ratio('No',first_word) > 80:
            roi_no = image_org[roi_y_start:roi_y_end, roi_x_start-120:roi_x_start-30]
            text_no=''
            text_no = pytesseract.image_to_string(roi_no, config='--psm 6').strip().split(' ')[-1]
            cv2.rectangle(image_org, (roi_x_start-120, roi_y_start), (roi_x_start-30, roi_y_end), (0, 0, 255), 2)
            # Image.fromarray(image_org).show()
            if fuzz.partial_ratio('Cough',text_no) > 80:
                detected_yes_no['Cough']='No'
            else:
                detected_yes_no[text_no]='No'
        elif fuzz.partial_ratio('Yes',first_word) > 80:
            roi_yes = image_org[roi_y_start:roi_y_end, roi_x_start-200:roi_x_start-107]
            text_yes=''
            text_yes = pytesseract.image_to_string(roi_yes, config='--psm 6').strip().split(' ')[-1]
            cv2.rectangle(image_org, (roi_x_start-200, roi_y_start), (roi_x_start-107, roi_y_end), (0, 0, 255), 2)
            # Image.fromarray(image_org).show()
            if fuzz.partial_ratio('Cough',text_yes) > 75:
                detected_yes_no['Cough']='Yes'
            else:
                detected_yes_no[text_yes]='Yes'
        else:
            detected_options.append(first_word)

    print('detected_yes_no: ',detected_yes_no)

    print('detected_options: ',detected_options)
    return detected_options,detected_yes_no

def extract_text_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    text = pytesseract.image_to_string(roi_image).strip()
    # text = pytesseract.image_to_string(roi_image, config='--psm 6 -c tessedit_char_blacklist=.-+_,|;:').strip()
    return text

def extract_text_from_roi_line(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()

    text = pytesseract.image_to_string(roi_image,config='--psm 7').strip()

    # text = pytesseract.image_to_string(roi_image, config='--psm 6 -c tessedit_char_blacklist=.-+_,|;:').strip()
    return text

def validate_option(detected_text, options, threshold=80):
    validated_options = []

    for text in detected_text:
        for option in options:
            # Using token sort ratio to handle out of order issues and partial matches
            if fuzz.token_sort_ratio(text, option.split()[0]) > threshold:
                validated_options.append(option)

    if len(validated_options) == 0:
        return None
    else:
        print('validated options:', validated_options)
        return validated_options

def validate_option_old_method(detected_text,options):
    validated_options=[]
    for text in detected_text:
        for option in options:
            if text in option:
                validated_options.append(option)
    if len(validated_options)==0:
        return None
    else :
        print('validated options:',validated_options)
        return validated_options
#page 24
def update_excel_sheet(path,inputs):
    start_time = time.time()
    print ('opening the exel sheet...')
    app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
    wb = app.books.open(path)
    ws = wb.sheets['Basic- Auto']
    
    #page 24 verification
    # Flag to check if "r600" is found
    found_r600 = False
    
    # Check cells B69 to B89
    for i in range(69, 90):
        cell_value = ws.range(f'B{i}').value
        if cell_value == 'r600':
            found_r600 = True
            break  # exit loop if "r600" is found
    
    # Update cell B147 based on the flag
    if found_r600:
        inputs['B147'] = 'localized'
    else:
        inputs['B147']= 'Normal'
    
    
    for cell in inputs:
        ws.range(cell).value = inputs[cell]
    

    
    
    
    # Save and close
    wb.save()
    wb.close()
    app.quit()
    end_time = time.time()
    print ('closing the exel sheet...')
    execution_time = end_time - start_time
    print(f"Time of Updating the Excel sheet: {execution_time:.2f} seconds")
    print(f"Updated Excel sheet '{path}' with :")
    for cell in inputs:    
        print(f"value '{inputs[cell]}' in cell '{cell}'.")

def update_excel_sheet_old_method(path,inputs):
    start_time = time.time()
    print ('start_time...')
    workbook = load_workbook(filename=path, keep_vba=True)
    sheet = workbook.active  
    for cell in inputs:
        sheet[cell] = inputs[cell]
    workbook.save(filename=path)
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
    start_time = time.time()
    # pdf_path = ('BAD_QUALITY_2.pdf')
    # for pdf_path in ['BAD_QUALITY_2.pdf','BAD_QUALITY_3.pdf','BAD_QUALITY.pdf','FILLABLES_2.pdf','FILLABLES_3.pdf','FILLABLES_4.pdf','FILLABLES_5.pdf','FILLABLES_6.pdf','FOR_UPWORK_#1.pdf','FOR_UPWORK_#2.pdf','FOR_UPWORK_#3.pdf','FOR_UPWORK_#4.pdf','HIGH_QUALITY_2.pdf','HIGH_QUALITY_3.pdf','HIGH_QUALITY.pdf','MEDIUM_QUALITY.pdf']:
    for pdf_path in ['HIGH_QUALITY_2.pdf','HIGH_QUALITY_3.pdf','HIGH_QUALITY.pdf','MEDIUM_QUALITY.pdf']:
        print(pdf_path)
        excel_path = ('WORKING.xlsm')
        page=24
        noise=False# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
        plure=False
        skewed=True
        if pdf_path=='FOR_UPWORK_#4.pdf':
            noise=True# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
            plure=True
            skewed=True
        elif pdf_path=='HIGH_QUALITY_3.pdf' or pdf_path=='BAD_QUALITY_3.pdf' :
            skewed=True
        #***********************************valus that can be tweaked************************************************
        #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_radio_ratio=30
        y_radio_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        image_org = pdf_to_image(pdf_path,page)
        # Image.fromarray(image_org).show()
        #kayen des pdf ki mscanyiin mayliin 
        if skewed:
            corrected = correct_skew(image_org)
            image_org=corrected
            # Image.fromarray(corrected).show()


        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #hada denoiser ynahi el noise li kayen f ba3d les pdf lakin ma ykhasarch l ketba ga3 tema ani ayrou daymen yemchi mechi ghi m3a li fihom el noise
        if noise:
            denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
            image_org=denoised
            # Image.fromarray(denoised).show()
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        # # # 4. Sharpening    hadi method khdokhra bach dir charpning besah masa3detnich li rani nekhdem biha dork khir hadi dir charpening 9awi bezaf li rani nekhdem biha 3la hsab ra9m li dirou 
        # kernel = np.array([[-1,-1,-1], 
        #                    [-1, 9,-1],
        #                    [-1,-1,-1]])
        # sharpened = cv2.filter2D(denoised, -1, kernel)
        # image_org=sharpened
        if plure:
            # Convert the numpy array to an Image object
            image_org_pil = Image.fromarray(image_org)

            # Sharpen the image
            enhancer = ImageEnhance.Sharpness(image_org_pil)
            image_sharpened_pil = enhancer.enhance(4.0)  #3.0 hiya li kanet temchi men 9bel # 4.0 is the enhancement factor; higher values mean more sharpening

            # Convert the PIL image back to a numpy array for visualization
            sharpened = np.array(image_sharpened_pil)
            image_org=sharpened
            # Image.fromarray(sharpened).show()

        #hna ki kout ndiir bezaf sharpening kan ykhorjou des trace noire fel blayes elboyed tema ki n detecti text fi blasa vide yehseb kayen text tema ,tema seyit ndir denoise pour une dexieme fois bach enahi hadouk kes traces
        # denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
        # image_org=denoised
        output_image=image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_radio = preprocess_image(output_image,thresh_radio_ratio)
        thresh_check = preprocess_image(output_image,thresh_check_ratio)

        output_image, circles = detect_radio_buttons(thresh_radio,output_image)

        filled_radio_buttons,output_image = detect_filled_button(thresh_check,circles,output_image,filled_radio_ratio)
        
        if filled_radio_buttons :
            output_image, roi_radio_coordinates = extract_text_roi(output_image, filled_radio_buttons,x_radio_ratio,y_radio_ratio)

            print('********************************radio_text*************************************************************')
            
            detected_radio_text,detected_radio_yesNo= extract_text_from_roi_radios(image_org, roi_radio_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma

            excel_inputs={}


            print('********************************Cough *************************************************************')
            
            options=['Productive','Non-productive']
            validated_radio_text=validate_option(detected_radio_text,options)
            # Check the conditions and update excel_inputs['B56'] accordingly
            cough_status = detected_radio_yesNo.get('Cough', '')  # Default to empty string if 'Cough' is not in the dict
            if cough_status == 'No':
                pass  # excel_inputs['B56'] remains empty
            elif cough_status == 'Yes' or (validated_radio_text and 'Productive' in validated_radio_text) or (validated_radio_text and 'Non-productive' in validated_radio_text):
                if validated_radio_text and 'Productive' in validated_radio_text:
                    excel_inputs['B56'] = 'PRODUCTIVE COUGH'
                elif validated_radio_text and 'Non-productive' in validated_radio_text:
                    excel_inputs['B56'] = 'NON-PRODUCTIVE COUGH'
                else:
                    excel_inputs['B56'] = 'COUGH'


            print('********************************intermittent/continuous*************************************************************')
            
            options2=['intermittent','continuous']
            validated_radio_text2=validate_option(detected_radio_text,options2)
            if validated_radio_text2 and 'intermittent' in validated_radio_text2:
                excel_inputs['B59'] = 'intermittent'
            elif validated_radio_text2 and 'continuous' in validated_radio_text2:
                excel_inputs['B59'] = 'continuous'
        else:
            print("No filled button detected.")

        O2_ROI=detect_word_location2(image_org,'LPM',40)
        if O2_ROI:
            x1,y1,x2,y2=O2_ROI[0]
            print(x2-x1)
            if x2-x1<60:
                x1=x1-130
            elif x2-x1<155:
                x1=x1-60
            roi=x1,y1,x2,y2
            O2_text=extract_text_from_roi_line(image_org,roi)
            if '@' not in O2_text:
                x1=x1-60
                roi=x1,y1,x2,y2
                O2_text=extract_text_from_roi_line(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('O2 text:',O2_text)
            # regular expression pattern to match text between "O2@" and "LPM"
            pattern = r'@\s*(.*?)\s*LPM'

            # find matches in the string
            match = re.search(pattern, O2_text)
            # if a match is found, extract the text
            if match:

                extracted_O2_text = match.group(1).replace('_','')
                print('O2 text:',extracted_O2_text)
                excel_inputs['C59']=extracted_O2_text
            else:
                print('there is nothing in o2@')
        else:
            print('cannot detect the word o2')
        print('********************************circulation*************************************************************')

        output_image, table_circles = detect_radio_buttons2(thresh_radio,output_image)

        table_filled_buttons,output_image = detect_filled_button(thresh_check,table_circles,output_image,filled_radio_ratio)

        if table_filled_buttons:
            # Step 1: Sort the circles based on the x position
            sorted_circles = sorted(table_circles, key=lambda circle: circle[0])

            # Step 2 and 3: Divide the list into groups of 5 and sort each group based on y position
            final_sorted_circles = []
            for i in range(0, len(sorted_circles), 5):
                group = sorted(sorted_circles[i:i+5], key=lambda circle: circle[1])
                final_sorted_circles.extend(group)

            # Step 3: Extracting the position (row, column) of each circle in the given list
            positions = []
            for circle in table_filled_buttons:
                index = final_sorted_circles.index(circle)
                row = (index % 5) + 1
                column = (index // 5) + 1
                positions.append((row, column))

            print('marked positions in the table (row, column):',positions)

            # Mapping for rows and columns based on the description
            row_mapping = {1: '145', 2: '146'}
            column_mapping = { 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

            # Initialize excel_inputs dictionary

            # Step 4: Iterate through the positions and build the excel_inputs dictionary
            for row, column in positions:
                # Skip if the column or row is not in the mapping
                if column not in column_mapping or row not in row_mapping:
                    continue
                # Determine the Excel row and column based on the mapping
                excel_row = row_mapping[row]
                excel_column = column_mapping[column]

                # Assign the appropriate value to the cell ('1' or '0')
                value = '1' if column != 2 else '0'
                cell = f"{excel_column}{excel_row}"

                # Update the excel_inputs dictionary
                excel_inputs[cell] = value

            # Check if any row has no markings, and if so, leave the cells empty
            for row in ['145', '146']:
                cells = [f"{col}{row}" for col in ['C', 'D', 'E', 'F', 'G']]
                if not any(cell in excel_inputs for cell in cells):
                    for cell in cells:
                        excel_inputs[cell] = ''

            print(excel_inputs)
        else:
            print("No filled button detected.")


        print('********************************breath sounds*************************************************************')

        line1_ROI=detect_word_location(image_org,'Anterior:',800)
        line2_ROI=detect_word_location(image_org,'Posterior:',800)
        if line1_ROI:
            x1,y1,x2,y2=line1_ROI[0]
            if y2-y1 >40:
                y1=y1+4
                y2=y2-10
            roi=x1+10,y1,x1+150,y2
            # x11,y11,x22,y22=roi
            # cv2.rectangle(output_image, (x11, y11), (x22, y22), (255, 0, 255), 2)
            line1=extract_text_from_roi(image_org,roi)
            roi=x1+10+300,y1,x2,y2
            # x11,y11,x22,y22=roi
            # cv2.rectangle(output_image, (x11, y11), (x22, y22), (255, 0, 255), 2)
            line1=line1+' '+extract_text_from_roi(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('line1 text:',line1.replace('\n',' ').replace('.',' ').replace('_',''))
        if line2_ROI:
            x1,y1,x2,y2=line2_ROI[0]
            print(y2-y1)
            if y2-y1 >38:
                y1=y1+8
            roi=x1,y1,x2,y2
            line2=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('line2 text:',line2.replace('\n',' ').replace('.',' ').replace('_',''))
        if line2_ROI:
            x1,y1,x2,y2=line2_ROI[0]
            roi=x1,y2+6,x2,y2+40
            print(y2+6-y2+40)
            line3=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y2+6), (x2, y2+40), (255, 0, 255), 2)
            print('line3 text:',line3.replace('\n',' ').replace('.',' ').replace('_',''))

        # Adjusted regular expressions to extract field texts
        regex = re.compile(r'(Right|Left)(\sUpper|\sLower)?\s*(\w+)')

        # Adjusted Mapping of field texts to cell numbers
        cell_mapping = {
            'Right': 'C150',
            'Left': 'D150',
            'Right Upper': 'C151',
            'Left Upper': 'D151',
            'Right Lower': 'C152',
            'Left Lower': 'D152'
        }

        # Extracting field texts and constructing excel_inputs dictionary
        for i, line in enumerate([line1.replace('\n',' ').replace('.',' ').replace('_',''), line2.replace('\n',' ').replace('.',' ').replace('_',''), line3.replace('\n',' ').replace('.',' ').replace('_','')], start=1):
            matches = regex.findall(line)
            for match in matches:
                field_text = match[2].strip()  # Extracting field text
                position = f'{match[0]} {match[1].strip()}'.strip()  # Constructing position
                print(field_text)
                if field_text.lower() not in ["clear", "c", "cta"]:
                    cell = cell_mapping.get(position)  # Getting cell number
                    if cell:
                        excel_inputs[cell] = field_text.upper()  # Adding to excel_inputs


        Image.fromarray(output_image).show()
            #******************updating the excel file*************************************
        update_excel_sheet(excel_path, excel_inputs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.2f} seconds")
# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py