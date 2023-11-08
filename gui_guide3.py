import time
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image ,ImageDraw, ImageEnhance
import pytesseract
from openpyxl import load_workbook
import xlwings as xw
from fuzzywuzzy import fuzz #You're trying to find the location of a phrase in an image using OCR (Optical Character Recognition) with the pytesseract library. Due to the inaccuracies in OCR, you may get slightly incorrect readings of the phrase. One way to handle this is to use fuzzy string matching to detect phrases that are "close enough" to the target phrase.
import fitz
import os
import re

# default fcts nafshom ta3 l page 10
def pdf_to_image(pdf_path,page):
    images = convert_from_path(pdf_path)
    return np.array(images[page-1])

def pdf_to_images(pdf_path, zoom=200/72):
    # Load the PDF
    doc = fitz.open(pdf_path)
    
    # List to hold images for all pages
    images = []
    
    # Iterate over all pages
    for page_num in range(doc.page_count):
        # Get the current page
        page = doc.load_page(page_num)
        
        # Create a matrix for zooming
        mat = fitz.Matrix(zoom, zoom)
        
        # Get the pixmap using the matrix for higher resolution
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to numpy array
        if pix.alpha:
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)  # RGBA
        else:
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)  # RGB
        
        # Append the image to the list
        images.append(image_np)
    
    return images

def correct_skew(image):
    # print('start tilt correc')
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

    # print(tilt_angle)
    # Rotate the image to correct the tilt
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), tilt_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    # print('end tilt correc')
    return rotated_image

def distance(box1, box2):
    # Compute the distance between the centers of two boxes
    return np.sqrt((box1[0] + box1[2]/2 - box2[0] - box2[2]/2)**2 + (box1[1] + box1[3]/2 - box2[1] - box2[3]/2)**2)

def preprocess_image(image_org,ratio):
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, ratio, 255, cv2.THRESH_BINARY)
    return thresh

def preprocess_image_and_smouth_lines(image_org,ratio):

        # Convert the image to grayscale
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to binarize
    _, binary = cv2.threshold(gray, ratio, 255, cv2.THRESH_BINARY_INV)

    # Use morphological transformations to smooth the lines
    kernel_length = np.array(binary).shape[1]//80  # Define the kernel size
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Use vertical kernel to detect and smooth vertical lines
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # Use horizontal kernel to detect and smooth horizontal lines
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    # Combine the two binary images
    table_segment = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    table_segment = cv2.erode(~table_segment, kernel, iterations=2)
    _, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    # Invert the image for visualization
    # table_segment = cv2.cvtColor(~table_segment, cv2.COLOR_GRAY2BGR)
    # Image.fromarray(table_segment).show()

    return table_segment

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
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)

        center_x = x + w // 2
        center_y = y + h // 2
        # diameter =( w // 2 ) -2 # or h, since it's approximately a square  #kkanet ( w // 2 ) wkanet temchi m3a kamel les pdf li semouhoum for upwork
        #moraha seyit nrod diameter static tema daymen nafs el valeur mechi 3la hsab wach ydetecti l program wmchat bien 
        diameter =11
        # diameter=diameter-3#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        squares.append((center_x, center_y, diameter))
    
    return output, squares

def delet_drop_arrow(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 25 and w <= 35 and h >= 12 and h <=25:  # Check if side length is at least 18 pixels
                # print(w,h)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x-3, y-4), (x+w-10, y+h+4), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y-4), (x, y+1), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y+h-1), (x, y+h+4), (255, 255, 255), -1)

    return output, filtered_boxes

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

        for i in range(circles.shape[0]):
            #radit r=10 daymen mechi 3la hsab cha ydetecti l code kima l checkboxes radithom diameter=11 haka wlat khir fel detection ta3 filled buttons pareceque daymen nafs l size men9bel kan kayen li ydetectihom kbar kayen li sghar tema hadak el ratio s3ib bach nhadedou ida 0.5 wela 0.6 welaa....
            x, y, r = circles[i]
            circles[i] = (x, y, 10)
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    return output, circles

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
            filled_buttons.append((x, y, r))

    return filled_buttons,output

def detect_word_location(img, word, length, threshold=80):
    # Crop the image based on the specified y-axis values
    cropped_img = img[850:, :]
    hocr_data = pytesseract.image_to_pdf_or_hocr(cropped_img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)+850, int(x2)+r, int(y2)+850#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            bounding_boxes.append((x1+97, y1, x1+97+length, y2))

    return bounding_boxes

def detect_word_location2(img, word, length, threshold=75):
    # Crop the image based on the specified y-axis values
    # cropped_img = img[850:, :]
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []
    first=True
    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold :
            if first:
                first=False
            else:
                # print(line)
                x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
                bounding_boxes.append((x2+94, y1-7, x2+length, y2))

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
                current_box=(x1+174, y1-7, x1+168+length, y2)#kima dert m3a Other welit nehseb l x2 men x1 w nzid valeur kima hna 161 parceque l x2 wlat taghlat 
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

def extract_text_roi(image_org, filled_buttons,x_ratio,y_ratio):
    output = image_org.copy()
    rois_coordinates=[]
    for x, y, r in filled_buttons:#ani dayerha twila bezaf lakhaterch ocr ye9ra ri lkelma lewla li le9raha tema
        #kanet +5
        roi_x_start = x + r+6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/8/2023
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

def extract_text_from_roi_radios(image_org, roi_coordinates):
    detected_options=[]
    detected_yes_no={}
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip() 
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word

        #hna kout dayer ta3amoul khas m3a ek yes w no bach ychouf lkelma li 9bel el yes wela no w ydirhom f dict mais memba3d l9it beli wach raw hab l client khati ga3 li fihoum yes wela no aya radithom commentaire bah ma ytha9louch l code fel batel
        # if 'No' in first_word:
        #     roi_no = image_org[roi_y_start:roi_y_end, roi_x_start-200:roi_x_start-30]
        #     text_no = pytesseract.image_to_string(roi_no, config='--psm 6').strip().split(' ')[-1]
        #     cv2.rectangle(image_org, (roi_x_start-300, roi_y_start), (roi_x_start-30, roi_y_end), (0, 0, 255), 2)
        #     # Image.fromarray(image_org).show()
        #     detected_yes_no[text_no]='No'
        # elif 'Yes' in first_word:
        #     roi_yes = image_org[roi_y_start:roi_y_end, roi_x_start-310:roi_x_start-100]
        #     text_yes = pytesseract.image_to_string(roi_yes, config='--psm 6').strip().split(' ')[-1]
        #     cv2.rectangle(image_org, (roi_x_start-390, roi_y_start), (roi_x_start-110, roi_y_end), (0, 0, 255), 2)
        #     # Image.fromarray(image_org).show()
        #     detected_yes_no[text_yes]='Yes'
        # else:
            # detected_options.append(first_word)

        detected_options.append(first_word)
    # print('detected_yes_no: ',detected_yes_no)

    print('detected_options: ',detected_options)
    return detected_options,detected_yes_no

def extract_text_from_roi(image_org, roi_coordinate,line=False):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    
    text = pytesseract.image_to_string(roi_image).strip()
    
    # text = pytesseract.image_to_string(roi_image, config='--psm 6 -c tessedit_char_blacklist=.-+_,|;:').strip()
    return text

def extract_text_from_roi_line(image_org, roi_coordinate,line=False):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    
    text = pytesseract.image_to_string(roi_image, config='--psm 7').strip()
    
    # text = pytesseract.image_to_string(roi_image, config='--psm 6 -c tessedit_char_blacklist=.-+_,|;:').strip()
    return text

def extract_numbers_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    number = pytesseract.image_to_string(roi_image, config='--psm 6 outputbase digits').strip()
    return number

def extract_number_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    number = pytesseract.image_to_string(roi_image, config='--psm 10').strip()
    return number

def extract_number_from_roi_checks(image_org, roi_coordinates):
    detected_options=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6 outputbase digits').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if not(first_word=='Inability' and 'cope'==text.split(' ')[2]):
            detected_options.append(first_word)
    print('detected_options: ',detected_options)
    return detected_options

def validate_option(detected_text, options, threshold=80):
    validated_options = []

    for text in detected_text:
        for option in options:
            # Using token sort ratio to handle out of order issues and partial matches
            if fuzz.token_sort_ratio(text, option.split()[0]) > threshold:
                if option=="Ist":
                    option='1st'
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

def update_excel_sheet(path,inputs,update_b147=True):
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
    if found_r600 and update_b147:
        inputs['B147'] = 'localized'
    else:
        inputs['B147']= 'Normal'
    
    
    for cell in inputs:
        ws.range(cell).value = inputs[cell]


    # Save and close
    wb.save()
    value_from_L4 = ws.range('L4').value
    value_from_O4 = ws.range('O4').value
    value_from_R4 = ws.range('R4').value
    wb.close()
    app.quit()
    end_time = time.time()
    print ('closing the exel sheet...')
    execution_time = end_time - start_time
    print(f"Time of Updating the Excel sheet: {execution_time:.2f} seconds")
    print(f"Updated Excel sheet '{path}' with :")
    for cell in inputs:    
        print(f"value '{inputs[cell]}' in cell '{cell}'.")
    return value_from_L4,value_from_O4,value_from_R4

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

# modified fcts
def detect_checkboxes0_p6(thresh, img, x1, y1, x2, y2):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h

            # Check if the bounding box lies within the needed_area
            
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28:
                if x >= x1 and (x + w) <= x2 and y >= y1 and (y + h) <= y2:
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

def detect_word_location_p18(img, word, length, threshold=90):
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)-r, int(x2)+r, int(y2)+r
            bounding_boxes.append((x2+12, y1, x2+length, y2))

    return bounding_boxes

def detect_filled_button_p9(thresh, circles,image_org, filled_threshold_ratio):
    output = image_org.copy()
    filled_buttons = []
    filled_buttons_number=[]#page 9----------------------
    for i, (x, y, r) in enumerate(circles):
        r=r-4#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023 kanet r=r-4 w kanet temchi m3a kamel li semouhoum for upwork
        roi = thresh[y-r:y+r, x-r:x+r]
        black_pixel_count = np.sum(roi == 0)
        total_pixel_count = np.pi * r * r
        if black_pixel_count / total_pixel_count > filled_threshold_ratio:
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            # print(i)
            if i in [0,1,2,3,4,5,6]:
                filled_buttons_number.append(inverse_number(i))#page 9----------------------
    print(filled_buttons_number)
    return filled_buttons_number,output

def detect_filled_button_p22(thresh, circles,image_org, filled_threshold_ratio): 
    output = image_org.copy()
    filled_buttons = []
    filled_buttons_number=[]#page 9----------------------
    for i, (x, y, r) in enumerate(circles):
        r=r-4#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023 kanet r=r-4 w kanet temchi m3a kamel li semouhoum for upwork
        roi = thresh[y-r:y+r, x-r:x+r]
        black_pixel_count = np.sum(roi == 0)
        total_pixel_count = np.pi * r * r
        if black_pixel_count / total_pixel_count > filled_threshold_ratio:
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            # print(i)
            if i in [0,1,2,3,4,5,6,7,8,9]:
                filled_buttons_number.append(inverse_number_p22(i))#page 9----------------------
    print(filled_buttons_number)
    return filled_buttons_number,output

def detect_checkboxes_p9(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28 and y >1500 and y<2030:  # Check if side length is at least 18 pixels
                # print(y)
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

def detect_checkboxes_p19(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 85 and w <= 130 and h >= 75 and h <= 115 and y<700:  # Check if side length is at least 18 pixels
                # print(w,h)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_checkboxes_p24(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 105 and w <= 120 and h >= 150 and h <= 180:  # Check if side length is at least 18 pixels
                # print(w,h)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def extract_text_roi_p22(image_org, filled_buttons,x_ratio,y_ratio):
    output = image_org.copy()
    rois_coordinates=[]
    for x, y, r in filled_buttons:
        roi_x_start = x + r+55
        roi_x_end = x + x_ratio * r  
        roi_y_start = y - r- y_ratio
        roi_y_end = y + r+ y_ratio
        cv2.rectangle(output, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 0, 255), 2)
        rois_coordinates.append((roi_x_start, roi_y_start, roi_x_end, roi_y_end))

        #numero 6 
        roi_x_start = roi_x_start 
        roi_x_end = roi_x_end 
        roi_y_start = roi_y_start - 15
        roi_y_end = roi_y_end - 15
        cv2.rectangle(output, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 100, 100), 2)
        rois_coordinates.append((roi_x_start, roi_y_start, roi_x_end, roi_y_end))
    return output, rois_coordinates

def detect_checkboxes_p22p2(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 155 and w <= 185 and h >= 60 and h <= 75:  # Check if side length is at least 18 pixels
                # print( w, h)
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


    # Sort filtered_boxes based on x value in ascending order
    filtered_boxes.sort(key=lambda box: box[0])

    # Draw the filtered boxes on the image
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_checkboxes_p25(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28 and y>750 and y<920:  # Check if side length is at least 18 pixels
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

def detect_checkboxes0_p25(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            # w >= 20 and w <= 200 and h >= 43 and h <= 48
            # w >= 68 and w <= 74 and h >= 44 and h <= 50
            #********************************************************************************************
            if  ((w >= 90 and w <= 109) or (w >= 60 and w <= 74)) and h >= 32 and h <= 50 and y<600:  # Check if side length is at least 18 pixels
                # print(w,h,y)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_word_location_p25(img, word, length, threshold=80):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            if y1>750 and y1<920:
                print(y1)
                bounding_boxes.append((x1+92, y1-5, x1+92+length, y2))

    return bounding_boxes

def detect_phrase_location_p25(img, phrase, length, threshold=75):
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
                current_box=(x2+69, y1-10, x2+69+length, y2-6)#kima dert m3a Other welit nehseb l x2 men x1 w nzid valeur kima hna 161 parceque l x2 wlat taghlat 
                bounding_boxes.append(tuple(current_box))
        i += 1

    return bounding_boxes

def detect_filled_button_p35(thresh, circles,image_org, filled_threshold_ratio):
    output = image_org.copy()
    filled_buttons = []

    for (x, y, r) in circles:
        r=r-5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023 kanet r=r-4 w kanet temchi m3a kamel li semouhoum for upwork
        roi = thresh[y-r:y+r, x-r:x+r]
        black_pixel_count = np.sum(roi == 0)
        total_pixel_count = np.pi * r * r
        if black_pixel_count / total_pixel_count > filled_threshold_ratio:
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            filled_buttons.append((x, y, r))

    return filled_buttons,output

def detect_checkboxes_p35(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 12 and w <= 30 and y>1150:  # Check if side length is at least 18 pixels
                # print(w)
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
        diameter =( w // 2 ) -2 # or h, since it's approximately a square  #kkanet ( w // 2 ) wkanet temchi m3a kamel les pdf li semouhoum for upwork
        #moraha seyit nrod diameter static tema daymen nafs el valeur mechi 3la hsab wach ydetecti l program wmchat bien 

        diameter =10
        # diameter=diameter-3#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        squares.append((center_x, center_y, diameter))
    
    return output, squares

def detect_checkboxes_p22(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28 and y<800:  # Check if side length is at least 18 pixels
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

def extract_text_from_roi_p22p2(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()
    return text

#fct special b quelque page
#page 6 methods
def detect_needed_area(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if  w >= 545 and w <= 630 and h >= 275 and h <= 325:  # Check if side length is at least 18 pixels
                bounding_boxes.append((x, y, w, h))
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)
    return output, bounding_boxes

#page 18 methods
def extract_needed_section(img_np):
    # Step 2: Crop the image to the detected section.
    y1, y2 = ( 328, 1314)
    cropped_image = img_np[y1:y2,:]
    # Image.fromarray(cropped_image).show()
    return cropped_image

def detect_Total_aria(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if  w >= 85 and w <= 106 and h >= 40 and h <= 48:  # Check if side length is at least 18 pixels
                # print(w,h)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

# page 22 methods
def filter_numbers(detected_list):
    # Create a list of numbers 1-9 in various formats
    numbers = [str(i) for i in range(1, 10)]
    numbers_with_period = [str(i) + '.' for i in range(1, 10)]
    numbers_with_comma_space = [str(i) + ',' for i in range(1, 10)]

    # Combine all the formats into one list
    all_valid_formats = numbers + numbers_with_period + numbers_with_comma_space

    # Filter the list, remove comma or point if they exist, and ensure no duplicates
    filtered_list = []
    for item in detected_list:
        cleaned_item = item.replace('.', '').replace(',', '')
        if item in all_valid_formats and cleaned_item not in filtered_list:
            filtered_list.append(cleaned_item)

    return filtered_list

def map_numbers_to_cells(filtered_numbers):
    # Mapping of numbers to Excel cells
    number_to_cell_map = {
        1: 'F14',
        2: 'F15',
        3: 'F16',
        4: 'F17',
        5: 'F18',
        6: 'F19',
        7: 'F20',
        8: 'F21',
        9: 'F22'
        }

    # Create a dictionary with cell address as key and number (without dot) as value
    excel_inputs = {number_to_cell_map[num]: str(num) for num in filtered_numbers if num!=10}
    return excel_inputs

def inverse_number_p22(n):#hadi l fct bach te9leb el numero li nel9awah pareque l image ki teteskana teteskana mel taht lel foug tema li houwa lewel (1) rah ykoun 6 fel list li tekhroujena et etc tema nepelbouhoum b hadi
    mapping = {
        9: 1,
        8: 2,
        7: 3,
        6: 4,
        5: 5,
        4: 6,
        3: 7,
        2: 8,
        1: 9,
        0: 10

    }
    return mapping.get(n, "Invalid Input")

#page 9 methods
def inverse_number(n):#hadi l fct bach te9leb el numero li nel9awah pareque l image ki teteskana teteskana mel taht lel foug tema li houwa lewel (1) rah ykoun 6 fel list li tekhroujena et etc tema nepelbouhoum b hadi
    mapping = {
        6: 1,
        5: 2,
        4: 3,
        3: 4,
        2: 5,
        1: 6,
        0: 7
    }
    return mapping.get(n, "Invalid Input")

def map_numbers_to_cells_p9(filtered_numbers):
    # Mapping of numbers to Excel cells
    number_to_cell_map = {
        1: 'G56',
        2: 'G57',
        3: 'G58',
        4: 'G59',
        5: 'G60',
        6: 'G61',
        }
    
    # Create a dictionary with cell address as key and number (without dot) as value
    excel_inputs = {number_to_cell_map[num]: '1' for num in filtered_numbers if num!=7}
    return excel_inputs

#page 22 p2 methods
def format_site_texts(site1_text, site2_text, site3_text):
    # Function to handle special replacements
    def handle_special_replacements(text):
        if text == "BLE’s":
            return "HIP AREA"
        return text

    # Apply special replacements, remove "|" and line breaks, and filter out the empty sites
    sites = [handle_special_replacements(site).replace("|", "").replace("\n", " ") 
            for site in [site1_text, site2_text, site3_text] if site]

    # Use the join_strings function to format the text
    formatted_text = join_strings(sites)

    return formatted_text.upper()

# full pdf method:
def image_preparation(image_org,noise, plure, skewed):
    # Image.fromarray(image_org).show()
    #kayen des pdf ki mscanyiin mayliin 

    if skewed:
        print('correcting_skew')
        corrected = correct_skew(image_org)
        image_org=corrected
        # Image.fromarray(corrected).show()

    if noise:
        print('denoising')
        denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
        image_org=denoised
        # Image.fromarray(denoised).show()

    if plure:
        print('sharpning')
        # Convert the numpy array to an Image object
        image_org_pil = Image.fromarray(image_org)

        # Sharpen the image
        enhancer = ImageEnhance.Sharpness(image_org_pil)
        image_sharpened_pil = enhancer.enhance(10.0)  #3.0 hiya li kanet temchi men 9bel # 4.0 is the enhancement factor; higher values mean more sharpening

        # Convert the PIL image back to a numpy array for visualization
        sharpened = np.array(image_sharpened_pil)
        
        image_org=sharpened
        # Image.fromarray(sharpened).show()

    return image_org

def page6(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 6------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_aria_ratio=200
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        thresh_aria = preprocess_image(image_org,thresh_aria_ratio)

        filled_check_buttons=None

        output_image, needed_area = detect_needed_area(thresh_aria,output_image)
        # print(needed_area)
        # Image.fromarray(output_image).show()
        if needed_area:
            x,y,w,h=needed_area[0]
            x1,y1,x2,y2 =x, y, x+w, y+h
            output_image, squares = detect_checkboxes0_p6(thresh_check, output_image,x1,y1,x2,y2)

            filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        if  filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)

            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma


            print('***************************LEARNING_BARRIER***********************************************************')
            # Mapping from the first word to the full description
            word_mapping = {
                "Psychosocial": "Psychosocial",
                "Physical": "Physical",
                "Functional": "Functional cognition",
                "Mental": "Mental Health Disability",
                "Read": "Unable to read",
                "Write": "Unable to write"
            }

            # Convert the first word to the full description
            #old method
            # full_descriptions = [word_mapping[word] for word in detected_check_text if word in word_mapping]
            full_descriptions = []
            for detected_word in detected_check_text:
                for key in word_mapping:
                    if fuzz.token_sort_ratio(detected_word, key) > 90:
                        full_descriptions.append(word_mapping[key])
                        break  # Break the loop once a match is found for the detected word

            # Handle special cases for Read and Write
            if "Unable to read" in full_descriptions and "Unable to write" in full_descriptions:
                full_descriptions.remove("Unable to read")
                full_descriptions.remove("Unable to write")
                full_descriptions.append("Unable to read and write")

            # Use the provided join_strings function to create the final string
            LEARNING_BARRIER=join_strings(full_descriptions)
            print(f"LEARNING_BARRIER: {LEARNING_BARRIER}")
            excel_inputs['I30']=LEARNING_BARRIER

        else:
            print("No filled button detected.")
        if show_imgs:
            Image.fromarray(output_image).show()
    except:
        print("*the program was not able to continue reading the page 6 of this pdf!*")
    else:
        return excel_inputs

def page9(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 9------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)

        output_image, squares = detect_checkboxes_p9(thresh_check,output_image)
        #page 9----------------------
        filled_check_buttons_number,output_image = detect_filled_button_p9(thresh_check,squares,output_image,filled_check_ratio)

        if show_imgs:
            Image.fromarray(output_image).show()
        if filled_check_buttons_number:
            print('***************************bihavior***********************************************************')

            excel_inputs = map_numbers_to_cells_p9(filled_check_buttons_number)
            print(excel_inputs)

            return excel_inputs
        else:
            print("No filled button detected.")

    except:
        print("*the program was not able to continue reading the page 9 of this pdf!*")
    else:
        return excel_inputs

def page10(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 10------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_radio_ratio=18
        y_radio_ratio=6

        output_image= image_org.copy()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(output_image,thresh_check_ratio)
        thresh_radio = preprocess_image(output_image,thresh_radio_ratio)

        output_image, squares = detect_checkboxes(thresh_check,output_image)
        output_image, circles = detect_radio_buttons(thresh_radio,output_image)
        
        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)
        filled_radio_buttons,output_image = detect_filled_button(thresh_check,circles,output_image,filled_radio_ratio)

        if filled_radio_buttons or filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
            output_image, roi_radio_coordinates = extract_text_roi(output_image, filled_radio_buttons,x_radio_ratio,y_radio_ratio)

            print('********************************radio_text*************************************************************')
            detected_radio_text,detected_radio_yesNo= extract_text_from_roi_radios(image_org, roi_radio_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
            # ***************************marital status *************************************************************
            print('***************************marital status *************************************************************')
            marital_options = ["Single", "Married", "Divorced", "Widower"]
            validated_marital_options = validate_option(detected_radio_text,marital_options,80)
            if validated_marital_options!=None:
                marital_status = validated_marital_options[0]
                print(f"marital_status: {marital_status}")
                excel_inputs['I33']=marital_status
            else :
                print('nothing selected for marital status')

            # ***************************Feelings/emotions*************************************************************
            print('***************************Feelings/emotions***********************************************************')
            Feelings_emotions_options = ["Angry","Fear","Sadness","Discouraged","Lonely","Depressed","Helpless","Content","Happy","Hopeful","Motivated"]
            validated_Feelings_options = validate_option(detected_check_text,Feelings_emotions_options,80)
            #Other:
            
            other_ROI=detect_word_location(image_org,'other',300)


            x1,y1,x2,y2=other_ROI[0]
            if y2-y1>28:
                y=int(y1+(y2-y1)/2)
                y1=y-14
                y2=y+14
            roi=x1,y1,x2,y2
            Other_text=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Other:',Other_text)
            if Other_text=='' and validated_Feelings_options!=None:
                Feelings_emotions= join_strings(validated_Feelings_options)
                print(f"Feelings/emotions: {Feelings_emotions}")
                excel_inputs['I43']=Feelings_emotions
            elif validated_Feelings_options!=None:
                validated_Feelings_options.append(Other_text)
                Feelings_emotions= join_strings(validated_Feelings_options)
                print(f"Feelings/emotions: {Feelings_emotions}")
                excel_inputs['I43']=Feelings_emotions
            elif Other_text!='':
                Feelings_emotions=Other_text
                print(f"Feelings/emotions: {Feelings_emotions}")
                excel_inputs['I43']=Feelings_emotions
            else:
                print('nothing selected for Feelings/emotions')

            # ***************************Feelings/emotions*************************************************************
            print('***************************inability********************************************************************')
            inability_options = ["Lack of motivation","Inability to recognize problems","Unrealistic expectations","Denial of problems"]
            validated_inability_options = validate_option(detected_check_text,inability_options,80)
            if validated_inability_options!=None:
                inability= join_strings(validated_inability_options)
                print(f"inability: {inability}")
                excel_inputs['I44']=inability
            else:
                print('nothing selected for inability')

            # ***************************Spiritual_resource*************************************************************
            print('***************************Spiritual_resource***********************************************************')
            Spiritual_resource_ROI=detect_word_location2(image_org,'Spiritual',300,75)#75 ta3 threshold lazem tkoun 75 bah ydetecti Spiritual daymen koun 80 kayen pdf ma ye9rahach 
            if len(Spiritual_resource_ROI)!=0:
                x1,y1,x2,y2=Spiritual_resource_ROI[0]
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                Spiritual_resource=extract_text_from_roi(image_org,Spiritual_resource_ROI[0],True)
                print(f"Spiritual resource: {Spiritual_resource}")
                excel_inputs['I32']=Spiritual_resource
        else:
            print("No filled button detected.")
        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 10 of this pdf!*")
    else:
        return excel_inputs

def page18(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 18------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:    
        # image_org=image_preparation(image_org, plure=True)
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023


        needed_section_image=extract_needed_section(image_org)
        image_org=needed_section_image
        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        output_image, boxes = detect_Total_aria(thresh_check,output_image)

        # Image.fromarray(output_image).show()

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+4, y+9, x+w-60, y+h-9
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        Total_text=extract_numbers_from_roi(image_org,box)
        print('Total_text:',Total_text)
        # Image.fromarray(output_image).show()
        if Total_text not in ['0','1','2','3','4','5','6','7','8','9','10']:
            output_image, squares = detect_checkboxes(thresh_check,output_image)
            filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)
            if filled_check_buttons:
                filled_count = len(filled_check_buttons)
                print("number of marked boxes",filled_count)
                total=filled_count
            else:
                print("No filled button detected.")
                total=0
        else:
            total=int(Total_text)

        print(f"FALL RISKS TOTAL: {total}")
        excel_inputs['F65']=total

        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 18 of this pdf!*")
    else:
        return excel_inputs

def page19(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 19------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=200#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_box_ratio=80



        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        thresh_box = preprocess_image(image_org,thresh_box_ratio)

        can=True
        image_org, boxes = detect_checkboxes_p19(thresh_check,image_org)
        image_org, drop_arrow = delet_drop_arrow(thresh_box,image_org)


        if len(boxes)>0:
            x, y, w, h = boxes[0]
        else:
            thresh_check = preprocess_image_and_smouth_lines(image_org,thresh_box_ratio)
            image_org, boxes = detect_checkboxes_p19(thresh_check,image_org)
            if len(boxes)>0:
                x, y, w, h = boxes[0]
            else:
                can=False
        if can:
                
            x1,y1,x2,y2 =x+36, y+37, x+w-60, y+h-32
            # print(x1,y1,x2,y2)

            box=x1,y1,x2,y2
            urinary_text=extract_numbers_from_roi(image_org,box)
            if '4' in urinary_text:
                urinary_text='1'
            cv2.rectangle(image_org, (x1, y1), (x2, y2), (255, 0, 255), 2)
            if urinary_text not in ['0','1','2']:
                urinary_text='0'

            print('urinary_text:',urinary_text)
            if urinary_text in ['0','1','2']:
                urinary_number=int (urinary_text)

                urinary_mapping = {
                0: "No Incontinance",
                1: "Incontinent",
                2: "Foley"
            }

                urinary=urinary_mapping.get(urinary_number, "Invalid urinary_number")

                print(f"urinary: {urinary}")

                if urinary != "Invalid urinary_number":
                    excel_inputs['F53']=urinary


        else:
            print('cannot detect the box')
        if show_imgs:
            Image.fromarray(image_org).show()

    except:
        print("*the program was not able to continue reading the page 19 of this pdf!*")
    else:
        return excel_inputs

def page22(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 22------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=170

        filled_check_ratio=0.6
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=14
        y_check_ratio=6


        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)

        output_image, squares = detect_checkboxes_p22(thresh_check,output_image)
        #page 9----------------------
        filled_check_buttons_number,output_image = detect_filled_button_p22(thresh_check,squares,output_image,filled_check_ratio)

        if show_imgs:
            Image.fromarray(output_image).show()
        if filled_check_buttons_number:
            print('***************************Risk for Hospitalization***********************************************************')
            excel_inputs = map_numbers_to_cells(filled_check_buttons_number)
            print(excel_inputs)
        else:
            print("No filled button detected.")


    except:
        print("*the program was not able to continue reading the page 22 of this pdf!*")
    else:
        return excel_inputs

def page22p2(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 22p2------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        output_image= image_org.copy()

        #-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check_ratio=170

        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        # Image.fromarray(thresh_check).show()
        output_image, boxes = detect_checkboxes_p22p2(thresh_check,output_image)
        # Image.fromarray(output_image).show()

        if len(boxes)!=3:
            # print('cannot detect the sites, I will now try to smouth the lines :')
            thresh_check = preprocess_image_and_smouth_lines(image_org,thresh_check_ratio)
            # Image.fromarray(thresh_check).show()
            output_image, boxes = detect_checkboxes_p22p2(thresh_check,output_image)
            # Image.fromarray(output_image).show()

        if len(boxes)==3:
            x, y, w, h = boxes[0]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            box=x1,y1,x2,y2
            site1_text=extract_text_from_roi_p22p2(image_org,box)
            site1_text=site1_text.replace('‘','')
            print('site1:',site1_text)


            x, y, w, h = boxes[1]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box=x1,y1,x2,y2
            site2_text=extract_text_from_roi_p22p2(image_org,box)
            site2_text=site2_text.replace('‘','')
            print('site2:',site2_text)

            
            x, y, w, h = boxes[2]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            box=x1,y1,x2,y2
            site3_text=extract_text_from_roi_p22p2(image_org,box)
            site3_text=site3_text.replace('‘','')
            print('site3:',site3_text)



            # Assuming you've already gotten the values for site1_text, site2_text, and site3_text
            PAIN_ASSESSMENT = format_site_texts(site1_text, site2_text, site3_text)

            # Update the Excel sheet
            
            print(f"PAIN ASSESSMENT: {PAIN_ASSESSMENT}")
            excel_inputs['H20'] = PAIN_ASSESSMENT
        else:
            print('cannot detect the three sites')

        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 22p2 of this pdf!*")
    else:
        return excel_inputs

def page24(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 24------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=200#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)

        output_image, boxes = detect_checkboxes_p24(thresh_check,output_image)

        # Image.fromarray(output_image).show()


        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-61, y+h-100

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        dyspneic_text=extract_numbers_from_roi(image_org,box)
        if dyspneic_text not in ['0','1','2','3','4']:
            x1=x1-1

            box=x1,y1,x2,y2
            dyspneic_text=extract_numbers_from_roi(image_org,box)


        print('dyspneic_text:',dyspneic_text)
        dyspneic_number=int (dyspneic_text)
        
        dyspneic_mapping = {
        0: "NONE",
        1: "20 feet",
        2: "Moderate",
        3: "Minimal",
        4: "At rest"
    }

        dyspneic=dyspneic_mapping.get(dyspneic_number, "Invalid dyspneic_number")
        
        print(f"dyspneic: {dyspneic}")


        if dyspneic != "Invalid dyspneic_number":
            excel_inputs['B65']=dyspneic


        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 24 of this pdf!*")
    else:
        return excel_inputs

def page25(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 25------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=200#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_check_ratio=30
        y_check_ratio=6

        output_image= image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)

        output_image, boxes = detect_checkboxes0_p25(thresh_check,output_image)

        # Image.fromarray(output_image).show()

        for i in range(2):
            x, y, w, h = boxes[i]
            x1,y1,x2,y2 =x+7, y+8, x+w-int((x+w-x)/2), y+h-8
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

            box=x1,y1,x2,y2
            number=extract_numbers_from_roi(image_org,box)
            if i==1:
                print('Height:',number)
                excel_inputs['I8']=number
            else:
                print('weight:',number)
                excel_inputs['H8']=number


        
        #guid2
        output_image, squares = detect_checkboxes_p25(thresh_check,output_image)
        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates)
            print('***************************nutritional********************************************************************')
            nutritional_options = ["Renal", "NPO", "Controlled Carbohydrate", "General", "NAS"]
            validated_nutritional_options = validate_option(detected_check_text,nutritional_options,80)
        else:
            validated_nutritional_options=[]
            print("No filled button detected.")

        #other:
        other_ROI=detect_word_location_p25(image_org,'other',410)
        if other_ROI:
            x1,y1,x2,y2=other_ROI[0]
            roi=x1+1,y1,x2,y2
            # print(y2-y1)
            Other_text=extract_text_from_roi(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Other:',Other_text)
        else:
            Other_text=''
            print("cannot detect the word other")
        
        # Nutritional requirements (diet):
        requirements_ROI=detect_phrase_location_p25(image_org, 'Nutritional requirements', 480)
        if requirements_ROI:
            x1,y1,x2,y2=requirements_ROI[0]
            roi=x1,y1,x2,y2
            requirements_text=extract_text_from_roi(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Nutritional requirements (diet):',requirements_text)
        else:
            requirements_ROI=detect_phrase_location_p25(image_org, 'Nutritionalrequirements', 480)
            if requirements_ROI:
                x1,y1,x2,y2=requirements_ROI[0]
                roi=x1,y1,x2,y2
                requirements_text=extract_text_from_roi(image_org,roi)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                print('Nutritional requirements (diet):',requirements_text)
            else:
                requirements_text=''
                print("cannot detect the phrase Nutritional requirements (diet):")

        unique_options=[]

        # Check if Other_text and requirements_text are the same using fuzz library
        if fuzz.ratio(Other_text.lower(), requirements_text.lower()) >= 80 :
            # If they are the same, use only one of them
            unique_options = validated_nutritional_options + [Other_text.lower()]
        else:
            unique_options = validated_nutritional_options + [Other_text.lower(), requirements_text.lower()]

        # Remove duplicates and any empty strings
        unique_options = list(filter(None, unique_options))
        # Use join_strings function to join all the options
        final_diet_text = join_strings(unique_options)

        if not final_diet_text.endswith(" diet") and final_diet_text!='':
            final_diet_text += " diet"


        print('nutritional status: ',final_diet_text.lower())

        # Update the excel_inputs dictionary for cell F48
        excel_inputs['F48'] = final_diet_text.lower()

        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print ("*the program was not able to continue reading the page 25 of this pdf!*")
    else:
        return excel_inputs

def page35(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 35------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)

        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023



        output_image= image_org.copy()

        thresh_check = preprocess_image(image_org,thresh_check_ratio)

        output_image, squares = detect_checkboxes_p35(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button_p35(thresh_check,squares,output_image,filled_check_ratio)

        if  filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)

            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates)# detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma



            options = ["Bath bench", "Cane", "Walker", "Wheelchair", "Grab Bars", "Hospital Bed", "commode"]
            validated_options = validate_option(detected_check_text,options,80)
            if validated_options!=None:
                
                # Mapping of numbers to Excel cells
                option_to_cell_map = {
                    "Bath bench": 'J55',
                    "Cane": 'J53',
                    "Walker": 'J54',
                    "Wheelchair": 'J57',
                    "Grab Bars": 'J56',
                    "Hospital Bed": 'J59',
                    "commode": 'J58',  
                    }

                excel_inputs = {option_to_cell_map[option]: 1 for option in validated_options}
                print(excel_inputs)

            else :
                print('nothing selected for marital status')

        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 35 of this pdf!*")
    else:
        return excel_inputs

#modified fct guide 2
def detect_checkboxes_p1(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            # w >= 20 and w <= 200 and h >= 43 and h <= 48
            # w >= 68 and w <= 74 and h >= 44 and h <= 50
            #********************************************************************************************
            if  w >= 450 and w <= 550 and h >= 150 and h <= 200 and y >1500:  # Check if side length is at least 18 pixels
                # print(w,h,y)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_word_location_p1(img, word, length, threshold=80):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            if y1<600:
                bounding_boxes.append((x2-3, y1-10, x2+length, y2-2))

    return bounding_boxes

def detect_checkboxes_p4(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            # w >= 20 and w <= 200 and h >= 43 and h <= 48
            # w >= 68 and w <= 74 and h >= 44 and h <= 50
            #********************************************************************************************
            if  w >= 530 and w <= 600 and h >= 30 and h <= 70 and y >270 and y<400:  # Check if side length is at least 18 pixels
                
                bounding_boxes.append((x, y, w, h))

    # Filter boxes based on distance
    filtered_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        keep = True
        for other_box in bounding_boxes:
            if distance(box, other_box) < 50:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    # Draw the filtered boxes on the image
    for box in filtered_boxes:
        x, y, w, h = box
        # print(w,h,y,'hhh')
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_checkboxes_p5(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28 and y>1070 and y<1320:  # Check if side length is at least 18 pixels
                # print(y)
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

def detect_checkboxes2_p5(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            # w >= 20 and w <= 200 and h >= 43 and h <= 48
            # w >= 68 and w <= 74 and h >= 44 and h <= 50
            #********************************************************************************************
            if  w >= 100 and w <= 200 and h >= 30 and h <= 70 and y>1850:  # Check if side length is at least 18 pixels
                # print( w, h, y)
                bounding_boxes.append((x, y, w, h))

    # Filter boxes based on distance
    filtered_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        keep = True
        for other_box in bounding_boxes:
            if distance(box, other_box) < 50:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    # Draw the filtered boxes on the image
    for box in filtered_boxes:
        x, y, w, h = box
        # print(w,h,y,'hhh')
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_radio_buttons_p5(thresh, image_org):
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
        filtered_circles=[]
        for i in range(circles.shape[0]):
            
            #radit r=10 daymen mechi 3la hsab cha ydetecti l code kima l checkboxes radithom diameter=11 haka wlat khir fel detection ta3 filled buttons pareceque daymen nafs l size men9bel kan kayen li ydetectihom kbar kayen li sghar tema hadak el ratio s3ib bach nhadedou ida 0.5 wela 0.6 welaa....
            x, y, r = circles[i]
            if y>1180 and y<1320:
                filtered_circles.append((x, y, 10))
            # circles[i] = (x, y, 10)
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    return output, filtered_circles

def detect_word_location_p5(img, word, length, threshold=80):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            if y1<600:
                # print(y1)
                bounding_boxes.append((x2+47, y1-5, x2+47+length, y2))

    return bounding_boxes

def extract_text_from_roi_checks_p6(image_org, roi_coordinates):
    detected_options=[]
    detected_options2=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if roi_x_start<900 and roi_x_start>600 :
            # print('-->',roi_y_start)  
            detected_options.append(first_word)
        elif roi_x_start<1300 and roi_x_start>900:
            # print('-->',roi_y_start)
            detected_options2.append(first_word)
    print('detected_options for deaf: ',detected_options)
    print('detected_options for hearing aid: ',detected_options)
    return detected_options,detected_options2

def extract_text_from_roi_checks2_p6(image_org, roi_coordinates):
    detected_options=[]
    detected_options2=[]
    detected_options3=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if roi_x_start<800 and roi_x_start>600:
            # print('-->',roi_y_start)
            detected_options.append(first_word)
        elif roi_x_start<1400 and roi_x_start>1150:
            # print('-->',roi_y_start)
            detected_options2.append(first_word)
        elif roi_x_start<1600 and roi_x_start>1400:
            # print('-->',roi_y_start)
            detected_options3.append(first_word)
    print('detected_options for blurred: ',detected_options)
    print('detected_options for glaucoma: ',detected_options2)
    print('detected_options for cataracta: ',detected_options3)

    return detected_options,detected_options2,detected_options3

def extract_text_from_roi_checks3_p6(image_org, roi_coordinates):
    detected_options=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        detected_options.append(first_word)
    return detected_options

def extract_numbers_from_roi_p6(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    number = pytesseract.image_to_string(roi_image, config='--psm 7').strip()
    return number

def delet_drop_arrow_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 25 and w <= 35 and h >= 12 and h <=25:  # Check if side length is at least 18 pixels
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x-1, y-4), (x+w-10, y+h+4), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y-4), (x, y+1), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y+h-1), (x, y+h+4), (255, 255, 255), -1)
        
        # cv2.rectangle(output, (x-2, y-4), (x, y+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y-4), (x, y+3), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-2, y+h-4), (x, y+h+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y+h-3), (x, y+h+4), (255, 255, 255), -1)

    return output, filtered_boxes

def detect_checkboxes_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28  and y<600 and x>600 and x<1300:  # Check if side length is at least 18 pixels
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

def detect_checkboxes2_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28  and y>800 and y<1000:  # Check if side length is at least 18 pixels
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

def detect_checkboxes3_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28  and y>1200 and y<1350:  # Check if side length is at least 18 pixels
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

def detect_word_location_p6(img, word, length, threshold=70):
    # Crop the image based on the specified y-axis values
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r = 4
        if fuzz.partial_ratio(line, word) >= threshold:
            # print(line)
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1), int(x2)+r, int(y2)#edt 750 lakhaterch rani dayer crop le teswira b 750 fel y axe tema lawem n3awed nzido
            if y1<1600 and y1>1400:
                bounding_boxes.append((x2+52, y1-10, x2+52+length, y2))


    return bounding_boxes

def detect_code_area_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 100 and w <= 130 and h >= 100 and h <= 200 and y<600:  # Check if side length is at least 18 pixels
                # print(w,h,y)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def detect_code_area2_p6(thresh,img):
    output = img.copy()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    bounding_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            #********************************************************************************************
            if w >= 100 and w <= 130 and h >= 170 and h <= 220 and y>600 and y<800:  # Check if side length is at least 18 pixels
                # print(w,h,y)
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
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255,255), 2)


    return output, filtered_boxes

def preprocess_image_and_smouth_lines_p6(image_org,ratio):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to binarize
    _, binary = cv2.threshold(gray, ratio, 255, cv2.THRESH_BINARY_INV)

    # Use morphological transformations to smooth the lines
    kernel_length = np.array(binary).shape[1]//40  # Define the kernel size
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Use vertical kernel to detect and smooth vertical lines
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # Use horizontal kernel to detect and smooth horizontal lines
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    # Combine the two binary images
    table_segment = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    table_segment = cv2.erode(~table_segment, kernel, iterations=2)
    _, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    # Invert the image for visualization
    # table_segment = cv2.cvtColor(~table_segment, cv2.COLOR_GRAY2BGR)
    # Image.fromarray(table_segment).show()

    return table_segment

def detect_radio_buttons_p24(thresh, image_org):
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

def detect_filled_button_p24(thresh, circles,image_org, filled_threshold_ratio):
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

def detect_word_location_p24(img, word, length, threshold=80):
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

def extract_text_roi_p24(image_org, filled_buttons,x_ratio,y_ratio):
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

def extract_text_from_roi_radios_p24(image_org, roi_coordinates):
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

def detect_radio_buttons2_p24(thresh, image_org):
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

def detect_word_location2_p24(img, word, length, threshold=60):
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

#fct special b quelque page
#page1
def validate_one_option(detected_text, options, threshold=80):
    validated_options = []
    normalized_detected_text = detected_text.upper()
    for option in options:
        # Using token_set_ratio to handle partial matches more effectively
        if fuzz.token_set_ratio(detected_text, option) > threshold:
            validated_options.append(option)

    if len(validated_options) == 0:
        return None
    else:
        print("validated option:", validated_options)
        return validated_options

def transform_date(date_str):
    try:
        month, day, year = date_str.split('/')
        
        # Pad day and month with 0 if they are single digit
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        
        # Convert two-digit year to four digits
        year = '20' + year
        
        # Join the elements with a period
        transformed_date = f"{month}.{day}.{year}"
    except:
        print("Unable to read the date. This may be because the field is empty or the date is not in the 'Month/Day/Year' format.")
        return ''
    return transformed_date

def determine_priority_value(priority_text):
    # Define possible texts for each priority level
    high_options = ["1", "HIGH"]
    medium_options = ["2", "MOD", "MEDIUM", "MODERATE"]
    low_options = ["3", "LOW"]

    # Validate options for detected text
    high_validated = validate_one_option(priority_text, high_options)
    medium_validated = validate_one_option(priority_text, medium_options)
    low_validated = validate_one_option(priority_text, low_options)

    # Determine the priority value based on validated options
    if high_validated:
        return "1 [HIGH]"
    elif medium_validated:
        return "2"
    elif low_validated:
        return "3 [LOW]"
    else:
        return None
#page4
def update_language_information(language_text, excel_inputs):
    # Initialize threshold for fuzzy matching
    threshold = 85
    
    # Format the language text as required (first letter capitalized, rest lowercase)
    formatted_language_text = language_text.strip().capitalize()
    
    # Check if the language is close enough to "English"
    if fuzz.ratio(formatted_language_text, "English") > threshold:
        excel_inputs['I28'] = formatted_language_text
        excel_inputs['I29'] = ''
        
    # Check if the language is close enough to "Armenian"
    elif fuzz.ratio(formatted_language_text, "Armenian") > threshold:
        excel_inputs['I28'] = formatted_language_text
        excel_inputs['I29'] = 'SN speaks Armenian'
        
    # For any other language
    else:
        excel_inputs['I28'] = formatted_language_text
        excel_inputs['I29'] = f"PCG speaks {formatted_language_text}"

    return excel_inputs

#page6
def update_dentures_cell(validated_DENTURES_options):
    # Initialize an empty string to store the phrase for cell H17
    h17_value = ""
    
    # If "Dentures" is not marked, there's nothing to do
    if "Dentures" not in validated_DENTURES_options:
        return None
    
    # Initialize a list to store individual phrases
    phrases = []
    
    # Check for "Upper" and "Lower"
    if "Upper" in validated_DENTURES_options and "Lower" in validated_DENTURES_options:
        phrases.append("UPPER AND LOWER")
    elif "Upper" in validated_DENTURES_options:
        phrases.append("UPPER")
    elif "Lower" in validated_DENTURES_options:
        phrases.append("LOWER")
        
    # Check for "Partial"
    if "Partial" in validated_DENTURES_options:
        phrases.append("PARTIAL")
    
    # Combine phrases and add "DENTURES" at the end
    h17_value = " ".join(phrases) + " DENTURES"
    
    return h17_value
    # Update the excel_inputs dictionary

def page1(image_org,noise, plure, skewed,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 1------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)
        output_image=image_org.copy()

        thresh_check_ratio=150#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        Date_ROI=detect_word_location_p1(image_org,'DATE:',80)
        if Date_ROI:
            x1,y1,x2,y2=Date_ROI[0]
            roi=x1,y1,x2,y2
            # print(y2-y1)
            Date_text=extract_text_from_roi_line(image_org,roi)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Date text:',Date_text)
            Date=transform_date(Date_text.replace('.',''))
            print('Date :',Date)
            excel_inputs["E140"] = Date
        else:
            print('cannot detect the date')

        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        # Image.fromarray(thresh_check).show()
        output_image, boxes = detect_checkboxes_p1(thresh_check,output_image)

        # Image.fromarray(output_image).show()


        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+70, y+50, x+w-80, y+h-88
        # print (y2-y1)
        if y2-y1>25:
            y2=y1+25

        box=x1,y1,x2,y2
        Priority_text=extract_text_from_roi_line(image_org,box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        priority_value = determine_priority_value(Priority_text)
        if priority_value:
            print('priority:',priority_value)
            excel_inputs["B14"] = priority_value
        else:
            Priority_number=extract_numbers_from_roi(image_org,box)
            priority_value = determine_priority_value(str(Priority_number))
            if priority_value:
                print('priority:',priority_value)
                excel_inputs["B14"] = priority_value
            else:
                box=x1+17,y1,x1+27,y2
                Priority_number=extract_number_from_roi(image_org,box)
                if Priority_number=='4':
                    priority_value = determine_priority_value('1')
                if priority_value:
                    print('priority:',priority_value)
                    excel_inputs["B14"] = priority_value
                else:
                    print('cannot detect the priority')
        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 1 of this pdf!*")
    else:
        return excel_inputs

def page4(image_org,noise, plure, skewed,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 4------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=150#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023


        output_image=image_org.copy()


        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        # Image.fromarray(thresh_check).show()
        output_image, boxes = detect_checkboxes_p4(thresh_check,output_image)

        # Image.fromarray(output_image).show()



        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+5, y+12, x+w-5, y+h-12
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        language_text=extract_text_from_roi_line(image_org,box)
        print(language_text)
        excel_inputs = update_language_information(language_text, excel_inputs)
        print(excel_inputs)

        if show_imgs:
            Image.fromarray(output_image).show()


    except:
        print("*the program was not able to continue reading the page 4 of this pdf!*")
    else:
        return excel_inputs

def page6p2(image_org,noise, plure, skewed,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 6p2------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=200#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_box_ratio=80
        thresh_lines_ratio=140


        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_check_ratio=5
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023


        output_image=image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        thresh_box = preprocess_image(image_org,thresh_box_ratio)
        image_org, drop_arrow = delet_drop_arrow_p6(thresh_box,image_org)
        output_image, drop_arrow = delet_drop_arrow_p6(thresh_box,output_image)
        thresh_code = preprocess_image(image_org,180)
        output_image, boxes = detect_code_area_p6(thresh_check,output_image)

        # Image.fromarray(image_org).show()

        if len(boxes)==0:
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines_p6(image_org,thresh_lines_ratio)
            output_image, boxes = detect_code_area_p6(thresh_check_smouth_lines,output_image)

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-85
        if y2-y1>36:
            y2=y2-10
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        hearing_text=extract_numbers_from_roi_p6(thresh_code,box)
        # if hearing_text not in ['0','1','2','3','4']:
        #     x1=x1+1
        #     box=x1,y1,x2,y2
        #     hearing_text=extract_numbers_from_roi(image_org,box)


        if '1' in hearing_text:
            hearing_text='1'
        if '2' in hearing_text:
            hearing_text='2'
        if '3' in hearing_text:
            hearing_text='3'
        if 'z' in hearing_text:
            hearing_text='2'
        if '4' in hearing_text:
            hearing_text='1'
        if '7' in hearing_text:
            hearing_text='2'
        if 'a' in hearing_text:
            hearing_text='2'


        if hearing_text not in ['0','1','2','3']:
            hearing_text='0'


        print('hearing_text:',hearing_text)


        if hearing_text in ['0','1','2','3']:
            hearing_number=int(hearing_text)
        else:
            hearing_number=4#bah fel dict yji "Invalid hearing_number" aya if hearing != "Invalid hearing_number": matemchich

        hearing_mapping = {
        0: "ADEQUATE",
        1: "MINIMAL",
        2: "MODERATLY IMPAIRED HEARING DIFFICULTY IN BILATERAL EARS",
        3: "HIGHLY IMPAIRED",
    }

        hearing=hearing_mapping.get(hearing_number, "Invalid hearing_number")

        print(f"hearing: {hearing}")

        if hearing != "Invalid hearing_number":
            excel_inputs['B65']=hearing

        # second part------------------------------------------------------------------------

        output_image, squares = detect_checkboxes_p6(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        detected_check_text_deaf=[]
        detected_check_text_hearing=[]
        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)

            print('********************************check_text*************************************************************')
            detected_check_text_deaf,detected_check_text_hearing= extract_text_from_roi_checks_p6(image_org, roi_check_coordinates)

        # Handle the Deaf condition
        deaf_status = ''
        if 'L' in detected_check_text_deaf and 'R' in detected_check_text_deaf:
            deaf_status = 'BILATERAL EAR DEAF'
        elif 'L' in detected_check_text_deaf:
            deaf_status = 'LEFT EAR DEAF'
        elif 'R' in detected_check_text_deaf:
            deaf_status = 'RIGHT EAR DEAF'

        # Handle the Hearing Aid condition
        hearing_aid_status = ''
        if 'L' in detected_check_text_hearing and 'R' in detected_check_text_hearing:
            hearing_aid_status = 'BILATERAL EAR HEARING AID'
        elif 'L' in detected_check_text_hearing:
            hearing_aid_status = 'LEFT HEARING AID'
        elif 'R' in detected_check_text_hearing:
            hearing_aid_status = 'RIGHT HEARING AID'
        
        # Combine the deaf and hearing aid statuses for cell I14
        if deaf_status and hearing_aid_status:
            excel_inputs['I14'] = f"{deaf_status} AND {hearing_aid_status}"
        elif deaf_status:
            excel_inputs['I14'] = deaf_status
        elif hearing_aid_status:
            excel_inputs['I14'] = hearing_aid_status


        #third part---------------------------------------------------------------------------------

        output_image, boxes = detect_code_area2_p6(thresh_check,output_image)
        # Image.fromarray(image_org).show()

        if len(boxes)==0:
            # print('smouthig...')
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines_p6(image_org,thresh_lines_ratio)
            output_image, boxes = detect_code_area2_p6(thresh_check_smouth_lines,output_image)

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-116
        # print(y2-y1)
        if y2-y1>36:
            y2=y2-10
            y1=y1+3
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2

        vision_text=extract_numbers_from_roi_p6(thresh_code,box)


        if '1' in vision_text:
            vision_text='1'
        if '2' in vision_text:
            vision_text='2'
        if '3' in vision_text:
            vision_text='3'

        if 'c' in vision_text:
            vision_text='0'
        if 'z' in vision_text:
            vision_text='2'
        if '4' in vision_text:
            vision_text='1'
        if '7' in vision_text:
            vision_text='2'
        if 'a' in vision_text:
            vision_text='2'
        if vision_text not in ['0','1','2','3','4']:
            vision_text='0'




        print('vision_text:',vision_text)
        


        if vision_text in ['0','1','2','3','4']:
            vision_number=int(vision_text)
        else:
            vision_number=5#bah fel dict yji "Invalid vision_number" aya if vision != "Invalid vision_number": matemchich

        vision_mapping = {
        0: "ADEQUATE VISION",
        1: "IMPAIRED VISION",
        2: "MODERATLY IMPAIRED VISION",
        3: "HIGHLY IMPAIRED VISION",
        4:"SEVERELY IMPAIRED VISION"}


        vision=vision_mapping.get(vision_number, "Invalid vision_number")

        print(f"vision: {vision}")

        if vision != "Invalid vision_number":
            excel_inputs['J13']=vision


        #fourth part---------------------------------------------------------------------------------


        output_image, squares = detect_checkboxes2_p6(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)
        # cv2.rectangle(output_image, (1400, 900), (1600, 920), (255, 0, 255), 4)
        detected_check_text_deaf=[]
        detected_check_text_hearing=[]
        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
        
            print('********************************check_text*************************************************************')
            detected_check_text_blurred,detected_check_text_glaucoma,detected_check_text_cataracta= extract_text_from_roi_checks2_p6(image_org, roi_check_coordinates)

        vision_list=[]
        # Handle the blurred condition
        blurred_status = ''
        if 'L' in detected_check_text_blurred and 'R' in detected_check_text_blurred:
            blurred_status = 'BILATERAL EYE BLURRED VISION'
        elif 'L' in detected_check_text_blurred:
            blurred_status = 'LEFT EYE BLURRED VISION'
        elif 'R' in detected_check_text_blurred:
            blurred_status = 'RIGHT EYE BLURRED VISION'
        if blurred_status:
            vision_list.append(blurred_status)
        # Handle the glaucoma Aid condition
        glaucoma_status = ''
        if 'L' in detected_check_text_glaucoma and 'R' in detected_check_text_glaucoma:
            glaucoma_status = 'BILATERAL EYE GLAUCOMA'
        elif 'L' in detected_check_text_glaucoma:
            glaucoma_status = 'LEFT EYE GLAUCOMA'
        elif 'R' in detected_check_text_glaucoma:
            glaucoma_status = 'RIGHT EYE GLAUCOMA'
        if glaucoma_status:
            vision_list.append(glaucoma_status)

        # Handle the cataracta Aid condition
        cataracta_status = ''
        if 'L' in detected_check_text_cataracta and 'R' in detected_check_text_cataracta:
            cataracta_status = 'BILATERAL EYE CATARACT'
        elif 'L' in detected_check_text_cataracta:
            cataracta_status = 'LEFT EYE CATARACT'
        elif 'R' in detected_check_text_cataracta:
            cataracta_status = 'RIGHT EYE CATARACT'
        if cataracta_status:
            vision_list.append(cataracta_status)

        vision=join_strings(vision_list)
        print('vision:',vision.upper())
        if vision:
            excel_inputs['I13']=vision.upper()

    #fivth part------------------------------------------------------------------------------------
        x_check_ratio=25
        y_check_ratio=6
        
        output_image, squares = detect_checkboxes3_p6(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)

            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks3_p6(image_org, roi_check_coordinates)
            print('***************************DENTURES status *************************************************************')
            DENTURES_options = ["Dentures", "Upper", "Lower", "Partial"]
            validated_DENTURES_options = validate_option(detected_check_text,DENTURES_options,80)
            if validated_DENTURES_options!=None:
                DENTURES=update_dentures_cell(validated_DENTURES_options)
                print("DENTURES: ",DENTURES)
                excel_inputs['H17'] = DENTURES

    #sixth part------------------------------------------------------------------------------------


        Educational_ROI=detect_word_location_p6(image_org,'Educational',300)
        if Educational_ROI:
            x1,y1,x2,y2=Educational_ROI[1]
            roi=x1,y1,x2,y2
            # print(y2-y1)
            Educational_text=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Educational level:',Educational_text)
            if Educational_text:
                # Capitalize the first letter and add it to excel_inputs
                excel_inputs['I31'] = Educational_text.capitalize()
        else:
            print("cannot detect the word 'Educational level' in this page")
        # Image.fromarray(image_org).show()
        if show_imgs:
            Image.fromarray(output_image).show()




    except:
        print("*the program was not able to continue reading the page 6p2 of this pdf!*")
    else:
        return excel_inputs

def page5(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 5------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_radio_ratio=10
        y_radio_ratio=6

        output_image=image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(output_image,thresh_check_ratio)
        thresh_radio = preprocess_image(output_image,thresh_radio_ratio)

        output_image, squares = detect_checkboxes_p5(thresh_check,output_image)
        output_image, circles = detect_radio_buttons_p5(thresh_radio,output_image)
        
        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)
        filled_radio_buttons,output_image = detect_filled_button(thresh_check,circles,output_image,filled_radio_ratio)

        if filled_radio_buttons or filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
            output_image, roi_radio_coordinates = extract_text_roi(output_image, filled_radio_buttons,x_radio_ratio,y_radio_ratio)

            print('********************************radio_text*************************************************************')
            detected_radio_text,detected_radio_yesNo= extract_text_from_roi_radios(image_org, roi_radio_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
            # ***************************marital status *************************************************************
            print('***************************IMMUNIZATIONS*************************************************************')
            IMMUNIZATIONS_options = ["initial vaccine series","Shingles","Tetanus","Pneumonia","Hepatitis C","Influenza"]
            validated_IMMUNIZATIONS_options = validate_option(detected_check_text,IMMUNIZATIONS_options,80)
            booster_options = ["Ist","1st", "2nd", "3rd", "4th", "5th"]
            validated_booster_options = validate_option(detected_radio_text,booster_options,80)
            if validated_booster_options:
                booster=validated_booster_options[0]
                print('booster: ',booster)
            else:
                booster=None
            
            if validated_IMMUNIZATIONS_options and "initial vaccine series" in validated_IMMUNIZATIONS_options:
                validated_IMMUNIZATIONS_options = list(reversed(validated_IMMUNIZATIONS_options))
                if booster:
                    validated_IMMUNIZATIONS_options[-1]=f"COVID-19 INITIAL VACCINE SERIES WITH {booster.upper()} BOOSTER"
                else:
                    validated_IMMUNIZATIONS_options[-1]="COVID-19 INITIAL VACCINE SERIES"
            else:
                if booster:
                    if validated_IMMUNIZATIONS_options:
                        validated_IMMUNIZATIONS_options.append(f"{booster.upper()} COVID BOOSTER")
                    else:
                        validated_IMMUNIZATIONS_options=[]
                        validated_IMMUNIZATIONS_options.append(f"{booster.upper()} COVID BOOSTER")

            if validated_IMMUNIZATIONS_options:
                IMMUNIZATIONS_status = join_strings(validated_IMMUNIZATIONS_options)
                print(f"IMMUNIZATIONS_status: {IMMUNIZATIONS_status}")
                excel_inputs['E8']=IMMUNIZATIONS_status
            else :
                print('nothing selected for IMMUNIZATIONS status')

        else:
            print("No filled button detected.")


        #guid2
        #Cancer:
        Cancer_ROI=detect_word_location_p5(image_org,'Cancer',300)
        if Cancer_ROI:
            x1,y1,x2,y2=Cancer_ROI[0]
            roi=x1,y1,x2,y2
            # print(y2-y1)
            Cancer_text=extract_text_from_roi(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Cancer:',Cancer_text)
            excel_inputs['B23']=Cancer_text
        else:
            Cancer_text=''
            print("cannot detect the word Cancer")



        output_image, boxes = detect_checkboxes2_p5(thresh_check,output_image)
        if boxes:
            # Image.fromarray(output_image).show()



            x, y, w, h = boxes[0]
            x1,y1,x2,y2 =x+5, y+12, x+w-5, y+h-12
            box=x1,y1,x2,y2
            Date_text=extract_text_from_roi(image_org,box)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            print('Date text:',Date_text)
            Date=transform_date(Date_text.replace('.',''))
            print('Date :',Date)
            excel_inputs["E136"] = Date
        else:
            print('cannot detect the date field')


        if show_imgs:
            Image.fromarray(output_image).show()


    except:
        print("*the program was not able to continue reading the page 5 of this pdf!*")
    else:
        return excel_inputs

def page24p2(image_org,noise, plure, skewed ,show_imgs):
    print('------------------------------------------------------------------------------------------------------page 24p2------------------------------------------------------------------------------------------------------')
    excel_inputs={}
    try:
        image_org=image_preparation(image_org,noise, plure, skewed)


        thresh_check_ratio=170#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_radio_ratio=30
        y_radio_ratio=6

        output_image=image_org.copy()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        thresh_radio = preprocess_image(output_image,thresh_radio_ratio)
        thresh_check = preprocess_image(output_image,thresh_check_ratio)

        output_image, circles = detect_radio_buttons_p24(thresh_radio,output_image)

        filled_radio_buttons,output_image = detect_filled_button_p24(thresh_check,circles,output_image,filled_radio_ratio)

        if filled_radio_buttons :
            output_image, roi_radio_coordinates = extract_text_roi_p24(output_image, filled_radio_buttons,x_radio_ratio,y_radio_ratio)

            print('********************************radio_text*************************************************************')
            
            detected_radio_text,detected_radio_yesNo= extract_text_from_roi_radios_p24(image_org, roi_radio_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma



            print('********************************Cough *************************************************************')
            
            options=['Productive','Non-productive']
            validated_radio_text=validate_option(detected_radio_text,options)
            # Check the conditions and update excel_inputs['B56'] accordingly
            cough_status = detected_radio_yesNo.get('Cough', '')  # Default to empty string if 'Cough' is not in the dict
            if cough_status == 'No':
                excel_inputs['B56'] = ''
            elif cough_status == 'Yes' or (validated_radio_text and 'Productive' in validated_radio_text) or (validated_radio_text and 'Non-productive' in validated_radio_text):
                if validated_radio_text and 'Productive' in validated_radio_text:
                    excel_inputs['B56'] = 'PRODUCTIVE COUGH'
                elif validated_radio_text and 'Non-productive' in validated_radio_text:
                    excel_inputs['B56'] = 'NON-PRODUCTIVE COUGH'
                else:
                    excel_inputs['B56'] = 'COUGH'
                print(excel_inputs['B56'])


            print('********************************intermittent/continuous*************************************************************')
            
            options2=['intermittent','continuous']
            validated_radio_text2=validate_option(detected_radio_text,options2)
            if validated_radio_text2 and 'intermittent' in validated_radio_text2:
                excel_inputs['B59'] = 'intermittent'
            elif validated_radio_text2 and 'continuous' in validated_radio_text2:
                excel_inputs['B59'] = 'continuous'
        else:
            print("No filled button detected.")

        O2_ROI=detect_word_location2_p24(image_org,'LPM',40)
        if O2_ROI:
            x1,y1,x2,y2=O2_ROI[0]
            # print(x2-x1)
            if x2-x1<60:
                x1=x1-130
            elif x2-x1<155:
                x1=x1-60
            roi=x1,y1,x2,y2-4
            O2_text=extract_text_from_roi(image_org,roi)
            if '@' not in O2_text:
                x1=x1-60
                roi=x1,y1,x2,y2
                O2_text=extract_text_from_roi(image_org,roi)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # print('O2 text:',O2_text)
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
            print('cannot detect the word LPM')

        print('********************************circulation*************************************************************')

        output_image, table_circles = detect_radio_buttons2_p24(thresh_radio,output_image)

        table_filled_buttons,output_image = detect_filled_button_p24(thresh_check,table_circles,output_image,filled_radio_ratio)

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

        line1_ROI=detect_word_location_p24(image_org,'Anterior:',800)
        line2_ROI=detect_word_location_p24(image_org,'Posterior:',800)

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
            # print('line1 text:',line1.replace('\n',' ').replace('.',' ').replace('_',''))
        if line2_ROI:
            x1,y1,x2,y2=line2_ROI[0]
            if y2-y1 >38:
                y1=y1+8
            roi=x1,y1,x2,y2
            line2=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # print('line2 text:',line2.replace('\n',' ').replace('.',' ').replace('_',''))
        if line2_ROI:
            x1,y1,x2,y2=line2_ROI[0]
            roi=x1,y2+6,x2,y2+40
            line3=extract_text_from_roi(image_org,roi)

            cv2.rectangle(output_image, (x1, y2+6), (x2, y2+40), (255, 0, 255), 2)
            # print('line3 text:',line3.replace('\n',' ').replace('.',' ').replace('_',''))

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

        if show_imgs:
            Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 24p2 of this pdf!*")
    else:
        return excel_inputs

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QFileDialog, QCheckBox, QDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QScrollArea, QFormLayout, QComboBox

class ScanThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to send progress updates
    # paragraph_signal = pyqtSignal(str)
    L4_signal=pyqtSignal(str)
    O4_signal=pyqtSignal(str)
    R4_signal=pyqtSignal(str)
    done=pyqtSignal()
    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        all_pages = pdf_to_images(self.app.pdf_path)

        # Your scanning logic will be placed here.
        # For simplicity, I'm distributing the progress equally across pages.
        progress_steps = 100 / 18  
        current_progress = 0
        self.progress_signal.emit(1)

        p1_excel_inputs = page1(np.array(all_pages[0]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p4_excel_inputs = page4(np.array(all_pages[3]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p5_excel_inputs = page5(np.array(all_pages[4]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p6_excel_inputs = page6(np.array(all_pages[5]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p6p2_excel_inputs = page6p2(np.array(all_pages[5]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p9_excel_inputs = page9(np.array(all_pages[8]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p10_excel_inputs = page10(np.array(all_pages[9]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p18_excel_inputs = page18(np.array(all_pages[17]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p19_excel_inputs = page19(np.array(all_pages[18]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p22_excel_inputs = page22(np.array(all_pages[21]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p22p2_excel_inputs = page22p2(np.array(all_pages[21]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p24_excel_inputs = page24(np.array(all_pages[23]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p24p2_excel_inputs = page24p2(np.array(all_pages[23]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p25_excel_inputs = page25(np.array(all_pages[24]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        p35_excel_inputs = page35(np.array(all_pages[34]), self.app.denoising, self.app.sharpen, self.app.correcting_skew, self.app.show_imgs)
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))

        excel_inputs = {}
        if p1_excel_inputs:
            excel_inputs.update(p1_excel_inputs)
        if p4_excel_inputs:
            excel_inputs.update(p4_excel_inputs)
        if p5_excel_inputs:
            excel_inputs.update(p5_excel_inputs)
        if p6_excel_inputs:
            excel_inputs.update(p6_excel_inputs)
        if p6p2_excel_inputs:
            excel_inputs.update(p6p2_excel_inputs)
        if p9_excel_inputs:
            excel_inputs.update(p9_excel_inputs)
        if p10_excel_inputs:
            excel_inputs.update(p10_excel_inputs)
        if p18_excel_inputs:
            excel_inputs.update(p18_excel_inputs)
        if p19_excel_inputs:
            excel_inputs.update(p19_excel_inputs)
        if p22_excel_inputs:
            excel_inputs.update(p22_excel_inputs)
        if p22p2_excel_inputs:
            excel_inputs.update(p22p2_excel_inputs)
        if p24_excel_inputs:
            excel_inputs.update(p24_excel_inputs)
        if p24p2_excel_inputs:
            excel_inputs.update(p24p2_excel_inputs)
        if p25_excel_inputs:
            excel_inputs.update(p25_excel_inputs)
        if p35_excel_inputs:
            excel_inputs.update(p35_excel_inputs)

        widget_to_excel = {
 'J13': self.app.visionJ13,
 'J14': self.app.hearingJ14,
 'J20': self.app.painJ20,
 'F50': self.app.dietF50,
 'F53': self.app.incontinanceF53,
 'B62': self.app.edemaB62,
 'B65': self.app.sopB65,
 'F105': self.app.cmF105,
 'B147': self.app.edemaB147,
 'E8': self.app.line_editE8,
 'H8': self.app.line_editH8,
 'I8': self.app.line_editI8,
 'I13': self.app.line_editI13,
 'E14': self.app.line_editE14,
 'F14': self.app.line_editF14,
 'I14': self.app.line_editI14,
 'E15': self.app.line_editE15,
 'F15': self.app.line_editF15,
 'E16': self.app.line_editE16,
 'F16': self.app.line_editF16,
 'H17': self.app.line_editH17,
 'J17': self.app.line_editJ17,
 'E17': self.app.line_editE17,
 'F17': self.app.line_editF17,
 'E18': self.app.line_editE18,
 'F18': self.app.line_editF18,
 'H20': self.app.line_editH20,
 'E19': self.app.line_editE19,
 'F19': self.app.line_editF19,
 'E20': self.app.line_editE20,
 'F20': self.app.line_editF20,
 'H23': self.app.line_editH23,
 'I23': self.app.line_editI23,
 'H24': self.app.line_editH24,
 'I24': self.app.line_editI24,
 'E21': self.app.line_editE21,
 'F21': self.app.line_editF21,
 'E22': self.app.line_editE22,
 'F22': self.app.line_editF22,
 'F28': self.app.line_editF28,
 'F29': self.app.line_editF29,
 'F30': self.app.line_editF30,
 'F31': self.app.line_editF31,
 'F32': self.app.line_editF32,
 'F33': self.app.line_editF33,
 'F34': self.app.line_editF34,
 'F35': self.app.line_editF35,
 'F36': self.app.line_editF36,
 'F37': self.app.line_editF37,
 'F38': self.app.line_editF38,
 'F39': self.app.line_editF39,
 'F40': self.app.line_editF40,
 'F41': self.app.line_editF41,
 'F42': self.app.line_editF42,
 'F43': self.app.line_editF43,
 'F44': self.app.line_editF44,
 'I28': self.app.line_editI28,
 'I29': self.app.line_editI29,
 'I30': self.app.line_editI30,
 'I31': self.app.line_editI31,
 'I32': self.app.line_editI32,
 'I33': self.app.line_editI33,
 'I34': self.app.line_editI34,
 'I35': self.app.line_editI35,
 'I36': self.app.line_editI36,
 'I37': self.app.line_editI37,
 'I38': self.app.line_editI38,
 'I39': self.app.line_editI39,
 'I40': self.app.line_editI40,
 'I41': self.app.line_editI41,
 'I42': self.app.line_editI42,
 'I43': self.app.line_editI43,
 'I44': self.app.line_editI44,
 'B53': self.app.line_editB53,
 'I53': self.app.line_editI53,
 'I54': self.app.line_editI54,
 'I55': self.app.line_editI55,
 'I56': self.app.line_editI56,
 'I57': self.app.line_editI57,
 'I58': self.app.line_editI58,
 'I59': self.app.line_editI59,
 'I60': self.app.line_editI60,
 'I61': self.app.line_editI61,
 'I62': self.app.line_editI62,
 'J53': self.app.line_editJ53,
 'J54': self.app.line_editJ54,
 'J55': self.app.line_editJ55,
 'J56': self.app.line_editJ56,
 'J57': self.app.line_editJ57,
 'J58': self.app.line_editJ58,
 'J59': self.app.line_editJ59,
 'J60': self.app.line_editJ60,
 'J61': self.app.line_editJ61,
 'J62': self.app.line_editJ62,
 'F56': self.app.line_editF56,
 'F57': self.app.line_editF57,
 'F58': self.app.line_editF58,
 'F59': self.app.line_editF59,
 'F60': self.app.line_editF60,
 'F61': self.app.line_editF61,
 'G56': self.app.line_editG56,
 'G57': self.app.line_editG57,
 'G58': self.app.line_editG58,
 'G59': self.app.line_editG59,
 'G60': self.app.line_editG60,
 'G61': self.app.line_editG61,
 'C59': self.app.line_editC59,
 'B59': self.app.line_editB59,
 'B56': self.app.line_editB56,
 'F65': self.app.line_editF65,
 'I65': self.app.line_editI65,
 'C145': self.app.line_editC145,
 'D145': self.app.line_editD145,
 'E145': self.app.line_editE145,
 'F145': self.app.line_editF145,
 'G145': self.app.line_editG145,
 'C146': self.app.line_editC146,
 'D146': self.app.line_editD146,
 'E146': self.app.line_editE146,
 'F146': self.app.line_editF146,
 'G146': self.app.line_editG146,
 'C150': self.app.line_editC150,
 'C151': self.app.line_editC151,
 'C152': self.app.line_editC152,
 'D150': self.app.line_editD150,
 'D151': self.app.line_editD151,
 'D152': self.app.line_editD152,

 'C2': self.app.poc_typeC2,
 'E5': self.app.musculoskeletalE5,
 'H5': self.app.hypertensionH5,
 'A5': self.app.templateA5,
 'B8': self.app.snv_frequencyB8,
 'J8': self.app.line_editJ8,
 'J11': self.app.directiveJ11,
 'B11': self.app.pt_or_notB11,
 'B14': self.app.priority_codeB14,
 'B20': self.app.copd_asthmaB20,
 'G101': self.app.typeG101,
 'G122': self.app.SHUNTtypeG122,
 'I122': self.app.SHUNTlocaI122,
 'E11': self.app.line_editE11,
 'H11': self.app.line_editH11,
 'B69': self.app.line_editB69,
 'B26': self.app.line_editB26,
 'B70': self.app.line_editB70,
 'B27': self.app.line_editB27,
 'B71': self.app.line_editB71,
 'B28': self.app.line_editB28,
 'B72': self.app.line_editB72,
 'B29': self.app.line_editB29,
 'B17': self.app.line_editB17,
 'B73': self.app.line_editB73,
 'B30': self.app.line_editB30,
 'B74': self.app.line_editB74,
 'B31': self.app.line_editB31,
 'B75': self.app.line_editB75,
 'B32': self.app.line_editB32,
 'B76': self.app.line_editB76,
 'B33': self.app.line_editB33,
 'B23': self.app.line_editB23,
 'B77': self.app.line_editB77,
 'B34': self.app.line_editB34,
 'B78': self.app.line_editB78,
 'B35': self.app.line_editB35,
 'B79': self.app.line_editB79,
 'B36': self.app.line_editB36,
 'B80': self.app.line_editB80,
 'B37': self.app.line_editB37,
 'B81': self.app.line_editB81,
 'B38': self.app.line_editB38,
 'B82': self.app.line_editB82,
 'B39': self.app.line_editB39,
 'B83': self.app.line_editB83,
 'B40': self.app.line_editB40,
 'B84': self.app.line_editB84,
 'B41': self.app.line_editB41,
 'B85': self.app.line_editB85,
 'B42': self.app.line_editB42,
 'B86': self.app.line_editB86,
 'B43': self.app.line_editB43,
 'B87': self.app.line_editB87,
 'B44': self.app.line_editB44,
 'B88': self.app.line_editB88,
 'B45': self.app.line_editB45,
 'B89': self.app.line_editB89,
 'B46': self.app.line_editB46,
 'B47': self.app.line_editB47,
 'B48': self.app.line_editB48,
 'F48': self.app.line_editF48,
 'I65': self.app.line_editI65,
 'B49': self.app.line_editB49,
 'B50': self.app.line_editB50,
 'I70': self.app.line_editI70,
 'I74': self.app.line_editI74,
 'I78': self.app.line_editI78,
 'I82': self.app.line_editI82,
 'I83': self.app.line_editI83,
 'I87': self.app.line_editI87,
 'I88': self.app.line_editI88,
 'I93': self.app.line_editI93,
 'J93': self.app.line_editJ93,
 'I94': self.app.line_editI94,
 'J94': self.app.line_editJ94,
 'B101': self.app.line_editB101,
 'E107': self.app.line_editE107,
 'F107': self.app.line_editF107,
 'G107': self.app.line_editG107,
 'H107': self.app.line_editH107,
 'I107': self.app.line_editI107,
 'J107': self.app.line_editJ107,
 'E108': self.app.line_editE108,
 'F108': self.app.line_editF108,
 'G108': self.app.line_editG108,
 'H108': self.app.line_editH108,
 'I108': self.app.line_editI108,
 'J108': self.app.line_editJ108,
 'E109': self.app.line_editE109,
 'F109': self.app.line_editF109,
 'G109': self.app.line_editG109,
 'H109': self.app.line_editH109,
 'I109': self.app.line_editI109,
 'J109': self.app.line_editJ109,
 'E110': self.app.line_editE110,
 'F110': self.app.line_editF110,
 'G110': self.app.line_editG110,
 'H110': self.app.line_editH110,
 'I110': self.app.line_editI110,
 'J110': self.app.line_editJ110,
 'E111': self.app.line_editE111,
 'F111': self.app.line_editF111,
 'G111': self.app.line_editG111,
 'H111': self.app.line_editH111,
 'I111': self.app.line_editI111,
 'J111': self.app.line_editJ111,
 'E115': self.app.line_editE115,
 'F115': self.app.line_editF115,
 'G115': self.app.line_editG115,
 'H115': self.app.line_editH115,
 'I115': self.app.line_editI115,
 'E116': self.app.line_editE116,
 'F116': self.app.line_editF116,
 'G116': self.app.line_editG116,
 'H116': self.app.line_editH116,
 'I116': self.app.line_editI116,
 'E118': self.app.line_editE118,
 'E122': self.app.line_editE122,
 'E123': self.app.line_editE123,
 'E127': self.app.line_editE127,
 'E128': self.app.line_editE128,
 'E129': self.app.line_editE129,
 'E130': self.app.line_editE130,
 'E134': self.app.line_editE134,
 'E135': self.app.line_editE135,
 'E136': self.app.line_editE136,
 'E137': self.app.line_editE137,
 'E138': self.app.line_editE138,
 'E139': self.app.line_editE139,
 'E140': self.app.line_editE140}


        start_time = time.time()
        print ('opening the exel sheet...')
        app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
        wb = app.books.open(self.app.excel_path)
        ws = wb.sheets['Basic- Auto']
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))
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
        if found_r600 :
            excel_inputs['B147'] = 'localized'
        else:
            excel_inputs['B147']= 'Normal'


        for cell in excel_inputs:
            ws.range(cell).value = excel_inputs[cell]

        # Save and close
        wb.save()
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))
        # Iterate over the mapping
        for cell, widget in widget_to_excel.items():
            value = ws.range(cell).value
            if isinstance(widget, QComboBox):
                # Find the index of the value in the combo box and set it
                index = widget.findText(str(value))
                if index != -1:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            else:
                pass


        value_from_L4 = ws.range('L4').value
        value_from_O4 = ws.range('O4').value
        value_from_R4 = ws.range('R4').value

        wb.close()
        app.quit()
        current_progress += progress_steps
        self.progress_signal.emit(int(current_progress))
        end_time = time.time()
        print ('closing the exel sheet...')
        execution_time = end_time - start_time
        print(f"Time of Updating the Excel sheet: {execution_time:.2f} seconds")
        print(f"Updated Excel sheet '{self.app.excel_path}' with :")
        for cell in excel_inputs:    
            print(f"value '{excel_inputs[cell]}' in cell '{cell}'.")


        self.L4_signal.emit(value_from_L4)
        self.O4_signal.emit(value_from_O4)
        self.R4_signal.emit(value_from_R4)
        current_progress = 0
        self.progress_signal.emit(int(current_progress))
        self.done.emit()

class RestartThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to send progress updates
    L4_signal=pyqtSignal(str)
    O4_signal=pyqtSignal(str)
    R4_signal=pyqtSignal(str)
    done=pyqtSignal()
    # paragraph_signal = pyqtSignal(str)
    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):

        widget_to_excel = {
 'J13': self.app.visionJ13,
 'J14': self.app.hearingJ14,
 'J20': self.app.painJ20,
 'F50': self.app.dietF50,
 'F53': self.app.incontinanceF53,
 'B62': self.app.edemaB62,
 'B65': self.app.sopB65,
 'F105': self.app.cmF105,
 'B147': self.app.edemaB147,
 'E8': self.app.line_editE8,
 'H8': self.app.line_editH8,
 'I8': self.app.line_editI8,
 'I13': self.app.line_editI13,
 'E14': self.app.line_editE14,
 'F14': self.app.line_editF14,
 'I14': self.app.line_editI14,
 'E15': self.app.line_editE15,
 'F15': self.app.line_editF15,
 'E16': self.app.line_editE16,
 'F16': self.app.line_editF16,
 'H17': self.app.line_editH17,
 'J17': self.app.line_editJ17,
 'E17': self.app.line_editE17,
 'F17': self.app.line_editF17,
 'E18': self.app.line_editE18,
 'F18': self.app.line_editF18,
 'H20': self.app.line_editH20,
 'E19': self.app.line_editE19,
 'F19': self.app.line_editF19,
 'E20': self.app.line_editE20,
 'F20': self.app.line_editF20,
 'H23': self.app.line_editH23,
 'I23': self.app.line_editI23,
 'H24': self.app.line_editH24,
 'I24': self.app.line_editI24,
 'E21': self.app.line_editE21,
 'F21': self.app.line_editF21,
 'E22': self.app.line_editE22,
 'F22': self.app.line_editF22,
 'F28': self.app.line_editF28,
 'F29': self.app.line_editF29,
 'F30': self.app.line_editF30,
 'F31': self.app.line_editF31,
 'F32': self.app.line_editF32,
 'F33': self.app.line_editF33,
 'F34': self.app.line_editF34,
 'F35': self.app.line_editF35,
 'F36': self.app.line_editF36,
 'F37': self.app.line_editF37,
 'F38': self.app.line_editF38,
 'F39': self.app.line_editF39,
 'F40': self.app.line_editF40,
 'F41': self.app.line_editF41,
 'F42': self.app.line_editF42,
 'F43': self.app.line_editF43,
 'F44': self.app.line_editF44,
 'I28': self.app.line_editI28,
 'I29': self.app.line_editI29,
 'I30': self.app.line_editI30,
 'I31': self.app.line_editI31,
 'I32': self.app.line_editI32,
 'I33': self.app.line_editI33,
 'I34': self.app.line_editI34,
 'I35': self.app.line_editI35,
 'I36': self.app.line_editI36,
 'I37': self.app.line_editI37,
 'I38': self.app.line_editI38,
 'I39': self.app.line_editI39,
 'I40': self.app.line_editI40,
 'I41': self.app.line_editI41,
 'I42': self.app.line_editI42,
 'I43': self.app.line_editI43,
 'I44': self.app.line_editI44,
 'B53': self.app.line_editB53,
 'I53': self.app.line_editI53,
 'I54': self.app.line_editI54,
 'I55': self.app.line_editI55,
 'I56': self.app.line_editI56,
 'I57': self.app.line_editI57,
 'I58': self.app.line_editI58,
 'I59': self.app.line_editI59,
 'I60': self.app.line_editI60,
 'I61': self.app.line_editI61,
 'I62': self.app.line_editI62,
 'J53': self.app.line_editJ53,
 'J54': self.app.line_editJ54,
 'J55': self.app.line_editJ55,
 'J56': self.app.line_editJ56,
 'J57': self.app.line_editJ57,
 'J58': self.app.line_editJ58,
 'J59': self.app.line_editJ59,
 'J60': self.app.line_editJ60,
 'J61': self.app.line_editJ61,
 'J62': self.app.line_editJ62,
 'F56': self.app.line_editF56,
 'F57': self.app.line_editF57,
 'F58': self.app.line_editF58,
 'F59': self.app.line_editF59,
 'F60': self.app.line_editF60,
 'F61': self.app.line_editF61,
 'G56': self.app.line_editG56,
 'G57': self.app.line_editG57,
 'G58': self.app.line_editG58,
 'G59': self.app.line_editG59,
 'G60': self.app.line_editG60,
 'G61': self.app.line_editG61,
 'C59': self.app.line_editC59,
 'B59': self.app.line_editB59,
 'B56': self.app.line_editB56,
 'F65': self.app.line_editF65,
 'I65': self.app.line_editI65,
 'C145': self.app.line_editC145,
 'D145': self.app.line_editD145,
 'E145': self.app.line_editE145,
 'F145': self.app.line_editF145,
 'G145': self.app.line_editG145,
 'C146': self.app.line_editC146,
 'D146': self.app.line_editD146,
 'E146': self.app.line_editE146,
 'F146': self.app.line_editF146,
 'G146': self.app.line_editG146,
 'C150': self.app.line_editC150,
 'C151': self.app.line_editC151,
 'C152': self.app.line_editC152,
 'D150': self.app.line_editD150,
 'D151': self.app.line_editD151,
 'D152': self.app.line_editD152,

 'C2': self.app.poc_typeC2,
 'E5': self.app.musculoskeletalE5,
 'H5': self.app.hypertensionH5,
 'A5': self.app.templateA5,
 'B8': self.app.snv_frequencyB8,
 'J8': self.app.line_editJ8,
 'J11': self.app.directiveJ11,
 'B11': self.app.pt_or_notB11,
 'B14': self.app.priority_codeB14,
 'B20': self.app.copd_asthmaB20,
 'G101': self.app.typeG101,
 'G122': self.app.SHUNTtypeG122,
 'I122': self.app.SHUNTlocaI122,
 'E11': self.app.line_editE11,
 'H11': self.app.line_editH11,
 'B69': self.app.line_editB69,
 'B26': self.app.line_editB26,
 'B70': self.app.line_editB70,
 'B27': self.app.line_editB27,
 'B71': self.app.line_editB71,
 'B28': self.app.line_editB28,
 'B72': self.app.line_editB72,
 'B29': self.app.line_editB29,
 'B17': self.app.line_editB17,
 'B73': self.app.line_editB73,
 'B30': self.app.line_editB30,
 'B74': self.app.line_editB74,
 'B31': self.app.line_editB31,
 'B75': self.app.line_editB75,
 'B32': self.app.line_editB32,
 'B76': self.app.line_editB76,
 'B33': self.app.line_editB33,
 'B23': self.app.line_editB23,
 'B77': self.app.line_editB77,
 'B34': self.app.line_editB34,
 'B78': self.app.line_editB78,
 'B35': self.app.line_editB35,
 'B79': self.app.line_editB79,
 'B36': self.app.line_editB36,
 'B80': self.app.line_editB80,
 'B37': self.app.line_editB37,
 'B81': self.app.line_editB81,
 'B38': self.app.line_editB38,
 'B82': self.app.line_editB82,
 'B39': self.app.line_editB39,
 'B83': self.app.line_editB83,
 'B40': self.app.line_editB40,
 'B84': self.app.line_editB84,
 'B41': self.app.line_editB41,
 'B85': self.app.line_editB85,
 'B42': self.app.line_editB42,
 'B86': self.app.line_editB86,
 'B43': self.app.line_editB43,
 'B87': self.app.line_editB87,
 'B44': self.app.line_editB44,
 'B88': self.app.line_editB88,
 'B45': self.app.line_editB45,
 'B89': self.app.line_editB89,
 'B46': self.app.line_editB46,
 'B47': self.app.line_editB47,
 'B48': self.app.line_editB48,
 'F48': self.app.line_editF48,
 'I65': self.app.line_editI65,
 'B49': self.app.line_editB49,
 'B50': self.app.line_editB50,
 'I70': self.app.line_editI70,
 'I74': self.app.line_editI74,
 'I78': self.app.line_editI78,
 'I82': self.app.line_editI82,
 'I83': self.app.line_editI83,
 'I87': self.app.line_editI87,
 'I88': self.app.line_editI88,
 'I93': self.app.line_editI93,
 'J93': self.app.line_editJ93,
 'I94': self.app.line_editI94,
 'J94': self.app.line_editJ94,
 'B101': self.app.line_editB101,
 'E107': self.app.line_editE107,
 'F107': self.app.line_editF107,
 'G107': self.app.line_editG107,
 'H107': self.app.line_editH107,
 'I107': self.app.line_editI107,
 'J107': self.app.line_editJ107,
 'E108': self.app.line_editE108,
 'F108': self.app.line_editF108,
 'G108': self.app.line_editG108,
 'H108': self.app.line_editH108,
 'I108': self.app.line_editI108,
 'J108': self.app.line_editJ108,
 'E109': self.app.line_editE109,
 'F109': self.app.line_editF109,
 'G109': self.app.line_editG109,
 'H109': self.app.line_editH109,
 'I109': self.app.line_editI109,
 'J109': self.app.line_editJ109,
 'E110': self.app.line_editE110,
 'F110': self.app.line_editF110,
 'G110': self.app.line_editG110,
 'H110': self.app.line_editH110,
 'I110': self.app.line_editI110,
 'J110': self.app.line_editJ110,
 'E111': self.app.line_editE111,
 'F111': self.app.line_editF111,
 'G111': self.app.line_editG111,
 'H111': self.app.line_editH111,
 'I111': self.app.line_editI111,
 'J111': self.app.line_editJ111,
 'E115': self.app.line_editE115,
 'F115': self.app.line_editF115,
 'G115': self.app.line_editG115,
 'H115': self.app.line_editH115,
 'I115': self.app.line_editI115,
 'E116': self.app.line_editE116,
 'F116': self.app.line_editF116,
 'G116': self.app.line_editG116,
 'H116': self.app.line_editH116,
 'I116': self.app.line_editI116,
 'E118': self.app.line_editE118,
 'E122': self.app.line_editE122,
 'E123': self.app.line_editE123,
 'E127': self.app.line_editE127,
 'E128': self.app.line_editE128,
 'E129': self.app.line_editE129,
 'E130': self.app.line_editE130,
 'E134': self.app.line_editE134,
 'E135': self.app.line_editE135,
 'E136': self.app.line_editE136,
 'E137': self.app.line_editE137,
 'E138': self.app.line_editE138,
 'E139': self.app.line_editE139,
 'E140': self.app.line_editE140}


        self.progress_signal.emit(4)
        excel_inputs={
        'C2': 'SOC',
        'I33': None,
        'I43': None,
        'I44': None,
        'I32': None,
        'G56': None,
        'G57': None,
        'G58': None,
        'G59': None,
        'G60': None,
        'G61': None,
        'I30': None,
        'F65': 6,
        'F53': 'Incontinent',
        'F14': None,
        'F15': None,
        'F16': None,
        'F17': None,
        'J17': None,
        'F18': None,
        'F19': None,
        'F20': None,
        'J20': 'DAILY',
        
        'F21': None,
        'F22': None,
        'H20': None,
        'B65': 'Moderate',
        'I8': 65,
        'H8': None,
        'J55': None,
        'J53': None,
        'J54': None,
        'J57': None,
        'J56': None,
        'J59': None,
        'J58': None,
        'B14': 2,
        'E140': None,
        'I28': None,
        'I29': None,
        'E8': None,
        'E136': None,
        'B23': None,
        'J14': 'MINIMAL',
        'I14': None,
        'J13': 'IMPAIRED VISION',
        'I13': None,
        'H17': None,
        'I31': None,
        'B56': None,
        'C145': None,
        'D145': None,
        'E145': None,
        'F145': None,
        'G145': None,
        'C146': None,
        'D146': None,
        'E146': None,
        'F146': None,
        'G146': None,
        'B147': 'Normal',
        'C150': None,
        'C151': None,
        'C152': None,
        'D150': None,
        'D151': None,
        'D152': None,
        'B59': None,
        'C59': None,
        'F48': 'NAS, renal, controlled carbohydrate, ncs, low fat/cholesterol, high fiber, and nas, controlled carb, ncs, renal, high fiber diet',
        'A5': 'BASIC',
        'E5': '$$$Musculoskeletal',
        'H5': '$$$Hypertensive heart',
        'B8': '2wx2, 1wx7',
        'B11': 'NONE',
        'E11': None,
        'H11': None,
        'J11': 'DNR',
        'B17': None,
        'B20': 'COPD',
        'E14': 1,
        'E15': 2,
        'E16': 3,
        'E17': 4,
        'E18': 5,
        'E19': 6,
        'E20': 7,
        'E21': 8,
        'E22': 9,
        'H23': 'ROM',
        'H24': 'WEAK',
        'I23': None,
        'I24': None,
        'B26': 'NONE',
        'B27': 'NONE',
        'B28': 'NONE',
        'B29': 'NONE',
        # 'B30': 'NONE',
        # 'B31': 'NONE',
        # 'B32': 'NONE',
        # 'B33': 'NONE',
        # 'B34': 'NONE',
        # 'B35': 'NONE',
        # 'B36': 'NONE',
        # 'B37': 'NONE',
        # 'B38': 'NONE',
        # 'B39': 'NONE',
        # 'B40': 'NONE',
        # 'B41': 'NONE',
        # 'B42': 'NONE',
        # 'B43': 'NONE',
        'B44': 'NONE',
        'B45': 'NONE',
        'B46': 'NONE',
        'B47': 'NONE',
        'B48': 'NONE',
        'B49': 'NONE',
        'B50': 'NONE',
        'F28': 'LANGUAGE',
        'F29': 'LANGUAGE BAR.',
        'F30': 'LEARNING BAR',
        'F31': 'EDUCATION',
        'F32': 'SPIRITUAL RESOURCE',
        'F33': 'MARITAL STATUS',
        'F34': '# OF CHILDREN',
        'F35': 'GENDER OF CHILDREN',
        'F36': 'CHILDREN NEARBY YES OR NO?',
        'F37': 'DRIVE YES OR NO?',
        'F38': 'JOB YES OR NO?',
        'F39': 'SLEEP',
        'F40': 'REST',
        'F41': 'HOURS OF SLEEP',
        'F42': ' REST',
        'F43': 'FEELINGS/EMOTIONS',
        'F44': 'INABILITY TO COPE WITH',
        'I34': None,
        'I35': None,
        'I36': None,
        'I37': None,
        'I38': None,
        'I39': 'NONE',
        'I40': 'NONE',
        'I41': None,
        'I42': None,
        'F50': 'NAS diet',
        'B53': None,
        'B62': None,
        'F56': 'Memory',
        'F57': 'Impaired',
        'F58': 'Verbal dis',
        'F59': 'Physical a',
        'F60': 'Disruptive',
        'F61': 'Delusiona',
        'J60': None,
        'J61': None,
        'I62': None,
        'J62': None,
        'I53': 'Cane',
        'I54': 'Walker',
        'I55': 'Bathbenc',
        'I56': 'Grab Bars',
        'I57': 'Wheelcha',
        'I58': 'Commod',
        'I59': 'Bed Hospi',
        'I60': 'IM Syringe',
        'I61': 'Oth. SQ Syringe',
        'I65': None,
        'B69': None,
        'B70': None,
        'B71': None,
        'B72': None,
        'B73': None,
        'B74': None,
        'B75': None,
        'B76': None,
        'B77': None,
        'B78': None,
        'B79': None,
        'B80': None,
        'B81': None,
        'B82': None,
        'B83': None,
        'B84': None,
        'B85': None,
        'B86': None,
        'B87': None,
        'B88': None,
        'B89': None,
        'I70': None,
        'I74': None,
        'I78': None,
        'I82': None,
        'I83': None,
        'I87': None,
        'I88': None,
        'I93': None,
        'J93': None,
        'I94': None,
        'J94': None,
        'E97': 'ADD O2 SUPPLIES FOR LIFE STAR',
        'B101': None,
        'G101': 'Syringe',
        'B106': 'WOUND #',
        'B107': 'WOUND LOCATION',
        'B108': 'WOUND SIZE',
        'B109': 'HEIGHT',
        'B110': 'LENGTH',
        'B111': 'WIDTH',
        
        'F105': 'cm',
        'E106': '#1',
        'F106': '#2',
        'G106': '#3',
        'H106': '#4',
        'I106': '#5',
        'J106': '#6',
        'E107': None,
        'F107': None,
        'G107': None,
        'H107': None,
        'I107': None,
        'J107': None,
        'E109': None,
        'F109': None,
        'G109': None,
        'H109': None,
        'I109': None,
        'J109': None,
        'E110': None,
        'F110': None,
        'G110': None,
        'H110': None,
        'I110': None,
        'J110': None,
        'E111': None,
        'F111': None,
        'G111': None,
        'H111': None,
        'I111': None,
        'J111': None,
        'E114': 'SITE#1',
        'F114': 'SITE#2',
        'G114': 'SITE#3',
        'H114': 'SITE#4',
        'I114': 'SITE#5',
        'E115': None,
        'F115': None,
        'G115': None,
        'H115': None,
        'I115': None,
        'E116': None,
        'F116': None,
        'G116': None,
        'H116': None,
        'I116': None,
        'E118': None,
        'E122': None,
        'G122': 'SHUNT',
        'I122': 'left',
        'E123': None,
        'E127': 'right upper arm double lumen midline',
        'E128': 'Ertapenem 1gm/50mL normal saline, administer over 30min via RUE midline for 10 days (start date 09/26/2023, end date 10/05/2023). Flush with 5-10mL Normal Saline 0.9% before and after each use. Flush with 3-5mL Heparin 100units/mL 5mL syring after each use',
        'E129': 'Ertapenem 1gm/50mL',
        'E130': None,
        'E134': None,
        'E135': None,
        'E137': None,
        'E138': None,
        'E139': None}
        start_time = time.time()
        print ('opening the exel sheet...')
        app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
        wb = app.books.open(self.app.excel_path)
        self.progress_signal.emit(24)
        ws = wb.sheets['Basic- Auto']
        for cell in excel_inputs:
            ws.range(cell).value = excel_inputs[cell]
        self.progress_signal.emit(53)
        # Save and close
        wb.save()

        # Iterate over the mapping
        for cell, widget in widget_to_excel.items():
            value = ws.range(cell).value
            if isinstance(widget, QComboBox):
                # Find the index of the value in the combo box and set it
                index = widget.findText(str(value))
                if index != -1:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            else:
                pass

        value_from_L4 = ws.range('L4').value
        value_from_O4 = ws.range('O4').value
        value_from_R4 = ws.range('R4').value
        self.L4_signal.emit(value_from_L4)
        self.O4_signal.emit(value_from_O4)
        self.R4_signal.emit(value_from_R4)
        self.progress_signal.emit(64)
        wb.close()
        self.progress_signal.emit(78)
        app.quit()
        self.progress_signal.emit(89)
        end_time = time.time()
        print ('closing the exel sheet...')
        execution_time = end_time - start_time
        print(f"Time of Updating the Excel sheet: {execution_time:.2f} seconds")
        print(f"Updated Excel sheet '{self.app.excel_path}' with :")
        for cell in excel_inputs:    
            print(f"value '{excel_inputs[cell]}' in cell '{cell}'.")
        self.progress_signal.emit(100)
        self.progress_signal.emit(0)
        self.done.emit()

class UpdateThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to send progress updates
    L4_signal=pyqtSignal(str)
    O4_signal=pyqtSignal(str)
    R4_signal=pyqtSignal(str)
    done=pyqtSignal()
    # paragraph_signal = pyqtSignal(str)
    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
            #dict hadi na9sa 3la lokhrin na9sin menha li fihom formule bah ki dir update matkhasarhach
        widget_to_excel = {
 'J13': self.app.visionJ13,
 'J14': self.app.hearingJ14,
 'J20': self.app.painJ20,
 'F50': self.app.dietF50,
 'F53': self.app.incontinanceF53,
 'B62': self.app.edemaB62,
 'B65': self.app.sopB65,
 'F105': self.app.cmF105,
 'B147': self.app.edemaB147,
 'E8': self.app.line_editE8,
 'H8': self.app.line_editH8,
 'I8': self.app.line_editI8,
 'I13': self.app.line_editI13,
 'E14': self.app.line_editE14,
 'F14': self.app.line_editF14,
 'I14': self.app.line_editI14,
 'E15': self.app.line_editE15,
 'F15': self.app.line_editF15,
 'E16': self.app.line_editE16,
 'F16': self.app.line_editF16,
 'H17': self.app.line_editH17,
 'J17': self.app.line_editJ17,
 'E17': self.app.line_editE17,
 'F17': self.app.line_editF17,
 'E18': self.app.line_editE18,
 'F18': self.app.line_editF18,
 'H20': self.app.line_editH20,
 'E19': self.app.line_editE19,
 'F19': self.app.line_editF19,
 'E20': self.app.line_editE20,
 'F20': self.app.line_editF20,
 'H23': self.app.line_editH23,
 'I23': self.app.line_editI23,
 'H24': self.app.line_editH24,
 'I24': self.app.line_editI24,
 'E21': self.app.line_editE21,
 'F21': self.app.line_editF21,
 'E22': self.app.line_editE22,
 'F22': self.app.line_editF22,
 'F28': self.app.line_editF28,
 'F29': self.app.line_editF29,
 'F30': self.app.line_editF30,
 'F31': self.app.line_editF31,
 'F32': self.app.line_editF32,
 'F33': self.app.line_editF33,
 'F34': self.app.line_editF34,
 'F35': self.app.line_editF35,
 'F36': self.app.line_editF36,
 'F37': self.app.line_editF37,
 'F38': self.app.line_editF38,
 'F39': self.app.line_editF39,
 'F40': self.app.line_editF40,
 'F41': self.app.line_editF41,
 'F42': self.app.line_editF42,
 'F43': self.app.line_editF43,
 'F44': self.app.line_editF44,
 'I28': self.app.line_editI28,
 'I29': self.app.line_editI29,
 'I30': self.app.line_editI30,
 'I31': self.app.line_editI31,
 'I32': self.app.line_editI32,
 'I33': self.app.line_editI33,
 'I34': self.app.line_editI34,
 'I35': self.app.line_editI35,
 'I36': self.app.line_editI36,
 'I37': self.app.line_editI37,
 'I38': self.app.line_editI38,
 'I39': self.app.line_editI39,
 'I40': self.app.line_editI40,
 'I41': self.app.line_editI41,
 'I42': self.app.line_editI42,
 'I43': self.app.line_editI43,
 'I44': self.app.line_editI44,
 'B53': self.app.line_editB53,
 'I53': self.app.line_editI53,
 'I54': self.app.line_editI54,
 'I55': self.app.line_editI55,
 'I56': self.app.line_editI56,
 'I57': self.app.line_editI57,
 'I58': self.app.line_editI58,
 'I59': self.app.line_editI59,
 'I60': self.app.line_editI60,
 'I61': self.app.line_editI61,
 'I62': self.app.line_editI62,
 'J53': self.app.line_editJ53,
 'J54': self.app.line_editJ54,
 'J55': self.app.line_editJ55,
 'J56': self.app.line_editJ56,
 'J57': self.app.line_editJ57,
 'J58': self.app.line_editJ58,
 'J59': self.app.line_editJ59,
 'J60': self.app.line_editJ60,
 'J61': self.app.line_editJ61,
 'J62': self.app.line_editJ62,
 'F56': self.app.line_editF56,
 'F57': self.app.line_editF57,
 'F58': self.app.line_editF58,
 'F59': self.app.line_editF59,
 'F60': self.app.line_editF60,
 'F61': self.app.line_editF61,
 'G56': self.app.line_editG56,
 'G57': self.app.line_editG57,
 'G58': self.app.line_editG58,
 'G59': self.app.line_editG59,
 'G60': self.app.line_editG60,
 'G61': self.app.line_editG61,
 'C59': self.app.line_editC59,
 'B59': self.app.line_editB59,
 'B56': self.app.line_editB56,
 'F65': self.app.line_editF65,
 'I65': self.app.line_editI65,
 'C145': self.app.line_editC145,
 'D145': self.app.line_editD145,
 'E145': self.app.line_editE145,
 'F145': self.app.line_editF145,
 'G145': self.app.line_editG145,
 'C146': self.app.line_editC146,
 'D146': self.app.line_editD146,
 'E146': self.app.line_editE146,
 'F146': self.app.line_editF146,
 'G146': self.app.line_editG146,
 'C150': self.app.line_editC150,
 'C151': self.app.line_editC151,
 'C152': self.app.line_editC152,
 'D150': self.app.line_editD150,
 'D151': self.app.line_editD151,
 'D152': self.app.line_editD152,

 'C2': self.app.poc_typeC2,
 'E5': self.app.musculoskeletalE5,
 'H5': self.app.hypertensionH5,
 'A5': self.app.templateA5,
 'B8': self.app.snv_frequencyB8,
#  'J8': self.app.line_editJ8,
 'J11': self.app.directiveJ11,
 'B11': self.app.pt_or_notB11,
 'B14': self.app.priority_codeB14,
 'B20': self.app.copd_asthmaB20,
 'G101': self.app.typeG101,
 'G122': self.app.SHUNTtypeG122,
 'I122': self.app.SHUNTlocaI122,
 'E11': self.app.line_editE11,
 'H11': self.app.line_editH11,
 'B69': self.app.line_editB69,
 'B26': self.app.line_editB26,
 'B70': self.app.line_editB70,
 'B27': self.app.line_editB27,
 'B71': self.app.line_editB71,
 'B28': self.app.line_editB28,
 'B72': self.app.line_editB72,
 'B29': self.app.line_editB29,
 'B17': self.app.line_editB17,
 'B73': self.app.line_editB73,
#  'B30': self.app.line_editB30,
 'B74': self.app.line_editB74,
#  'B31': self.app.line_editB31,
 'B75': self.app.line_editB75,
#  'B32': self.app.line_editB32,
 'B76': self.app.line_editB76,
#  'B33': self.app.line_editB33,
 'B23': self.app.line_editB23,
 'B77': self.app.line_editB77,
#  'B34': self.app.line_editB34,
 'B78': self.app.line_editB78,
#  'B35': self.app.line_editB35,
 'B79': self.app.line_editB79,
#  'B36': self.app.line_editB36,
 'B80': self.app.line_editB80,
#  'B37': self.app.line_editB37,
 'B81': self.app.line_editB81,
#  'B38': self.app.line_editB38,
 'B82': self.app.line_editB82,
#  'B39': self.app.line_editB39,
 'B83': self.app.line_editB83,
#  'B40': self.app.line_editB40,
 'B84': self.app.line_editB84,
#  'B41': self.app.line_editB41,
 'B85': self.app.line_editB85,
#  'B42': self.app.line_editB42,
 'B86': self.app.line_editB86,
#  'B43': self.app.line_editB43,
 'B87': self.app.line_editB87,
 'B44': self.app.line_editB44,
 'B88': self.app.line_editB88,
 'B45': self.app.line_editB45,
 'B89': self.app.line_editB89,
 'B46': self.app.line_editB46,
 'B47': self.app.line_editB47,
 'B48': self.app.line_editB48,
 'F48': self.app.line_editF48,
 'I65': self.app.line_editI65,
 'B49': self.app.line_editB49,
 'B50': self.app.line_editB50,
 'I70': self.app.line_editI70,
 'I74': self.app.line_editI74,
 'I78': self.app.line_editI78,
 'I82': self.app.line_editI82,
 'I83': self.app.line_editI83,
 'I87': self.app.line_editI87,
 'I88': self.app.line_editI88,
 'I93': self.app.line_editI93,
 'J93': self.app.line_editJ93,
 'I94': self.app.line_editI94,
 'J94': self.app.line_editJ94,
 'B101': self.app.line_editB101,
 'E107': self.app.line_editE107,
 'F107': self.app.line_editF107,
 'G107': self.app.line_editG107,
 'H107': self.app.line_editH107,
 'I107': self.app.line_editI107,
 'J107': self.app.line_editJ107,
#  'E108': self.app.line_editE108,
#  'F108': self.app.line_editF108,
#  'G108': self.app.line_editG108,
#  'H108': self.app.line_editH108,
#  'I108': self.app.line_editI108,
#  'J108': self.app.line_editJ108,
 'E109': self.app.line_editE109,
 'F109': self.app.line_editF109,
 'G109': self.app.line_editG109,
 'H109': self.app.line_editH109,
 'I109': self.app.line_editI109,
 'J109': self.app.line_editJ109,
 'E110': self.app.line_editE110,
 'F110': self.app.line_editF110,
 'G110': self.app.line_editG110,
 'H110': self.app.line_editH110,
 'I110': self.app.line_editI110,
 'J110': self.app.line_editJ110,
 'E111': self.app.line_editE111,
 'F111': self.app.line_editF111,
 'G111': self.app.line_editG111,
 'H111': self.app.line_editH111,
 'I111': self.app.line_editI111,
 'J111': self.app.line_editJ111,
 'E115': self.app.line_editE115,
 'F115': self.app.line_editF115,
 'G115': self.app.line_editG115,
 'H115': self.app.line_editH115,
 'I115': self.app.line_editI115,
 'E116': self.app.line_editE116,
 'F116': self.app.line_editF116,
 'G116': self.app.line_editG116,
 'H116': self.app.line_editH116,
 'I116': self.app.line_editI116,
 'E118': self.app.line_editE118,
 'E122': self.app.line_editE122,
 'E123': self.app.line_editE123,
 'E127': self.app.line_editE127,
 'E128': self.app.line_editE128,
 'E129': self.app.line_editE129,
 'E130': self.app.line_editE130,
 'E134': self.app.line_editE134,
 'E135': self.app.line_editE135,
 'E136': self.app.line_editE136,
 'E137': self.app.line_editE137,
 'E138': self.app.line_editE138,
 'E139': self.app.line_editE139,
 'E140': self.app.line_editE140}
        self.progress_signal.emit(4)
        excel_inputs={
        'C2': 'SOC',
        'I33': None,
        'I43': None,
        'I44': None,
        'I32': None,
        'G56': None,
        'G57': None,
        'G58': None,
        'G59': None,
        'G60': None,
        'G61': None,
        'I30': None,
        'F65': 6,
        'F53': 'Incontinent',
        'F14': None,
        'F15': None,
        'F16': None,
        'F17': None,
        'J17': None,
        'F18': None,
        'F19': None,
        'F20': None,
        'J20': 'DAILY',
        
        'F21': None,
        'F22': None,
        'H20': None,
        'B65': 'Moderate',
        'I8': 65,
        'H8': None,
        'J55': None,
        'J53': None,
        'J54': None,
        'J57': None,
        'J56': None,
        'J59': None,
        'J58': None,
        'B14': 2,
        'E140': None,
        'I28': None,
        'I29': None,
        'E8': None,
        'E136': None,
        'B23': None,
        'J14': 'MINIMAL',
        'I14': None,
        'J13': 'IMPAIRED VISION',
        'I13': None,
        'H17': None,
        'I31': None,
        'B56': None,
        'C145': None,
        'D145': None,
        'E145': None,
        'F145': None,
        'G145': None,
        'C146': None,
        'D146': None,
        'E146': None,
        'F146': None,
        'G146': None,
        'B147': 'Normal',
        'C150': None,
        'C151': None,
        'C152': None,
        'D150': None,
        'D151': None,
        'D152': None,
        'B59': None,
        'C59': None,
        'F48': 'NAS, renal, controlled carbohydrate, ncs, low fat/cholesterol, high fiber, and nas, controlled carb, ncs, renal, high fiber diet',
        'A5': 'BASIC',
        'E5': '$$$Musculoskeletal',
        'H5': '$$$Hypertensive heart',
        'B8': '2wx2, 1wx7',
        'B11': 'NONE',
        'E11': None,
        'H11': None,
        'J11': 'DNR',
        'B17': None,
        'B20': 'COPD',
        'E14': 1,
        'E15': 2,
        'E16': 3,
        'E17': 4,
        'E18': 5,
        'E19': 6,
        'E20': 7,
        'E21': 8,
        'E22': 9,
        'H23': 'ROM',
        'H24': 'WEAK',
        'I23': None,
        'I24': None,
        'B26': 'NONE',
        'B27': 'NONE',
        'B28': 'NONE',
        'B29': 'NONE',
        # 'B30': 'NONE',
        # 'B31': 'NONE',
        # 'B32': 'NONE',
        # 'B33': 'NONE',
        # 'B34': 'NONE',
        # 'B35': 'NONE',
        # 'B36': 'NONE',
        # 'B37': 'NONE',
        # 'B38': 'NONE',
        # 'B39': 'NONE',
        # 'B40': 'NONE',
        # 'B41': 'NONE',
        # 'B42': 'NONE',
        # 'B43': 'NONE',
        'B44': 'NONE',
        'B45': 'NONE',
        'B46': 'NONE',
        'B47': 'NONE',
        'B48': 'NONE',
        'B49': 'NONE',
        'B50': 'NONE',
        'F28': 'LANGUAGE',
        'F29': 'LANGUAGE BAR.',
        'F30': 'LEARNING BAR',
        'F31': 'EDUCATION',
        'F32': 'SPIRITUAL RESOURCE',
        'F33': 'MARITAL STATUS',
        'F34': '# OF CHILDREN',
        'F35': 'GENDER OF CHILDREN',
        'F36': 'CHILDREN NEARBY YES OR NO?',
        'F37': 'DRIVE YES OR NO?',
        'F38': 'JOB YES OR NO?',
        'F39': 'SLEEP',
        'F40': 'REST',
        'F41': 'HOURS OF SLEEP',
        'F42': ' REST',
        'F43': 'FEELINGS/EMOTIONS',
        'F44': 'INABILITY TO COPE WITH',
        'I34': None,
        'I35': None,
        'I36': None,
        'I37': None,
        'I38': None,
        'I39': 'NONE',
        'I40': 'NONE',
        'I41': None,
        'I42': None,
        'F50': 'NAS diet',
        'B53': None,
        'B62': None,
        'F56': 'Memory',
        'F57': 'Impaired',
        'F58': 'Verbal dis',
        'F59': 'Physical a',
        'F60': 'Disruptive',
        'F61': 'Delusiona',
        'J60': None,
        'J61': None,
        'I62': None,
        'J62': None,
        'I53': 'Cane',
        'I54': 'Walker',
        'I55': 'Bathbenc',
        'I56': 'Grab Bars',
        'I57': 'Wheelcha',
        'I58': 'Commod',
        'I59': 'Bed Hospi',
        'I60': 'IM Syringe',
        'I61': 'Oth. SQ Syringe',
        'I65': None,
        'B69': None,
        'B70': None,
        'B71': None,
        'B72': None,
        'B73': None,
        'B74': None,
        'B75': None,
        'B76': None,
        'B77': None,
        'B78': None,
        'B79': None,
        'B80': None,
        'B81': None,
        'B82': None,
        'B83': None,
        'B84': None,
        'B85': None,
        'B86': None,
        'B87': None,
        'B88': None,
        'B89': None,
        'I70': None,
        'I74': None,
        'I78': None,
        'I82': None,
        'I83': None,
        'I87': None,
        'I88': None,
        'I93': None,
        'J93': None,
        'I94': None,
        'J94': None,
        'E97': 'ADD O2 SUPPLIES FOR LIFE STAR',
        'B101': None,
        'G101': 'Syringe',
        'B106': 'WOUND #',
        'B107': 'WOUND LOCATION',
        'B108': 'WOUND SIZE',
        'B109': 'HEIGHT',
        'B110': 'LENGTH',
        'B111': 'WIDTH',
        'F105': 'cm',
        'E106': '#1',
        'F106': '#2',
        'G106': '#3',
        'H106': '#4',
        'I106': '#5',
        'J106': '#6',
        'E107': None,
        'F107': None,
        'G107': None,
        'H107': None,
        'I107': None,
        'J107': None,
        'E109': None,
        'F109': None,
        'G109': None,
        'H109': None,
        'I109': None,
        'J109': None,
        'E110': None,
        'F110': None,
        'G110': None,
        'H110': None,
        'I110': None,
        'J110': None,
        'E111': None,
        'F111': None,
        'G111': None,
        'H111': None,
        'I111': None,
        'J111': None,
        'E114': 'SITE#1',
        'F114': 'SITE#2',
        'G114': 'SITE#3',
        'H114': 'SITE#4',
        'I114': 'SITE#5',
        'E115': None,
        'F115': None,
        'G115': None,
        'H115': None,
        'I115': None,
        'E116': None,
        'F116': None,
        'G116': None,
        'H116': None,
        'I116': None,
        'E118': None,
        'E122': None,
        'G122': 'SHUNT',
        'I122': 'left',
        'E123': None,
        'E127': 'right upper arm double lumen midline',
        'E128': 'Ertapenem 1gm/50mL normal saline, administer over 30min via RUE midline for 10 days (start date 09/26/2023, end date 10/05/2023). Flush with 5-10mL Normal Saline 0.9% before and after each use. Flush with 3-5mL Heparin 100units/mL 5mL syring after each use',
        'E129': 'Ertapenem 1gm/50mL',
        'E130': None,
        'E134': None,
        'E135': None,
        'E137': None,
        'E138': None,
        'E139': None}
        start_time = time.time()
        print ('opening the exel sheet...')
        app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
        wb = app.books.open(self.app.excel_path)
        self.progress_signal.emit(24)
        ws = wb.sheets['Basic- Auto']
        for cell in excel_inputs:
            ws.range(cell).value = excel_inputs[cell]
        self.progress_signal.emit(53)
        # Save and close
        wb.save()

        # Iterate over the mapping
        for cell, widget in widget_to_excel.items():
            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                value = ''
            ws.range(cell).value = str(value)
        # Save and close the workbook
        wb.save()
        widget_to_excel = {
 'J13': self.app.visionJ13,
 'J14': self.app.hearingJ14,
 'J20': self.app.painJ20,
 'F50': self.app.dietF50,
 'F53': self.app.incontinanceF53,
 'B62': self.app.edemaB62,
 'B65': self.app.sopB65,
 'F105': self.app.cmF105,
 'B147': self.app.edemaB147,
 'E8': self.app.line_editE8,
 'H8': self.app.line_editH8,
 'I8': self.app.line_editI8,
 'I13': self.app.line_editI13,
 'E14': self.app.line_editE14,
 'F14': self.app.line_editF14,
 'I14': self.app.line_editI14,
 'E15': self.app.line_editE15,
 'F15': self.app.line_editF15,
 'E16': self.app.line_editE16,
 'F16': self.app.line_editF16,
 'H17': self.app.line_editH17,
 'J17': self.app.line_editJ17,
 'E17': self.app.line_editE17,
 'F17': self.app.line_editF17,
 'E18': self.app.line_editE18,
 'F18': self.app.line_editF18,
 'H20': self.app.line_editH20,
 'E19': self.app.line_editE19,
 'F19': self.app.line_editF19,
 'E20': self.app.line_editE20,
 'F20': self.app.line_editF20,
 'H23': self.app.line_editH23,
 'I23': self.app.line_editI23,
 'H24': self.app.line_editH24,
 'I24': self.app.line_editI24,
 'E21': self.app.line_editE21,
 'F21': self.app.line_editF21,
 'E22': self.app.line_editE22,
 'F22': self.app.line_editF22,
 'F28': self.app.line_editF28,
 'F29': self.app.line_editF29,
 'F30': self.app.line_editF30,
 'F31': self.app.line_editF31,
 'F32': self.app.line_editF32,
 'F33': self.app.line_editF33,
 'F34': self.app.line_editF34,
 'F35': self.app.line_editF35,
 'F36': self.app.line_editF36,
 'F37': self.app.line_editF37,
 'F38': self.app.line_editF38,
 'F39': self.app.line_editF39,
 'F40': self.app.line_editF40,
 'F41': self.app.line_editF41,
 'F42': self.app.line_editF42,
 'F43': self.app.line_editF43,
 'F44': self.app.line_editF44,
 'I28': self.app.line_editI28,
 'I29': self.app.line_editI29,
 'I30': self.app.line_editI30,
 'I31': self.app.line_editI31,
 'I32': self.app.line_editI32,
 'I33': self.app.line_editI33,
 'I34': self.app.line_editI34,
 'I35': self.app.line_editI35,
 'I36': self.app.line_editI36,
 'I37': self.app.line_editI37,
 'I38': self.app.line_editI38,
 'I39': self.app.line_editI39,
 'I40': self.app.line_editI40,
 'I41': self.app.line_editI41,
 'I42': self.app.line_editI42,
 'I43': self.app.line_editI43,
 'I44': self.app.line_editI44,
 'B53': self.app.line_editB53,
 'I53': self.app.line_editI53,
 'I54': self.app.line_editI54,
 'I55': self.app.line_editI55,
 'I56': self.app.line_editI56,
 'I57': self.app.line_editI57,
 'I58': self.app.line_editI58,
 'I59': self.app.line_editI59,
 'I60': self.app.line_editI60,
 'I61': self.app.line_editI61,
 'I62': self.app.line_editI62,
 'J53': self.app.line_editJ53,
 'J54': self.app.line_editJ54,
 'J55': self.app.line_editJ55,
 'J56': self.app.line_editJ56,
 'J57': self.app.line_editJ57,
 'J58': self.app.line_editJ58,
 'J59': self.app.line_editJ59,
 'J60': self.app.line_editJ60,
 'J61': self.app.line_editJ61,
 'J62': self.app.line_editJ62,
 'F56': self.app.line_editF56,
 'F57': self.app.line_editF57,
 'F58': self.app.line_editF58,
 'F59': self.app.line_editF59,
 'F60': self.app.line_editF60,
 'F61': self.app.line_editF61,
 'G56': self.app.line_editG56,
 'G57': self.app.line_editG57,
 'G58': self.app.line_editG58,
 'G59': self.app.line_editG59,
 'G60': self.app.line_editG60,
 'G61': self.app.line_editG61,
 'C59': self.app.line_editC59,
 'B59': self.app.line_editB59,
 'B56': self.app.line_editB56,
 'F65': self.app.line_editF65,
 'I65': self.app.line_editI65,
 'C145': self.app.line_editC145,
 'D145': self.app.line_editD145,
 'E145': self.app.line_editE145,
 'F145': self.app.line_editF145,
 'G145': self.app.line_editG145,
 'C146': self.app.line_editC146,
 'D146': self.app.line_editD146,
 'E146': self.app.line_editE146,
 'F146': self.app.line_editF146,
 'G146': self.app.line_editG146,
 'C150': self.app.line_editC150,
 'C151': self.app.line_editC151,
 'C152': self.app.line_editC152,
 'D150': self.app.line_editD150,
 'D151': self.app.line_editD151,
 'D152': self.app.line_editD152,

 'C2': self.app.poc_typeC2,
 'E5': self.app.musculoskeletalE5,
 'H5': self.app.hypertensionH5,
 'A5': self.app.templateA5,
 'B8': self.app.snv_frequencyB8,
 'J8': self.app.line_editJ8,
 'J11': self.app.directiveJ11,
 'B11': self.app.pt_or_notB11,
 'B14': self.app.priority_codeB14,
 'B20': self.app.copd_asthmaB20,
 'G101': self.app.typeG101,
 'G122': self.app.SHUNTtypeG122,
 'I122': self.app.SHUNTlocaI122,
 'E11': self.app.line_editE11,
 'H11': self.app.line_editH11,
 'B69': self.app.line_editB69,
 'B26': self.app.line_editB26,
 'B70': self.app.line_editB70,
 'B27': self.app.line_editB27,
 'B71': self.app.line_editB71,
 'B28': self.app.line_editB28,
 'B72': self.app.line_editB72,
 'B29': self.app.line_editB29,
 'B17': self.app.line_editB17,
 'B73': self.app.line_editB73,
 'B30': self.app.line_editB30,
 'B74': self.app.line_editB74,
 'B31': self.app.line_editB31,
 'B75': self.app.line_editB75,
 'B32': self.app.line_editB32,
 'B76': self.app.line_editB76,
 'B33': self.app.line_editB33,
 'B23': self.app.line_editB23,
 'B77': self.app.line_editB77,
 'B34': self.app.line_editB34,
 'B78': self.app.line_editB78,
 'B35': self.app.line_editB35,
 'B79': self.app.line_editB79,
 'B36': self.app.line_editB36,
 'B80': self.app.line_editB80,
 'B37': self.app.line_editB37,
 'B81': self.app.line_editB81,
 'B38': self.app.line_editB38,
 'B82': self.app.line_editB82,
 'B39': self.app.line_editB39,
 'B83': self.app.line_editB83,
 'B40': self.app.line_editB40,
 'B84': self.app.line_editB84,
 'B41': self.app.line_editB41,
 'B85': self.app.line_editB85,
 'B42': self.app.line_editB42,
 'B86': self.app.line_editB86,
 'B43': self.app.line_editB43,
 'B87': self.app.line_editB87,
 'B44': self.app.line_editB44,
 'B88': self.app.line_editB88,
 'B45': self.app.line_editB45,
 'B89': self.app.line_editB89,
 'B46': self.app.line_editB46,
 'B47': self.app.line_editB47,
 'B48': self.app.line_editB48,
 'F48': self.app.line_editF48,
 'I65': self.app.line_editI65,
 'B49': self.app.line_editB49,
 'B50': self.app.line_editB50,
 'I70': self.app.line_editI70,
 'I74': self.app.line_editI74,
 'I78': self.app.line_editI78,
 'I82': self.app.line_editI82,
 'I83': self.app.line_editI83,
 'I87': self.app.line_editI87,
 'I88': self.app.line_editI88,
 'I93': self.app.line_editI93,
 'J93': self.app.line_editJ93,
 'I94': self.app.line_editI94,
 'J94': self.app.line_editJ94,
 'B101': self.app.line_editB101,
 'E107': self.app.line_editE107,
 'F107': self.app.line_editF107,
 'G107': self.app.line_editG107,
 'H107': self.app.line_editH107,
 'I107': self.app.line_editI107,
 'J107': self.app.line_editJ107,
 'E108': self.app.line_editE108,
 'F108': self.app.line_editF108,
 'G108': self.app.line_editG108,
 'H108': self.app.line_editH108,
 'I108': self.app.line_editI108,
 'J108': self.app.line_editJ108,
 'E109': self.app.line_editE109,
 'F109': self.app.line_editF109,
 'G109': self.app.line_editG109,
 'H109': self.app.line_editH109,
 'I109': self.app.line_editI109,
 'J109': self.app.line_editJ109,
 'E110': self.app.line_editE110,
 'F110': self.app.line_editF110,
 'G110': self.app.line_editG110,
 'H110': self.app.line_editH110,
 'I110': self.app.line_editI110,
 'J110': self.app.line_editJ110,
 'E111': self.app.line_editE111,
 'F111': self.app.line_editF111,
 'G111': self.app.line_editG111,
 'H111': self.app.line_editH111,
 'I111': self.app.line_editI111,
 'J111': self.app.line_editJ111,
 'E115': self.app.line_editE115,
 'F115': self.app.line_editF115,
 'G115': self.app.line_editG115,
 'H115': self.app.line_editH115,
 'I115': self.app.line_editI115,
 'E116': self.app.line_editE116,
 'F116': self.app.line_editF116,
 'G116': self.app.line_editG116,
 'H116': self.app.line_editH116,
 'I116': self.app.line_editI116,
 'E118': self.app.line_editE118,
 'E122': self.app.line_editE122,
 'E123': self.app.line_editE123,
 'E127': self.app.line_editE127,
 'E128': self.app.line_editE128,
 'E129': self.app.line_editE129,
 'E130': self.app.line_editE130,
 'E134': self.app.line_editE134,
 'E135': self.app.line_editE135,
 'E136': self.app.line_editE136,
 'E137': self.app.line_editE137,
 'E138': self.app.line_editE138,
 'E139': self.app.line_editE139,
 'E140': self.app.line_editE140}

        self.progress_signal.emit(64)
        # Iterate over the mapping
        for cell, widget in widget_to_excel.items():
            value = ws.range(cell).value
            if isinstance(widget, QComboBox):
                # Find the index of the value in the combo box and set it
                index = widget.findText(str(value))
                if index != -1:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            else:
                pass

        value_from_L4 = ws.range('L4').value
        value_from_O4 = ws.range('O4').value
        value_from_R4 = ws.range('R4').value
        self.L4_signal.emit(value_from_L4)
        self.O4_signal.emit(value_from_O4)
        self.R4_signal.emit(value_from_R4)

        wb.close()
        self.progress_signal.emit(78)
        app.quit()
        self.progress_signal.emit(89)
        end_time = time.time()
        print ('closing the exel sheet...')
        execution_time = end_time - start_time
        print(f"Time of Updating the Excel sheet: {execution_time:.2f} seconds")
        print(f"Updated Excel sheet '{self.app.excel_path}' with :")
        for cell in excel_inputs:    
            print(f"value '{excel_inputs[cell]}' in cell '{cell}'.")
        self.progress_signal.emit(100)
        self.progress_signal.emit(0)
        self.done.emit()

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGraphicsView, QGraphicsScene, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import fitz  # PyMuPDF
from PIL import Image
from PyQt5.QtGui import QPainter, QTransform
from io import BytesIO
from PyQt5.QtCore import QThread, pyqtSignal

class choosePdfThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_loading = pyqtSignal()
    finished_rendering = pyqtSignal()
    request_display = pyqtSignal()

    def __init__(self, viewer, file_path):
        super().__init__()
        self.viewer = viewer
        self.file_path = file_path

    def run(self):
        self.progress_signal.emit(33)
        self.load_pdf()
        self.progress_signal.emit(66)
        self.finished_loading.emit()
        self.render_pdf()
        self.progress_signal.emit(100)
        self.finished_rendering.emit()
        self.request_display.emit()
        self.progress_signal.emit(0)

    def load_pdf(self):
        self.viewer.pdf = fitz.open(self.file_path)

    def render_pdf(self):
        self.viewer.render_pdf()

class chooseExcelThread(QThread):
    progress_signal = pyqtSignal(int)
    L4_signal=pyqtSignal(str)
    O4_signal=pyqtSignal(str)
    R4_signal=pyqtSignal(str)
    def __init__(self,app, file_path):
        super().__init__()
        self.app = app
        self.file_path = file_path

    def run(self):
        widget_to_excel = {
 'J13': self.app.visionJ13,
 'J14': self.app.hearingJ14,
 'J20': self.app.painJ20,
 'F50': self.app.dietF50,
 'F53': self.app.incontinanceF53,
 'B62': self.app.edemaB62,
 'B65': self.app.sopB65,
 'F105': self.app.cmF105,
 'B147': self.app.edemaB147,
 'E8': self.app.line_editE8,
 'H8': self.app.line_editH8,
 'I8': self.app.line_editI8,
 'I13': self.app.line_editI13,
 'E14': self.app.line_editE14,
 'F14': self.app.line_editF14,
 'I14': self.app.line_editI14,
 'E15': self.app.line_editE15,
 'F15': self.app.line_editF15,
 'E16': self.app.line_editE16,
 'F16': self.app.line_editF16,
 'H17': self.app.line_editH17,
 'J17': self.app.line_editJ17,
 'E17': self.app.line_editE17,
 'F17': self.app.line_editF17,
 'E18': self.app.line_editE18,
 'F18': self.app.line_editF18,
 'H20': self.app.line_editH20,
 'E19': self.app.line_editE19,
 'F19': self.app.line_editF19,
 'E20': self.app.line_editE20,
 'F20': self.app.line_editF20,
 'H23': self.app.line_editH23,
 'I23': self.app.line_editI23,
 'H24': self.app.line_editH24,
 'I24': self.app.line_editI24,
 'E21': self.app.line_editE21,
 'F21': self.app.line_editF21,
 'E22': self.app.line_editE22,
 'F22': self.app.line_editF22,
 'F28': self.app.line_editF28,
 'F29': self.app.line_editF29,
 'F30': self.app.line_editF30,
 'F31': self.app.line_editF31,
 'F32': self.app.line_editF32,
 'F33': self.app.line_editF33,
 'F34': self.app.line_editF34,
 'F35': self.app.line_editF35,
 'F36': self.app.line_editF36,
 'F37': self.app.line_editF37,
 'F38': self.app.line_editF38,
 'F39': self.app.line_editF39,
 'F40': self.app.line_editF40,
 'F41': self.app.line_editF41,
 'F42': self.app.line_editF42,
 'F43': self.app.line_editF43,
 'F44': self.app.line_editF44,
 'I28': self.app.line_editI28,
 'I29': self.app.line_editI29,
 'I30': self.app.line_editI30,
 'I31': self.app.line_editI31,
 'I32': self.app.line_editI32,
 'I33': self.app.line_editI33,
 'I34': self.app.line_editI34,
 'I35': self.app.line_editI35,
 'I36': self.app.line_editI36,
 'I37': self.app.line_editI37,
 'I38': self.app.line_editI38,
 'I39': self.app.line_editI39,
 'I40': self.app.line_editI40,
 'I41': self.app.line_editI41,
 'I42': self.app.line_editI42,
 'I43': self.app.line_editI43,
 'I44': self.app.line_editI44,
 'B53': self.app.line_editB53,
 'I53': self.app.line_editI53,
 'I54': self.app.line_editI54,
 'I55': self.app.line_editI55,
 'I56': self.app.line_editI56,
 'I57': self.app.line_editI57,
 'I58': self.app.line_editI58,
 'I59': self.app.line_editI59,
 'I60': self.app.line_editI60,
 'I61': self.app.line_editI61,
 'I62': self.app.line_editI62,
 'J53': self.app.line_editJ53,
 'J54': self.app.line_editJ54,
 'J55': self.app.line_editJ55,
 'J56': self.app.line_editJ56,
 'J57': self.app.line_editJ57,
 'J58': self.app.line_editJ58,
 'J59': self.app.line_editJ59,
 'J60': self.app.line_editJ60,
 'J61': self.app.line_editJ61,
 'J62': self.app.line_editJ62,
 'F56': self.app.line_editF56,
 'F57': self.app.line_editF57,
 'F58': self.app.line_editF58,
 'F59': self.app.line_editF59,
 'F60': self.app.line_editF60,
 'F61': self.app.line_editF61,
 'G56': self.app.line_editG56,
 'G57': self.app.line_editG57,
 'G58': self.app.line_editG58,
 'G59': self.app.line_editG59,
 'G60': self.app.line_editG60,
 'G61': self.app.line_editG61,
 'C59': self.app.line_editC59,
 'B59': self.app.line_editB59,
 'B56': self.app.line_editB56,
 'F65': self.app.line_editF65,
 'I65': self.app.line_editI65,
 'C145': self.app.line_editC145,
 'D145': self.app.line_editD145,
 'E145': self.app.line_editE145,
 'F145': self.app.line_editF145,
 'G145': self.app.line_editG145,
 'C146': self.app.line_editC146,
 'D146': self.app.line_editD146,
 'E146': self.app.line_editE146,
 'F146': self.app.line_editF146,
 'G146': self.app.line_editG146,
 'C150': self.app.line_editC150,
 'C151': self.app.line_editC151,
 'C152': self.app.line_editC152,
 'D150': self.app.line_editD150,
 'D151': self.app.line_editD151,
 'D152': self.app.line_editD152,

 'C2': self.app.poc_typeC2,
 'E5': self.app.musculoskeletalE5,
 'H5': self.app.hypertensionH5,
 'A5': self.app.templateA5,
 'B8': self.app.snv_frequencyB8,
 'J8': self.app.line_editJ8,
 'J11': self.app.directiveJ11,
 'B11': self.app.pt_or_notB11,
 'B14': self.app.priority_codeB14,
 'B20': self.app.copd_asthmaB20,
 'G101': self.app.typeG101,
 'G122': self.app.SHUNTtypeG122,
 'I122': self.app.SHUNTlocaI122,
 'E11': self.app.line_editE11,
 'H11': self.app.line_editH11,
 'B69': self.app.line_editB69,
 'B26': self.app.line_editB26,
 'B70': self.app.line_editB70,
 'B27': self.app.line_editB27,
 'B71': self.app.line_editB71,
 'B28': self.app.line_editB28,
 'B72': self.app.line_editB72,
 'B29': self.app.line_editB29,
 'B17': self.app.line_editB17,
 'B73': self.app.line_editB73,
 'B30': self.app.line_editB30,
 'B74': self.app.line_editB74,
 'B31': self.app.line_editB31,
 'B75': self.app.line_editB75,
 'B32': self.app.line_editB32,
 'B76': self.app.line_editB76,
 'B33': self.app.line_editB33,
 'B23': self.app.line_editB23,
 'B77': self.app.line_editB77,
 'B34': self.app.line_editB34,
 'B78': self.app.line_editB78,
 'B35': self.app.line_editB35,
 'B79': self.app.line_editB79,
 'B36': self.app.line_editB36,
 'B80': self.app.line_editB80,
 'B37': self.app.line_editB37,
 'B81': self.app.line_editB81,
 'B38': self.app.line_editB38,
 'B82': self.app.line_editB82,
 'B39': self.app.line_editB39,
 'B83': self.app.line_editB83,
 'B40': self.app.line_editB40,
 'B84': self.app.line_editB84,
 'B41': self.app.line_editB41,
 'B85': self.app.line_editB85,
 'B42': self.app.line_editB42,
 'B86': self.app.line_editB86,
 'B43': self.app.line_editB43,
 'B87': self.app.line_editB87,
 'B44': self.app.line_editB44,
 'B88': self.app.line_editB88,
 'B45': self.app.line_editB45,
 'B89': self.app.line_editB89,
 'B46': self.app.line_editB46,
 'B47': self.app.line_editB47,
 'B48': self.app.line_editB48,
 'F48': self.app.line_editF48,
 'I65': self.app.line_editI65,
 'B49': self.app.line_editB49,
 'B50': self.app.line_editB50,
 'I70': self.app.line_editI70,
 'I74': self.app.line_editI74,
 'I78': self.app.line_editI78,
 'I82': self.app.line_editI82,
 'I83': self.app.line_editI83,
 'I87': self.app.line_editI87,
 'I88': self.app.line_editI88,
 'I93': self.app.line_editI93,
 'J93': self.app.line_editJ93,
 'I94': self.app.line_editI94,
 'J94': self.app.line_editJ94,
 'B101': self.app.line_editB101,
 'E107': self.app.line_editE107,
 'F107': self.app.line_editF107,
 'G107': self.app.line_editG107,
 'H107': self.app.line_editH107,
 'I107': self.app.line_editI107,
 'J107': self.app.line_editJ107,
 'E108': self.app.line_editE108,
 'F108': self.app.line_editF108,
 'G108': self.app.line_editG108,
 'H108': self.app.line_editH108,
 'I108': self.app.line_editI108,
 'J108': self.app.line_editJ108,
 'E109': self.app.line_editE109,
 'F109': self.app.line_editF109,
 'G109': self.app.line_editG109,
 'H109': self.app.line_editH109,
 'I109': self.app.line_editI109,
 'J109': self.app.line_editJ109,
 'E110': self.app.line_editE110,
 'F110': self.app.line_editF110,
 'G110': self.app.line_editG110,
 'H110': self.app.line_editH110,
 'I110': self.app.line_editI110,
 'J110': self.app.line_editJ110,
 'E111': self.app.line_editE111,
 'F111': self.app.line_editF111,
 'G111': self.app.line_editG111,
 'H111': self.app.line_editH111,
 'I111': self.app.line_editI111,
 'J111': self.app.line_editJ111,
 'E115': self.app.line_editE115,
 'F115': self.app.line_editF115,
 'G115': self.app.line_editG115,
 'H115': self.app.line_editH115,
 'I115': self.app.line_editI115,
 'E116': self.app.line_editE116,
 'F116': self.app.line_editF116,
 'G116': self.app.line_editG116,
 'H116': self.app.line_editH116,
 'I116': self.app.line_editI116,
 'E118': self.app.line_editE118,
 'E122': self.app.line_editE122,
 'E123': self.app.line_editE123,
 'E127': self.app.line_editE127,
 'E128': self.app.line_editE128,
 'E129': self.app.line_editE129,
 'E130': self.app.line_editE130,
 'E134': self.app.line_editE134,
 'E135': self.app.line_editE135,
 'E136': self.app.line_editE136,
 'E137': self.app.line_editE137,
 'E138': self.app.line_editE138,
 'E139': self.app.line_editE139,
 'E140': self.app.line_editE140}
        
        app = xw.App(visible=False)  # Set visible to True if you want to see Excel open
        self.progress_signal.emit(21)
        wb = app.books.open(self.file_path)
        self.progress_signal.emit(34)
        ws = wb.sheets['Basic- Auto']
        self.progress_signal.emit(67)
        # Save and close
        wb.save()

        # Iterate over the mapping
        for cell, widget in widget_to_excel.items():
            value = ws.range(cell).value
            if isinstance(widget, QComboBox):
                # Find the index of the value in the combo box and set it
                index = widget.findText(str(value))
                if index != -1:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")
            else:
                pass

        self.progress_signal.emit(82)
        value_from_L4 = ws.range('L4').value
        value_from_O4 = ws.range('O4').value
        value_from_R4 = ws.range('R4').value
        self.L4_signal.emit(value_from_L4)
        self.O4_signal.emit(value_from_O4)
        self.R4_signal.emit(value_from_R4)
        wb.close()
        self.progress_signal.emit(100)
        app.quit()
        self.progress_signal.emit(0)

class PDFViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.zoom_factor = 0.7
        self.pdf = None

        layout = QVBoxLayout(self)

        # Graphics View and Scene
        self.view = QGraphicsView(self)
        layout.addWidget(self.view)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.view.wheelEvent = self.on_mouse_scroll

    def on_mouse_scroll(self, event):
        # Check for Ctrl key being pressed for zooming with scroll
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:  # Zoom in
                self.zoom_in()
            else:  # Zoom out
                self.zoom_out()
        else:
            # Default behavior: scrolling
            super(QGraphicsView, self.view).wheelEvent(event)

    def load_pdf(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
            if file_path:
                self.pdf = fitz.open(file_path)
                print('1')
                self.render_pdf()
                print('3')
                self.display_pdf()
                print('9')
        except:
            print(f"mouchkil f load ")      

    def render_pdf(self):
        
        images = []
        zoom=200//72
        for page_num in range(len(self.pdf)):
            page = self.pdf[page_num]
            zoom_matrix = fitz.Matrix(zoom,zoom)
            pix = page.get_pixmap(matrix=zoom_matrix)
            fmt = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(fmt, [pix.width, pix.height], pix.samples)
            images.append(img)

        # Stitch the images vertically
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        self.stitched_image = Image.new('RGB', (total_width, total_height))
        y_offset = 0
        for img in images:
            self.stitched_image.paste(img, (0, y_offset))
            y_offset += img.height

    def display_pdf(self):
        self.scene.clear()
        scaled_width = int(self.stitched_image.width * self.zoom_factor)
        scaled_height = int(self.stitched_image.height * self.zoom_factor)
        scaled_image = self.stitched_image.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Convert the PIL image to JPEG format
        buffered = BytesIO()
        scaled_image.save(buffered, format="JPEG")
        jpeg_data = buffered.getvalue()

        # Convert the JPEG data to QImage
        qt_image = QImage.fromData(jpeg_data)

        if not qt_image.isNull():
            pixmap = QPixmap.fromImage(qt_image)
            self.scene.addPixmap(pixmap)
            self.view.setSceneRect(0, 0, scaled_width, scaled_height)
            self.apply_zoom()
        else:
            print("Failed to convert JPEG data to QImage. QImage is null.")

    def zoom_in(self):
        self.zoom_factor = min(4.0, self.zoom_factor + 0.12)
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_factor = max(0.1, self.zoom_factor - 0.12)
        self.apply_zoom()

    def apply_zoom(self):
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        self.view.setTransform(transform)

from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QPalette, QFont
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QLineEdit, QLabel, QComboBox, QHBoxLayout, QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class FocusLineEdit(QLineEdit):
    def __init__(self, next_widget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_widget = next_widget  # Store the next widget to focus

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.next_widget:
                self.next_widget.setFocus()  # Set focus to the next widget

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.restarting=False
        self.scaning=False
        self.updating=False

        # Set Font to Segoe UI
        font = QFont("Calibri", 14)  # Using Segoe UI font
        self.setFont(font)

        # Define settings and file-selected flags as instance variables
        self.denoising = False
        self.sharpen = False
        self.correcting_skew = True
        self.show_imgs = True

        # Initialize the flags for chosen files
        self.pdf_selected = False
        self.excel_selected = False
                # Progress Bar
        self.progress_bar = QProgressBar(self)
        # Initialize the animation for the progress bar
        self.progress_animation = QPropertyAnimation(self.progress_bar, b"value")
        self.progress_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.apply_dark_theme()
        self.init_ui()
        # self.setFixedSize(1750,900)
        self.resize(1700, 900)
        self.showMaximized()

    def init_ui(self):
        line_edit_font = QFont("Calibri", 9)
        main_layout = QVBoxLayout()
        central_layout = QHBoxLayout()

        # Left Side: 8 text inputs with labels, file selectors, and file display labels
        left_layout = QVBoxLayout()

        # Text inputs with labels
        self.inputs = {}

        # Create the scrolling area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)


        # The widget that will contain all the input fields and their labels
        inputs_widget = QWidget()
        form_layout = QFormLayout()  # Form layout for the labels and inputs
        MAX_WIDTH = 200  
        MAX_WIDTH_LINEEDIT =100
        # POC TYPE ComboBox


        self.poc_typeC2 = QComboBox()
        self.poc_typeC2.setMaximumWidth(MAX_WIDTH)
        self.poc_typeC2.addItems(['SOC', 'Recert', 'ROC'])
        drob = QHBoxLayout()
        label1 = QLabel("POC TYPE:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("")
        label3 = QLabel("")
        drob.addWidget(label1)
        drob.addWidget(self.poc_typeC2)
        row = QHBoxLayout()
        row.addLayout(drob,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("TEMPLATE:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("Musculoskeletal:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("Hypertension:")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)



        # Musculoskeletal ComboBox
        
        self.templateA5 = QComboBox()
        self.templateA5.setMaximumWidth(MAX_WIDTH)
        self.templateA5.addItems(['BASIC', 'DM', 'DM/INSULIN', 'CHF', 'CHF/COPD', 'DM/CHF/COPD', 'DM/COPD', 'DM/CHF', 'DM/COPD', 'COPD/DM/INSULIN', 'CHF/DM/INSULIN', 'CHF/COPD & DM/INSULIN', 'COPD/O2'])
        
        self.musculoskeletalE5 = QComboBox()
        self.musculoskeletalE5.setMaximumWidth(MAX_WIDTH)
        self.musculoskeletalE5.addItems(['$$$Musculoskeletal', '$$$Osteoarthritis', '$$$Arthropathies'])
        self.hypertensionH5 = QComboBox()
        self.hypertensionH5.setMaximumWidth(MAX_WIDTH)
        self.hypertensionH5.addItems(['$$$Hypertensive heart', '$$$Normal BP', 'NONE'])
        row = QHBoxLayout()
        row.addWidget(self.templateA5,1)
        row.addWidget(self.musculoskeletalE5,1)
        row.addWidget(self.hypertensionH5,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("SNV Frequency:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("VACCINATION:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        
        hight_wight = QHBoxLayout()
        label3 = QLabel("WEIGTH")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label4 = QLabel("HEIGHT")
        label4.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label5 = QLabel("BMI")
        label5.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        
        hight_wight.addWidget(label3,1)
        hight_wight.addWidget(label4,1)
        hight_wight.addWidget(label5,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(hight_wight,1)
        form_layout.addRow(row)



        self.snv_frequencyB8 = QComboBox()
        self.snv_frequencyB8.setMaximumWidth(MAX_WIDTH)
        self.snv_frequencyB8.addItems(['2wx2, 1wx7', '1wx1, 2wx2, 1wx6', '2wx1, 1wx8', '1wx1, 2wx1, 1wx7', 'Once a Day x 10 Days, 1wx7'])
        self.line_editE8 = QLineEdit(self)
        self.line_editE8.setFont(line_edit_font)
        self.line_editE8.setMaximumWidth(200)
        hight_wight = QHBoxLayout()
        self.line_editH8 = QLineEdit(self)
        self.line_editH8.setFont(line_edit_font)
        self.line_editH8.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editH8.setMaximumWidth(65)
        self.line_editI8 = QLineEdit(self)
        self.line_editI8.setFont(line_edit_font)
        self.line_editI8.setMaximumWidth(65)
        self.line_editJ8 = QLineEdit(self)
        self.line_editJ8.setFont(line_edit_font)
        self.line_editJ8.setMaximumWidth(65)
        hight_wight.addWidget(self.line_editH8,1)
        hight_wight.addWidget(self.line_editI8,1)
        hight_wight.addWidget(self.line_editJ8,1)
        row = QHBoxLayout()
        row.addWidget(self.snv_frequencyB8,1)
        row.addWidget(self.line_editE8,1)
        row.addLayout(hight_wight,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)




        label1 = QLabel("PT OR NOT:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("SURGERIES:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("DIRECTIVE:")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label4 = QLabel("")
        DIRECTIVE1 = QHBoxLayout()
        DIRECTIVE1.addWidget(label3)
        DIRECTIVE1.addWidget(label4)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(DIRECTIVE1,1)
        form_layout.addRow(row)


        self.pt_or_notB11 = QComboBox()
        self.pt_or_notB11.setMaximumWidth(MAX_WIDTH)
        self.pt_or_notB11.addItems(['NONE', 'PT', 'OT', 'PT&OT', 'CHHA PT&CHHA', 'OT&CHHA', 'PT, OT, &CHHA'])
        self.line_editE11 = QLineEdit(self)
        self.line_editE11.setFont(line_edit_font)
        self.line_editE11.setMaximumWidth(200)
        row = QHBoxLayout()
        row.addWidget(self.pt_or_notB11, 1)  # Stretch factor of 1
        row.addWidget(self.line_editE11, 1)          # Stretch factor of 1
        self.directiveJ11 = QComboBox()
        self.directiveJ11.setMaximumWidth(100)
        self.directiveJ11.addItems(['DNR', 'CPR', 'POWER OF ATTORNEY', 'DNI', 'LIVING WILL', 'NO ARTFICIAL NUTRITION AND HYDRATION'])
        self.line_editH11 = QLineEdit(self)
        self.line_editH11.setFont(line_edit_font)
        self.line_editH11.setMaximumWidth(100)
        DIRECTIVE2 = QHBoxLayout()
        DIRECTIVE2.addWidget(self.line_editH11, 1)      # Stretch factor of 1
        DIRECTIVE2.addWidget(self.directiveJ11, 1)  # Stretch factor of 1
        row.addLayout(DIRECTIVE2, 1)  
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)




        label1 = QLabel("PRIORITY CODE:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("1-9 RISK FOR HOSP")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        VISION = QHBoxLayout()
        label3 = QLabel("VISION   ")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editI13 = QLineEdit(self)
        self.line_editI13.setFont(line_edit_font)
        self.line_editI13.setMaximumWidth(65)
        self.visionJ13 = QComboBox()
        self.visionJ13.setMaximumWidth(65)
        self.visionJ13.addItems(['ADEQUATE VISION', 'IMPAIRED VISION', 'MODERATLY IMPAIRED VISION', 'HIGHLY IMPAIRED VISION', 'SEVERELY IMPAIRED VISION'])
        VISION.addWidget(label3,1)
        VISION.addWidget(self.line_editI13,1)
        VISION.addWidget(self.visionJ13,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(VISION,1)
        form_layout.addRow(row)



        self.priority_codeB14 = QComboBox()
        self.priority_codeB14.setMaximumWidth(MAX_WIDTH)
        self.priority_codeB14.addItems(['1 [HIGH]', '2.0', '3 [LOW]'])
        RISK = QHBoxLayout()
        self.line_editE14 = QLineEdit(self)
        self.line_editE14.setFont(line_edit_font)
        self.line_editE14.setMaximumWidth(100)
        self.line_editF14 = QLineEdit(self)
        self.line_editF14.setFont(line_edit_font)
        self.line_editF14.setMaximumWidth(100)
        RISK.addWidget(self.line_editE14,1)
        RISK.addWidget(self.line_editF14,1)
        HEARING = QHBoxLayout()
        label1 = QLabel("HEARING")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editI14 = QLineEdit(self)
        self.line_editI14.setFont(line_edit_font)
        self.line_editI14.setMaximumWidth(65)
        self.hearingJ14 = QComboBox()
        self.hearingJ14.setMaximumWidth(65)
        self.hearingJ14.addItems(['MINIMAL', 'ADEQUATE', 'MODERATLY IMPAIRED HEARING DIFFICULTY IN BILATERAL EARS', 'UNABLE TO ASSESS HEARING', 'HIGHLY IMPAIRED'])
        HEARING.addWidget(label1,1)
        HEARING.addWidget(self.line_editI14,1)
        HEARING.addWidget(self.hearingJ14,1)
        row = QHBoxLayout()
        row.addWidget(self.priority_codeB14,1)
        row.addLayout(RISK,1)
        row.addLayout(HEARING,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        RISK = QHBoxLayout()
        self.line_editE15 = QLineEdit(self)
        self.line_editE15.setFont(line_edit_font)
        self.line_editE15.setMaximumWidth(100)
        self.line_editF15 = QLineEdit(self)
        self.line_editF15.setFont(line_edit_font)
        self.line_editF15.setMaximumWidth(100)
        RISK.addWidget(self.line_editE15,1)
        RISK.addWidget(self.line_editF15,1)
        
        label2 = QLabel("")
        
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)




        label1 = QLabel("PCG:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        RISK = QHBoxLayout()
        self.line_editE16 = QLineEdit(self)
        self.line_editE16.setFont(line_edit_font)
        self.line_editE16.setMaximumWidth(100)
        self.line_editF16 = QLineEdit(self)
        self.line_editF16.setFont(line_edit_font)
        self.line_editF16.setMaximumWidth(100)
        RISK.addWidget(self.line_editE16,1)
        RISK.addWidget(self.line_editF16,1)
        DENTURES = QHBoxLayout()
        label2 = QLabel("DENTURES")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("")
        DENTURES.addWidget(label2,1)
        DENTURES.addWidget(label3,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addLayout(DENTURES,1)
        form_layout.addRow(row)




        self.line_editB17 = QLineEdit(self)
        self.line_editB17.setFont(line_edit_font)
        self.line_editB17.setMaximumWidth(200)
        RISK = QHBoxLayout()
        self.line_editE17 = QLineEdit(self)
        self.line_editE17.setFont(line_edit_font)
        self.line_editE17.setMaximumWidth(100)
        self.line_editF17 = QLineEdit(self)
        self.line_editF17.setFont(line_edit_font)
        self.line_editF17.setMaximumWidth(100)
        RISK.addWidget(self.line_editE17,1)
        RISK.addWidget(self.line_editF17,1)
        DENTURES = QHBoxLayout()
        self.line_editH17 = QLineEdit(self)
        self.line_editH17.setFont(line_edit_font)
        self.line_editH17.setMaximumWidth(100)
        self.line_editJ17 = QLineEdit(self)
        self.line_editJ17.setFont(line_edit_font)
        self.line_editJ17.setMaximumWidth(100)
        DENTURES.addWidget(self.line_editH17,1)
        DENTURES.addWidget(self.line_editJ17,1)
        row = QHBoxLayout()
        row.addWidget(self.line_editB17,1)
        row.addLayout(RISK,1)
        row.addLayout(DENTURES,1)
        form_layout.addRow(row)


        
        label1 = QLabel("")
        RISK = QHBoxLayout()
        self.line_editE18 = QLineEdit(self)
        self.line_editE18.setFont(line_edit_font)
        self.line_editE18.setMaximumWidth(100)
        self.line_editF18 = QLineEdit(self)
        self.line_editF18.setFont(line_edit_font)
        self.line_editF18.setMaximumWidth(100)
        RISK.addWidget(self.line_editE18,1)
        RISK.addWidget(self.line_editF18,1)
        label2 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)




        label1 = QLabel("COPD/ASTHMA:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        RISK = QHBoxLayout()
        self.line_editE19 = QLineEdit(self)
        self.line_editE19.setFont(line_edit_font)
        self.line_editE19.setMaximumWidth(100)
        self.line_editF19 = QLineEdit(self)
        self.line_editF19.setFont(line_edit_font)
        self.line_editF19.setMaximumWidth(100)
        RISK.addWidget(self.line_editE19,1)
        RISK.addWidget(self.line_editF19,1)
        PAIN = QHBoxLayout()
        label2 = QLabel("PAIN (LOCATION)")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("")
        PAIN.addWidget(label2,1)
        PAIN.addWidget(label3,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addLayout(PAIN,1)
        form_layout.addRow(row)



        self.copd_asthmaB20 = QComboBox()
        self.copd_asthmaB20.setMaximumWidth(MAX_WIDTH)
        self.copd_asthmaB20.addItems(['COPD', 'Asthma'])
        RISK = QHBoxLayout()
        self.line_editE20 = QLineEdit(self)
        self.line_editE20.setFont(line_edit_font)
        self.line_editE20.setMaximumWidth(100)
        self.line_editF20 = QLineEdit(self)
        self.line_editF20.setFont(line_edit_font)
        self.line_editF20.setMaximumWidth(100)
        RISK.addWidget(self.line_editE20,1)
        RISK.addWidget(self.line_editF20,1)
        PAIN = QHBoxLayout()
        self.line_editH20 = QLineEdit(self)
        self.line_editH20.setFont(line_edit_font)
        self.line_editH20.setMaximumWidth(150)
        self.painJ20 = QComboBox()
        self.painJ20.setMaximumWidth(66)
        self.painJ20.addItems(['DAILY', 'LESS OFTEN THAN DAILY ', 'ALL OF THE TIME ', 'NO'])
        PAIN.addWidget(self.line_editH20,1)
        PAIN.addWidget(self.painJ20,1)
        row = QHBoxLayout()
        row.addWidget(self.copd_asthmaB20,1)
        row.addLayout(RISK,1)
        row.addLayout(PAIN,1)
        form_layout.addRow(row)



        
        label1 = QLabel("")
        RISK = QHBoxLayout()
        self.line_editE21 = QLineEdit(self)
        self.line_editE21.setFont(line_edit_font)
        self.line_editE21.setMaximumWidth(100)
        self.line_editF21 = QLineEdit(self)
        self.line_editF21.setFont(line_edit_font)
        self.line_editF21.setMaximumWidth(100)
        RISK.addWidget(self.line_editE21,1)
        RISK.addWidget(self.line_editF21,1)
        label2 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)



        label1 = QLabel("ONCOLOGY:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        RISK = QHBoxLayout()
        self.line_editE22 = QLineEdit(self)
        self.line_editE22.setFont(line_edit_font)
        self.line_editE22.setMaximumWidth(100)
        self.line_editF22 = QLineEdit(self)
        self.line_editF22.setFont(line_edit_font)
        self.line_editF22.setMaximumWidth(100)
        RISK.addWidget(self.line_editE22,1)
        RISK.addWidget(self.line_editF22,1)
        label2 = QLabel("EXTRA")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(RISK,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)
        


        self.line_editB23 = QLineEdit(self)
        self.line_editB23.setFont(line_edit_font)
        self.line_editB23.setMaximumWidth(200)
        label1 = QLabel("")
        EXTRA = QHBoxLayout()
        self.line_editH23 = QLineEdit(self)
        self.line_editH23.setFont(line_edit_font)
        self.line_editH23.setMaximumWidth(100)
        self.line_editI23 = QLineEdit(self)
        self.line_editI23.setFont(line_edit_font)
        self.line_editI23.setMaximumWidth(100)
        EXTRA.addWidget(self.line_editH23,1)
        EXTRA.addWidget(self.line_editI23,1)
        row = QHBoxLayout()
        row.addWidget(self.line_editB23,1)
        row.addWidget(label1,1)
        row.addLayout(EXTRA,1)
        form_layout.addRow(row)




        
        label1 = QLabel("")
        label2 = QLabel("")
        EXTRA = QHBoxLayout()
        self.line_editH24 = QLineEdit(self)
        self.line_editH24.setFont(line_edit_font)
        self.line_editH24.setMaximumWidth(100)
        self.line_editI24 = QLineEdit(self)
        self.line_editI24.setFont(line_edit_font)
        self.line_editI24.setMaximumWidth(100)
        EXTRA.addWidget(self.line_editH24,1)
        EXTRA.addWidget(self.line_editI24,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(EXTRA,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)




        label2 = QLabel("CODES:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("DIAGNOSES:")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)







        self.line_editB89 = FocusLineEdit( parent=self)
        self.line_editB88 = FocusLineEdit(next_widget=self.line_editB89, parent=self)
        self.line_editB87 = FocusLineEdit(next_widget=self.line_editB88, parent=self)
        self.line_editB86 = FocusLineEdit(next_widget=self.line_editB87, parent=self)
        self.line_editB85 = FocusLineEdit(next_widget=self.line_editB86, parent=self)
        self.line_editB84 = FocusLineEdit(next_widget=self.line_editB85, parent=self)
        self.line_editB83 = FocusLineEdit(next_widget=self.line_editB84, parent=self)
        self.line_editB82 = FocusLineEdit(next_widget=self.line_editB83, parent=self)
        self.line_editB81 = FocusLineEdit(next_widget=self.line_editB82, parent=self)
        self.line_editB80 = FocusLineEdit(next_widget=self.line_editB81, parent=self)
        self.line_editB79 = FocusLineEdit(next_widget=self.line_editB80, parent=self)
        self.line_editB78 = FocusLineEdit(next_widget=self.line_editB79, parent=self)
        self.line_editB77 = FocusLineEdit(next_widget=self.line_editB78, parent=self)
        self.line_editB76 = FocusLineEdit(next_widget=self.line_editB77, parent=self)
        self.line_editB75 = FocusLineEdit(next_widget=self.line_editB76, parent=self)
        self.line_editB74 = FocusLineEdit(next_widget=self.line_editB75, parent=self)
        self.line_editB73 = FocusLineEdit(next_widget=self.line_editB74, parent=self)
        self.line_editB72 = FocusLineEdit(next_widget=self.line_editB73, parent=self)
        self.line_editB71 = FocusLineEdit(next_widget=self.line_editB72, parent=self)
        self.line_editB70 = FocusLineEdit(next_widget=self.line_editB71, parent=self)
        self.line_editB69 = FocusLineEdit(next_widget=self.line_editB70, parent=self)
 

















        # self.line_editB69 = FocusLineEdit(next_widget=self.line_editE11, parent=self)
        self.line_editB69.setFont(line_edit_font)
        self.line_editB69.setMaximumWidth(350)
        self.line_editB26 = QLineEdit(self)
        self.line_editB26.setFont(line_edit_font)
        self.line_editB26.setMaximumWidth(350)
        self.line_editB26.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB69,1)
        row.addWidget(self.line_editB26,1)
        form_layout.addRow(row)





        # self.line_editB70 = FocusLineEdit(next_widget=self.line_editB71, parent=self)
        self.line_editB70.setFont(line_edit_font)
        self.line_editB70.setMaximumWidth(350) 
        self.line_editB27 = QLineEdit(self)
        self.line_editB27.setFont(line_edit_font)
        self.line_editB27.setMaximumWidth(350)
        self.line_editB27.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB70,1)
        row.addWidget(self.line_editB27,1)
        form_layout.addRow(row)




        # self.line_editB71 = QLineEdit(self)
        self.line_editB71.setFont(line_edit_font)
        self.line_editB71.setMaximumWidth(350) 
        self.line_editB28 = QLineEdit(self)
        self.line_editB28.setFont(line_edit_font)
        self.line_editB28.setMaximumWidth(350)
        self.line_editB28.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB71,1)
        row.addWidget(self.line_editB28,1)
        form_layout.addRow(row)




        # self.line_editB72 = QLineEdit(self)
        self.line_editB72.setFont(line_edit_font)
        self.line_editB72.setMaximumWidth(350) 
        self.line_editB29 = QLineEdit(self)
        self.line_editB29.setFont(line_edit_font)
        self.line_editB29.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB72,1)
        row.addWidget(self.line_editB29,1)
        form_layout.addRow(row)



        # self.line_editB73 = QLineEdit(self)
        self.line_editB73.setFont(line_edit_font)
        self.line_editB73.setMaximumWidth(350) 
        self.line_editB30 = QLineEdit(self)
        self.line_editB30.setFont(line_edit_font)
        self.line_editB30.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB73,1)
        row.addWidget(self.line_editB30,1)
        form_layout.addRow(row)


        
        # self.line_editB74 = QLineEdit(self)
        self.line_editB74.setFont(line_edit_font)
        self.line_editB74.setMaximumWidth(350) 
        self.line_editB31 = QLineEdit(self)
        self.line_editB31.setFont(line_edit_font)
        self.line_editB31.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB74,1)
        row.addWidget(self.line_editB31,1)
        form_layout.addRow(row)



        # self.line_editB75 = QLineEdit(self)
        self.line_editB75.setFont(line_edit_font)
        self.line_editB75.setMaximumWidth(350) 
        self.line_editB32 = QLineEdit(self)
        self.line_editB32.setFont(line_edit_font)
        self.line_editB32.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB75,1)
        row.addWidget(self.line_editB32,1)
        form_layout.addRow(row)


        
        # self.line_editB76 = QLineEdit(self)
        self.line_editB76.setFont(line_edit_font)
        self.line_editB76.setMaximumWidth(350) 
        self.line_editB33 = QLineEdit(self)
        self.line_editB33.setFont(line_edit_font)
        self.line_editB33.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB76,1)
        row.addWidget(self.line_editB33,1)
        form_layout.addRow(row)



        # self.line_editB77 = QLineEdit(self)
        self.line_editB77.setFont(line_edit_font)
        self.line_editB77.setMaximumWidth(350) 
        self.line_editB34 = QLineEdit(self)
        self.line_editB34.setFont(line_edit_font)
        self.line_editB34.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB77,1)
        row.addWidget(self.line_editB34,1)
        form_layout.addRow(row)



        # self.line_editB78 = QLineEdit(self)
        self.line_editB78.setFont(line_edit_font)
        self.line_editB78.setMaximumWidth(350) 
        self.line_editB35 = QLineEdit(self)
        self.line_editB35.setFont(line_edit_font)
        self.line_editB35.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB78,1)
        row.addWidget(self.line_editB35,1)
        form_layout.addRow(row)



        # self.line_editB79 = QLineEdit(self)
        self.line_editB79.setFont(line_edit_font)
        self.line_editB79.setMaximumWidth(350) 
        self.line_editB36 = QLineEdit(self)
        self.line_editB36.setFont(line_edit_font)
        self.line_editB36.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB79,1)
        row.addWidget(self.line_editB36,1)
        form_layout.addRow(row)


        
        # self.line_editB80 = QLineEdit(self)
        self.line_editB80.setFont(line_edit_font)
        self.line_editB80.setMaximumWidth(350) 
        self.line_editB37 = QLineEdit(self)
        self.line_editB37.setFont(line_edit_font)
        self.line_editB37.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB80,1)
        row.addWidget(self.line_editB37,1)
        form_layout.addRow(row)



        # self.line_editB81 = QLineEdit(self)
        self.line_editB81.setFont(line_edit_font)
        self.line_editB81.setMaximumWidth(350) 
        self.line_editB38 = QLineEdit(self)
        self.line_editB38.setFont(line_edit_font)
        self.line_editB38.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB81,1)
        row.addWidget(self.line_editB38,1)
        form_layout.addRow(row)


        
        # self.line_editB82 = QLineEdit(self)
        self.line_editB82.setFont(line_edit_font)
        self.line_editB82.setMaximumWidth(350) 
        self.line_editB39 = QLineEdit(self)
        self.line_editB39.setFont(line_edit_font)
        self.line_editB39.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB82,1)
        row.addWidget(self.line_editB39,1)
        form_layout.addRow(row)



        # self.line_editB83 = QLineEdit(self)
        self.line_editB83.setFont(line_edit_font)
        self.line_editB83.setMaximumWidth(350) 
        self.line_editB40 = QLineEdit(self)
        self.line_editB40.setFont(line_edit_font)
        self.line_editB40.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB83,1)
        row.addWidget(self.line_editB40,1)
        form_layout.addRow(row)



        # self.line_editB84 = QLineEdit(self)
        self.line_editB84.setFont(line_edit_font)
        self.line_editB84.setMaximumWidth(350) 
        self.line_editB41 = QLineEdit(self)
        self.line_editB41.setFont(line_edit_font)
        self.line_editB41.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB84,1)
        row.addWidget(self.line_editB41,1)
        form_layout.addRow(row)



        # self.line_editB85 = QLineEdit(self)
        self.line_editB85.setFont(line_edit_font)
        self.line_editB85.setMaximumWidth(350) 
        self.line_editB42 = QLineEdit(self)
        self.line_editB42.setFont(line_edit_font)
        self.line_editB42.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB85,1)
        row.addWidget(self.line_editB42,1)
        form_layout.addRow(row)



        # self.line_editB86 = QLineEdit(self)
        self.line_editB86.setFont(line_edit_font)
        self.line_editB86.setMaximumWidth(350) 
        self.line_editB43 = QLineEdit(self)
        self.line_editB43.setFont(line_edit_font)
        self.line_editB43.setMaximumWidth(350)
        row = QHBoxLayout()
        row.addWidget(self.line_editB86,1)
        row.addWidget(self.line_editB43,1)
        form_layout.addRow(row)



        # self.line_editB87 = QLineEdit(self)
        self.line_editB87.setFont(line_edit_font)
        self.line_editB87.setMaximumWidth(350) 
        self.line_editB44 = QLineEdit(self)
        self.line_editB44.setFont(line_edit_font)
        self.line_editB44.setMaximumWidth(350)
        self.line_editB44.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB87,1)
        row.addWidget(self.line_editB44,1)
        form_layout.addRow(row)



        # self.line_editB88 = QLineEdit(self)
        self.line_editB88.setFont(line_edit_font)
        self.line_editB88.setMaximumWidth(350) 
        self.line_editB45 = QLineEdit(self)
        self.line_editB45.setFont(line_edit_font)
        self.line_editB45.setMaximumWidth(350)
        self.line_editB45.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB88,1)
        row.addWidget(self.line_editB45,1)
        form_layout.addRow(row)



        # self.line_editB89 = QLineEdit(self)
        self.line_editB89.setFont(line_edit_font)
        self.line_editB89.setMaximumWidth(350) 
        self.line_editB46 = QLineEdit(self)
        self.line_editB46.setFont(line_edit_font)
        self.line_editB46.setMaximumWidth(350)
        self.line_editB46.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.line_editB89,1)
        row.addWidget(self.line_editB46,1)
        form_layout.addRow(row)


        label2 = QLabel("")
        self.line_editB47 = QLineEdit(self)
        self.line_editB47.setFont(line_edit_font)
        self.line_editB47.setMaximumWidth(350)
        self.line_editB47.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label2,1)
        row.addWidget(self.line_editB47,1)
        form_layout.addRow(row)



        label2 = QLabel("")
        self.line_editB48 = QLineEdit(self)
        self.line_editB48.setFont(line_edit_font)
        self.line_editB48.setMaximumWidth(350)
        self.line_editB48.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label2,1)
        row.addWidget(self.line_editB48,1)
        form_layout.addRow(row)


        label2 = QLabel("")
        self.line_editB49 = QLineEdit(self)
        self.line_editB49.setFont(line_edit_font)
        self.line_editB49.setMaximumWidth(350)
        self.line_editB49.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label2,1)
        row.addWidget(self.line_editB49,1)
        form_layout.addRow(row)

        label2 = QLabel("")
        self.line_editB50 = QLineEdit(self)
        self.line_editB50.setFont(line_edit_font)
        self.line_editB50.setMaximumWidth(350)
        self.line_editB50.setStyleSheet("""
QLineEdit { background-color: #aaa;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label2,1)
        row.addWidget(self.line_editB50,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("PSYCHOSOCIAL")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        for i in range(28, 45):
            setattr(self, f'line_editF{i}', QLineEdit(self))
            getattr(self, f'line_editF{i}').setFont(line_edit_font)
            getattr(self, f'line_editF{i}').setMaximumWidth(350)
            if i<32:
                getattr(self, f'line_editF{i}').setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
            
            if i==39:
                self.line_editI39 = QComboBox()
                self.line_editI39.addItems(['Inadequate', 'Adequate', 'NONE'])
                self.line_editI39.setStyleSheet("""
QComboBox { background-color: yellow;
    }
""")
            elif i==40:
                self.line_editI40 = QComboBox()
                self.line_editI40.addItems(['Inadequate', 'Adequate', 'NONE'])
            else:

                setattr(self, f'line_editI{i}', QLineEdit(self))
                getattr(self, f'line_editI{i}').setFont(line_edit_font)
                getattr(self, f'line_editI{i}').setMaximumWidth(350)
            
            row = QHBoxLayout()
            row.addWidget(getattr(self, f'line_editF{i}'), 1)
            row.addWidget(getattr(self, f'line_editI{i}'), 1)
            form_layout.addRow(row)


        label1 = QLabel("")
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        self.line_editF48 = QLineEdit(self)
        self.line_editF48.setFont(line_edit_font)
        row = QHBoxLayout()
        row.addWidget(self.line_editF48,1)
        form_layout.addRow(row)



        label1 = QLabel("DIET")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.dietF50 = QComboBox()
        self.dietF50.addItems(['regular diet', 'NAS diet', 'low fat and low cholesterol diet', 'NAS, low fat, and low cholesterol diet', 'NAS, NCS, low fat, low cholesterol, and controlled carbohydrate diet', 'NAS, low fat, low cholesterol, and controlled carbohydrate diet', 'controlled carbohydrate diet', 'NAS, low fat, low acid, and low cholesterol diet', 'low acid diet', 'NAS, low acid, low fat, low cholesterol, and controlled carbohydrate diet', 'NAS and heart healthy diet', 'controlled carbohydrate diet', 'NAS, NCS, low fat, low cholesterol, high fiber, renal, and controlled carbohydrate diet'])
        row = QHBoxLayout()
        row.addWidget(self.dietF50,1)
        form_layout.addRow(row)

        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("BREATH SOUNDS:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("INCONTINANCE:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("DME:")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)


        self.line_editB53 = QLineEdit(self)
        self.line_editB53.setFont(line_edit_font)
        self.line_editB53.setMaximumWidth(200)
        self.incontinanceF53 = QComboBox()
        self.incontinanceF53.setMaximumWidth(200)
        self.incontinanceF53.addItems(['Incontinent', 'No Incontinance'])
        DME = QHBoxLayout()
        self.line_editI53 = QLineEdit(self)
        self.line_editI53.setMaximumWidth(100)
        self.line_editI53.setFont(line_edit_font)
        self.line_editJ53 = QLineEdit(self)
        self.line_editJ53.setMaximumWidth(100)
        self.line_editJ53.setFont(line_edit_font)
        DME.addWidget(self.line_editI53,1)
        DME.addWidget(self.line_editJ53,1)
        row = QHBoxLayout()
        row.addWidget(self.line_editB53,1)
        row.addWidget(self.incontinanceF53,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        label2 = QLabel("")
        DME = QHBoxLayout()
        self.line_editI54 = QLineEdit(self)
        self.line_editI54.setMaximumWidth(100)
        self.line_editI54.setFont(line_edit_font)
        self.line_editJ54 = QLineEdit(self)
        self.line_editJ54.setMaximumWidth(100)
        self.line_editJ54.setFont(line_edit_font)
        DME.addWidget(self.line_editI54,1)
        DME.addWidget(self.line_editJ54,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        label1 = QLabel("COUGH")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("COGNITIVE")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        DME = QHBoxLayout()
        self.line_editI55 = QLineEdit(self)
        self.line_editI55.setMaximumWidth(100)
        self.line_editI55.setFont(line_edit_font)
        self.line_editJ55 = QLineEdit(self)
        self.line_editJ55.setMaximumWidth(100)
        self.line_editJ55.setFont(line_edit_font)
        DME.addWidget(self.line_editI55,1)
        DME.addWidget(self.line_editJ55,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        self.line_editB56 = QLineEdit(self)
        self.line_editB56.setMaximumWidth(200)
        self.line_editB56.setFont(line_edit_font)
        INCO = QHBoxLayout()
        self.line_editF56 = QLineEdit(self)
        self.line_editF56.setMaximumWidth(100)
        self.line_editF56.setFont(line_edit_font)
        self.line_editG56 = QLineEdit(self)
        self.line_editG56.setMaximumWidth(100)
        self.line_editG56.setFont(line_edit_font)
        INCO.addWidget(self.line_editF56,1)
        INCO.addWidget(self.line_editG56,1)
        DME = QHBoxLayout()
        self.line_editI56 = QLineEdit(self)
        self.line_editI56.setMaximumWidth(100)
        self.line_editI56.setFont(line_edit_font)
        self.line_editJ56 = QLineEdit(self)
        self.line_editJ56.setMaximumWidth(100)
        self.line_editJ56.setFont(line_edit_font)
        DME.addWidget(self.line_editI56,1)
        DME.addWidget(self.line_editJ56,1)
        row = QHBoxLayout()
        row.addWidget(self.line_editB56,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        
        label1 = QLabel("")
        INCO = QHBoxLayout()
        self.line_editF57 = QLineEdit(self)
        self.line_editF57.setMaximumWidth(100)
        self.line_editF57.setFont(line_edit_font)
        self.line_editG57 = QLineEdit(self)
        self.line_editG57.setMaximumWidth(100)
        self.line_editG57.setFont(line_edit_font)
        INCO.addWidget(self.line_editF57,1)
        INCO.addWidget(self.line_editG57,1)
        DME = QHBoxLayout()
        self.line_editI57 = QLineEdit(self)
        self.line_editI57.setMaximumWidth(100)
        self.line_editI57.setFont(line_edit_font)
        self.line_editJ57 = QLineEdit(self)
        self.line_editJ57.setMaximumWidth(100)
        self.line_editJ57.setFont(line_edit_font)
        DME.addWidget(self.line_editI57,1)
        DME.addWidget(self.line_editJ57,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        label1 = QLabel("OXYGEN")
        label1.setStyleSheet("""
QLabel { background-color: red;
         color:white;
         font-weight: bold; }
""")
        INCO = QHBoxLayout()
        self.line_editF58 = QLineEdit(self)
        self.line_editF58.setMaximumWidth(100)
        self.line_editF58.setFont(line_edit_font)
        self.line_editG58 = QLineEdit(self)
        self.line_editG58.setMaximumWidth(100)
        self.line_editG58.setFont(line_edit_font)
        INCO.addWidget(self.line_editF58,1)
        INCO.addWidget(self.line_editG58,1)
        DME = QHBoxLayout()
        self.line_editI58 = QLineEdit(self)
        self.line_editI58.setMaximumWidth(100)
        self.line_editI58.setFont(line_edit_font)
        self.line_editJ58 = QLineEdit(self)
        self.line_editJ58.setMaximumWidth(100)
        self.line_editJ58.setFont(line_edit_font)
        DME.addWidget(self.line_editI58,1)
        DME.addWidget(self.line_editJ58,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        OXYGEN = QHBoxLayout()
        self.line_editB59 = QLineEdit(self)
        self.line_editB59.setMaximumWidth(100)
        self.line_editB59.setFont(line_edit_font)
        self.line_editC59 = QLineEdit(self)
        self.line_editC59.setMaximumWidth(100)
        self.line_editC59.setFont(line_edit_font)
        OXYGEN.addWidget(self.line_editB59,1)
        OXYGEN.addWidget(self.line_editC59,1)
        INCO = QHBoxLayout()
        self.line_editF59 = QLineEdit(self)
        self.line_editF59.setMaximumWidth(100)
        self.line_editF59.setFont(line_edit_font)
        self.line_editG59 = QLineEdit(self)
        self.line_editG59.setMaximumWidth(100)
        self.line_editG59.setFont(line_edit_font)
        INCO.addWidget(self.line_editF59,1)
        INCO.addWidget(self.line_editG59,1)
        DME = QHBoxLayout()
        self.line_editI59 = QLineEdit(self)
        self.line_editI59.setMaximumWidth(100)
        self.line_editI59.setFont(line_edit_font)
        self.line_editJ59 = QLineEdit(self)
        self.line_editJ59.setMaximumWidth(100)
        self.line_editJ59.setFont(line_edit_font)
        DME.addWidget(self.line_editI59,1)
        DME.addWidget(self.line_editJ59,1)
        row = QHBoxLayout()
        row.addLayout(OXYGEN,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        INCO = QHBoxLayout()
        self.line_editF60 = QLineEdit(self)
        self.line_editF60.setMaximumWidth(100)
        self.line_editF60.setFont(line_edit_font)
        self.line_editG60 = QLineEdit(self)
        self.line_editG60.setMaximumWidth(100)
        self.line_editG60.setFont(line_edit_font)
        INCO.addWidget(self.line_editF60,1)
        INCO.addWidget(self.line_editG60,1)
        DME = QHBoxLayout()
        self.line_editI60 = QLineEdit(self)
        self.line_editI60.setMaximumWidth(100)
        self.line_editI60.setFont(line_edit_font)
        self.line_editJ60 = QLineEdit(self)
        self.line_editJ60.setMaximumWidth(100)
        self.line_editJ60.setFont(line_edit_font)
        DME.addWidget(self.line_editI60,1)
        DME.addWidget(self.line_editJ60,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        label1 = QLabel("EDEMA")
        label1.setStyleSheet("""
QLabel { background-color: red;
         color:white;
         font-weight: bold; }
""")
        INCO = QHBoxLayout()
        self.line_editF61 = QLineEdit(self)
        self.line_editF61.setMaximumWidth(100)
        self.line_editF61.setFont(line_edit_font)
        self.line_editG61 = QLineEdit(self)
        self.line_editG61.setMaximumWidth(100)
        self.line_editG61.setFont(line_edit_font)
        INCO.addWidget(self.line_editF61,1)
        INCO.addWidget(self.line_editG61,1)
        DME = QHBoxLayout()
        self.line_editI61 = QLineEdit(self)
        self.line_editI61.setMaximumWidth(100)
        self.line_editI61.setFont(line_edit_font)
        self.line_editJ61 = QLineEdit(self)
        self.line_editJ61.setMaximumWidth(100)
        self.line_editJ61.setFont(line_edit_font)
        DME.addWidget(self.line_editI61,1)
        DME.addWidget(self.line_editJ61,1)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addLayout(INCO,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)



        self.edemaB62 = QComboBox()
        self.edemaB62.setMaximumWidth(200)
        self.edemaB62.addItems([None,'+1', '+2', '+3', '+4', 'non-pitting', 'localized', 'localized non-pitting', 'localized +1', 'localized +2', 'localized +3', 'localized +4'])
        label1 = QLabel("")
        DME = QHBoxLayout()
        self.line_editI62 = QLineEdit(self)
        self.line_editI62.setMaximumWidth(100)
        self.line_editI62.setFont(line_edit_font)
        self.line_editJ62 = QLineEdit(self)
        self.line_editJ62.setMaximumWidth(100)
        self.line_editJ62.setFont(line_edit_font)
        DME.addWidget(self.line_editI62,1)
        DME.addWidget(self.line_editJ62,1)
        row = QHBoxLayout()
        row.addWidget(self.edemaB62,1)
        row.addWidget(label1,1)
        row.addLayout(DME,1)
        form_layout.addRow(row)

        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        
        label1 = QLabel("SOB:")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("FALL RISK SCORE:")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("DIGOXIN:")
        label3.setStyleSheet("""
QLabel { background-color: red;
         color:white;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)



        self.sopB65 = QComboBox()
        self.sopB65.setMaximumWidth(200)
        self.sopB65.addItems(['Moderate', 'Minimal', '20 feet', 'NONE', 'At rest'])
        self.line_editF65 = QLineEdit(self)
        self.line_editF65.setMaximumWidth(200)
        self.line_editF65.setFont(line_edit_font)
        self.line_editF65.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editI65 = QLineEdit(self)
        self.line_editI65.setMaximumWidth(200)
        self.line_editI65.setFont(line_edit_font)
        row = QHBoxLayout()
        row.addWidget(self.sopB65,1)
        row.addWidget(self.line_editF65,1)
        row.addWidget(self.line_editI65,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)
        
        label1 = QLabel("BLEEDING")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.line_editI70 = QLineEdit(self)
        self.line_editI70.setFont(line_edit_font)
        form_layout.addRow("MEDICATION NAME               ",self.line_editI70)
        label = form_layout.labelForField(self.line_editI70)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("CREAM")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        self.line_editI74 = QLineEdit(self)
        self.line_editI74.setFont(line_edit_font)
        form_layout.addRow("CREAM ORDER                      ",self.line_editI74)
        label = form_layout.labelForField(self.line_editI74)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")

        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("CLONIDINE")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.line_editI78 = QLineEdit(self)
        self.line_editI78.setFont(line_edit_font)
        form_layout.addRow("CLONIDINE ORDER                 ",self.line_editI78)
        label = form_layout.labelForField(self.line_editI78)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("SQ INJECTIONS")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.line_editI82 = QLineEdit(self)
        self.line_editI82.setFont(line_edit_font)
        form_layout.addRow("SQ INJECTION ORDER            ",self.line_editI82)
        label = form_layout.labelForField(self.line_editI82)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")

        self.line_editI83 = QLineEdit(self)
        self.line_editI83.setFont(line_edit_font)
        form_layout.addRow("MEDICATION NAME               ",self.line_editI83)
        label = form_layout.labelForField(self.line_editI83)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("ANTIBIOTIC")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.line_editI87 = QLineEdit(self)
        self.line_editI87.setFont(line_edit_font)
        form_layout.addRow("ANTIBIOTIC ORDER               ",self.line_editI87)
        label = form_layout.labelForField(self.line_editI87)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editI88 = QLineEdit(self)
        self.line_editI88.setFont(line_edit_font)
        form_layout.addRow("ANTIBIOTIC NAME                 ",self.line_editI88)
        label = form_layout.labelForField(self.line_editI88)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("EYE MEDICATIONS")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: red;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("Glaucoma")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("Conjunct")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        form_layout.addRow(row)

        self.line_editI93 = QLineEdit(self)
        self.line_editI93.setFont(line_edit_font)
        self.line_editJ93 = QLineEdit(self)
        self.line_editJ93.setFont(line_edit_font)
        row = QHBoxLayout()
        row.addWidget(self.line_editI93,1)
        row.addWidget(self.line_editJ93,1)
        form_layout.addRow("EYE MEDICATION ORDER      ",row)
        label = form_layout.labelForField(row)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")

        self.line_editI94 = QLineEdit(self)
        self.line_editI94.setFont(line_edit_font)
        self.line_editJ94 = QLineEdit(self)
        self.line_editJ94.setFont(line_edit_font)
        row = QHBoxLayout()
        row.addWidget(self.line_editI94,1)
        row.addWidget(self.line_editJ94,1)
        form_layout.addRow("MEDICATION NAME               ",row)
        label = form_layout.labelForField(row)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("DIABETES MELLITUS")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)

        label1 = QLabel("INSULIN TYPE")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("TYPE")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)


        self.line_editB101 = QLineEdit(self)
        
        self.line_editB101.setFont(line_edit_font)
        self.typeG101 = QComboBox()
        self.typeG101.addItems(['Syringe', 'Pen'])
        row = QHBoxLayout()
        row.addWidget(self.line_editB101,1)
        row.addWidget(self.typeG101,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("WOUND")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("                 ")
        label2 = QLabel("                 ")
        self.cmF105 = QComboBox()
        self.cmF105.setMaximumWidth(130)
        self.cmF105.addItems(['cm', 'in'])
        self.cmF105.setStyleSheet("""
QComboBox { background-color: yellow;
    }
""")

        label4 = QLabel("")
        label5 = QLabel("")
        label6 = QLabel("")
        label7 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(self.cmF105,1)
        row.addWidget(label4,1)
        row.addWidget(label5,1)
        row.addWidget(label6,1)
        row.addWidget(label7,1)
        form_layout.addRow(row)




        label1 = QLabel("WOUND #   ")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label2 = QLabel("#1")
        label2.setStyleSheet("""
QLabel { background-color: white;
          }
""")
        label3 = QLabel("#2")
        label3.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        label4 = QLabel("#3")
        label4.setStyleSheet("""
QLabel { background-color: white;
          }
""")
        label5 = QLabel("#4")
        label5.setStyleSheet("""
QLabel { background-color: white;
          }
""")
        label6 = QLabel("#5")
        label6.setStyleSheet("""
QLabel { background-color: white;
          }
""")
        label7 = QLabel("#6")
        label7.setStyleSheet("""
QLabel { background-color: white;
          }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        row.addWidget(label4,1)
        row.addWidget(label5,1)
        row.addWidget(label6,1)
        row.addWidget(label7,1)
        form_layout.addRow(row)



        label1 = QLabel("LOCATION")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editE107 = QLineEdit(self)
        self.line_editE107.setFont(line_edit_font)
        self.line_editE107.setMaximumWidth(80)
        self.line_editF107 = QLineEdit(self)
        self.line_editF107.setFont(line_edit_font)
        self.line_editF107.setMaximumWidth(80)
        self.line_editG107 = QLineEdit(self)
        self.line_editG107.setFont(line_edit_font)
        self.line_editG107.setMaximumWidth(80)
        self.line_editH107 = QLineEdit(self)
        self.line_editH107.setFont(line_edit_font)
        self.line_editH107.setMaximumWidth(80)
        self.line_editI107 = QLineEdit(self)
        self.line_editI107.setFont(line_edit_font)
        self.line_editI107.setMaximumWidth(80)
        self.line_editJ107 = QLineEdit(self)
        self.line_editJ107.setFont(line_edit_font)
        self.line_editJ107.setMaximumWidth(80)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE107,1)
        row.addWidget(self.line_editF107,1)
        row.addWidget(self.line_editG107,1)
        row.addWidget(self.line_editH107,1)
        row.addWidget(self.line_editI107,1)
        row.addWidget(self.line_editJ107,1)
        form_layout.addRow(row)



        label1 = QLabel("SIZE")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editE108 = QLineEdit(self)
        self.line_editE108.setFont(line_edit_font)
        self.line_editE108.setMaximumWidth(80)
        self.line_editF108 = QLineEdit(self)
        self.line_editF108.setFont(line_edit_font)
        self.line_editF108.setMaximumWidth(80)
        self.line_editG108 = QLineEdit(self)
        self.line_editG108.setFont(line_edit_font)
        self.line_editG108.setMaximumWidth(80)
        self.line_editH108 = QLineEdit(self)
        self.line_editH108.setFont(line_edit_font)
        self.line_editH108.setMaximumWidth(80)
        self.line_editI108 = QLineEdit(self)
        self.line_editI108.setFont(line_edit_font)
        self.line_editI108.setMaximumWidth(80)
        self.line_editJ108 = QLineEdit(self)
        self.line_editJ108.setFont(line_edit_font)
        self.line_editJ108.setMaximumWidth(80)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE108,1)
        row.addWidget(self.line_editF108,1)
        row.addWidget(self.line_editG108,1)
        row.addWidget(self.line_editH108,1)
        row.addWidget(self.line_editI108,1)
        row.addWidget(self.line_editJ108,1)
        form_layout.addRow(row)


        label1 = QLabel("HEIGHT")
        label1.setStyleSheet("""
QLabel { background-color: yellow;
         font-weight: bold; }
""")
        self.line_editE109 = QLineEdit(self)
        self.line_editE109.setFont(line_edit_font)
        self.line_editE109.setMaximumWidth(80)
        self.line_editE109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editF109 = QLineEdit(self)
        self.line_editF109.setFont(line_edit_font)
        self.line_editF109.setMaximumWidth(80)
        self.line_editF109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editG109 = QLineEdit(self)
        self.line_editG109.setFont(line_edit_font)
        self.line_editG109.setMaximumWidth(80)
        self.line_editG109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editH109 = QLineEdit(self)
        self.line_editH109.setFont(line_edit_font)
        self.line_editH109.setMaximumWidth(80)
        self.line_editH109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editI109 = QLineEdit(self)
        self.line_editI109.setFont(line_edit_font)
        self.line_editI109.setMaximumWidth(80)
        self.line_editI109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editJ109 = QLineEdit(self)
        self.line_editJ109.setFont(line_edit_font)
        self.line_editJ109.setMaximumWidth(80)
        self.line_editJ109.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE109,1)
        row.addWidget(self.line_editF109,1)
        row.addWidget(self.line_editG109,1)
        row.addWidget(self.line_editH109,1)
        row.addWidget(self.line_editI109,1)
        row.addWidget(self.line_editJ109,1)
        form_layout.addRow(row)


        label1 = QLabel("LENGTH")
        label1.setStyleSheet("""
QLabel { background-color: yellow;
         font-weight: bold; }
""")
        self.line_editE110 = QLineEdit(self)
        self.line_editE110.setFont(line_edit_font)
        self.line_editE110.setMaximumWidth(80)
        self.line_editE110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editF110 = QLineEdit(self)
        self.line_editF110.setFont(line_edit_font)
        self.line_editF110.setMaximumWidth(80)
        self.line_editF110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editG110 = QLineEdit(self)
        self.line_editG110.setFont(line_edit_font)
        self.line_editG110.setMaximumWidth(80)
        self.line_editG110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editH110 = QLineEdit(self)
        self.line_editH110.setFont(line_edit_font)
        self.line_editH110.setMaximumWidth(80)
        self.line_editH110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editI110 = QLineEdit(self)
        self.line_editI110.setFont(line_edit_font)
        self.line_editI110.setMaximumWidth(80)
        self.line_editI110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editJ110 = QLineEdit(self)
        self.line_editJ110.setFont(line_edit_font)
        self.line_editJ110.setMaximumWidth(80)
        self.line_editJ110.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE110,1)
        row.addWidget(self.line_editF110,1)
        row.addWidget(self.line_editG110,1)
        row.addWidget(self.line_editH110,1)
        row.addWidget(self.line_editI110,1)
        row.addWidget(self.line_editJ110,1)
        form_layout.addRow(row)




        label1 = QLabel("WIDTH")
        label1.setStyleSheet("""
QLabel { background-color: yellow;
         font-weight: bold; }
""")
        self.line_editE111 = QLineEdit(self)
        self.line_editE111.setFont(line_edit_font)
        self.line_editE111.setMaximumWidth(80)
        self.line_editE111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editF111 = QLineEdit(self)
        self.line_editF111.setFont(line_edit_font)
        self.line_editF111.setMaximumWidth(80)
        self.line_editF111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editG111 = QLineEdit(self)
        self.line_editG111.setFont(line_edit_font)
        self.line_editG111.setMaximumWidth(80)
        self.line_editG111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editH111 = QLineEdit(self)
        self.line_editH111.setFont(line_edit_font)
        self.line_editH111.setMaximumWidth(80)
        self.line_editH111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editI111 = QLineEdit(self)
        self.line_editI111.setFont(line_edit_font)
        self.line_editI111.setMaximumWidth(80)
        self.line_editI111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        self.line_editJ111 = QLineEdit(self)
        self.line_editJ111.setFont(line_edit_font)
        self.line_editJ111.setMaximumWidth(80)
        self.line_editJ111.setStyleSheet("""
QLineEdit { background-color: yellow;
    }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE111,1)
        row.addWidget(self.line_editF111,1)
        row.addWidget(self.line_editG111,1)
        row.addWidget(self.line_editH111,1)
        row.addWidget(self.line_editI111,1)
        row.addWidget(self.line_editJ111,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        
        label1 = QLabel("                                 ")
        label2 = QLabel("SITE#1")
        label2.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        label3 = QLabel("SITE#2")
        label3.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        label4 = QLabel("SITE#3")
        label4.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        label5 = QLabel("SITE#4")
        label5.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        label6 = QLabel("SITE#5")
        label6.setStyleSheet("""
QLabel { background-color: white;
         }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        row.addWidget(label4,1)
        row.addWidget(label5,1)
        row.addWidget(label6,1)
        form_layout.addRow(row)


        label1 = QLabel("                                 ")
        self.line_editE115 = QLineEdit(self)
        self.line_editE115.setFont(line_edit_font)
        self.line_editE115.setMaximumWidth(90)
        self.line_editF115 = QLineEdit(self)
        self.line_editF115.setFont(line_edit_font)
        self.line_editF115.setMaximumWidth(90)
        self.line_editG115 = QLineEdit(self)
        self.line_editG115.setFont(line_edit_font)
        self.line_editG115.setMaximumWidth(90)
        self.line_editH115 = QLineEdit(self)
        self.line_editH115.setFont(line_edit_font)
        self.line_editH115.setMaximumWidth(90)
        self.line_editI115 = QLineEdit(self)
        self.line_editI115.setFont(line_edit_font)
        self.line_editI115.setMaximumWidth(90)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE115,1)
        row.addWidget(self.line_editF115,1)
        row.addWidget(self.line_editG115,1)
        row.addWidget(self.line_editH115,1)
        row.addWidget(self.line_editI115,1)
        form_layout.addRow(row)



        label1 = QLabel("WOUND CARE ORDER")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editE116 = QLineEdit(self)
        self.line_editE116.setFont(line_edit_font)
        self.line_editE116.setMaximumWidth(90)
        self.line_editF116 = QLineEdit(self)
        self.line_editF116.setFont(line_edit_font)
        self.line_editF116.setMaximumWidth(90)
        self.line_editG116 = QLineEdit(self)
        self.line_editG116.setFont(line_edit_font)
        self.line_editG116.setMaximumWidth(90)
        self.line_editH116 = QLineEdit(self)
        self.line_editH116.setFont(line_edit_font)
        self.line_editH116.setMaximumWidth(90)
        self.line_editI116 = QLineEdit(self)
        self.line_editI116.setFont(line_edit_font)
        self.line_editI116.setMaximumWidth(90)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE116,1)
        row.addWidget(self.line_editF116,1)
        row.addWidget(self.line_editG116,1)
        row.addWidget(self.line_editH116,1)
        row.addWidget(self.line_editI116,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        self.line_editE118 = QLineEdit(self)
        self.line_editE118.setFont(line_edit_font)
        form_layout.addRow("DERMATITIS LOCATION         ",self.line_editE118)
        label = form_layout.labelForField(self.line_editE118)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("DIALYSIS")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("SHUNT LOCATION & TYPE     ")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editE122 = QLineEdit(self)
        self.line_editE122.setFont(line_edit_font)
        self.line_editE122.setMaximumWidth(135)
        self.SHUNTtypeG122 = QComboBox()
        self.SHUNTtypeG122.setMaximumWidth(135)
        self.SHUNTtypeG122.addItems(['SHUNT', 'Catheter'])
        self.SHUNTlocaI122 = QComboBox()
        self.SHUNTlocaI122.setMaximumWidth(135)
        self.SHUNTlocaI122.addItems(['left', 'right'])
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editE122,1)
        row.addWidget(self.SHUNTtypeG122,1)
        row.addWidget(self.SHUNTlocaI122,1)
        form_layout.addRow(row)


        self.line_editE123 = QLineEdit(self)
        self.line_editE123.setFont(line_edit_font)
        form_layout.addRow("DIALYSIS DAYS                   ",self.line_editE123)
        label = form_layout.labelForField(self.line_editE123)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("IV")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        self.line_editE127 = QLineEdit(self)
        self.line_editE127.setFont(line_edit_font)
        form_layout.addRow("IV LOCATION                         ",self.line_editE127)
        label = form_layout.labelForField(self.line_editE127)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE128 = QLineEdit(self)
        self.line_editE128.setFont(line_edit_font)
        form_layout.addRow("IV MEDICATION ORDER         ",self.line_editE128)
        label = form_layout.labelForField(self.line_editE128)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE129 = QLineEdit(self)
        self.line_editE129.setFont(line_edit_font)
        form_layout.addRow("IV MEDICATION NAME           ",self.line_editE129)
        label = form_layout.labelForField(self.line_editE129)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE130 = QLineEdit(self)
        self.line_editE130.setFont(line_edit_font)
        form_layout.addRow("IV SITE CARE ORDER             ",self.line_editE130)
        label = form_layout.labelForField(self.line_editE130)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("ROC")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        self.line_editE134 = QLineEdit(self)
        self.line_editE134.setFont(line_edit_font)
        form_layout.addRow("SOC Date                             ",self.line_editE134)
        label = form_layout.labelForField(self.line_editE134)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE135 = QLineEdit(self)
        self.line_editE135.setFont(line_edit_font)
        form_layout.addRow("Admission Date                   ",self.line_editE135)
        label = form_layout.labelForField(self.line_editE135)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE136 = QLineEdit(self)
        self.line_editE136.setFont(line_edit_font)
        form_layout.addRow("DC Date                               ",self.line_editE136)
        label = form_layout.labelForField(self.line_editE136)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE137 = QLineEdit(self)
        self.line_editE137.setFont(line_edit_font)
        form_layout.addRow("Weeks Remaining                ",self.line_editE137)
        label = form_layout.labelForField(self.line_editE137)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE138 = QLineEdit(self)
        self.line_editE138.setFont(line_edit_font)
        form_layout.addRow("Days Remaining                   ",self.line_editE138)
        label = form_layout.labelForField(self.line_editE138)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE139 = QLineEdit(self)
        self.line_editE139.setFont(line_edit_font)
        form_layout.addRow("Hospital Name                      ",self.line_editE139)
        label = form_layout.labelForField(self.line_editE139)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")
        self.line_editE140 = QLineEdit(self)
        self.line_editE140.setFont(line_edit_font)
        form_layout.addRow("ROC Date                             ",self.line_editE140)
        label = form_layout.labelForField(self.line_editE140)
        if label:
            label.setStyleSheet("background-color: orange; font-weight: bold;")



        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)


        
        label1 = QLabel("EDEMA")
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("""
QLabel { background-color: purple;
         color: white;
         font-size: 16px;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)



        label1 = QLabel("")
        label2 = QLabel("NON-PITTING")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label3 = QLabel("1")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label4 = QLabel("2")
        label4.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label5 = QLabel("3")
        label5.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label6 = QLabel("4")
        label6.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        row.addWidget(label4,1)
        row.addWidget(label5,1)
        row.addWidget(label6,1)
        form_layout.addRow(row)



        label1 = QLabel("RIGHT")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editC145 = QLineEdit(self)
        self.line_editC145.setFont(line_edit_font)
        self.line_editC145.setMaximumWidth(100)
        self.line_editD145 = QLineEdit(self)
        self.line_editD145.setFont(line_edit_font)
        self.line_editD145.setMaximumWidth(100)
        self.line_editE145 = QLineEdit(self)
        self.line_editE145.setFont(line_edit_font)
        self.line_editE145.setMaximumWidth(100)
        self.line_editF145 = QLineEdit(self)
        self.line_editF145.setFont(line_edit_font)
        self.line_editF145.setMaximumWidth(100)
        self.line_editG145 = QLineEdit(self)
        self.line_editG145.setFont(line_edit_font)
        self.line_editG145.setMaximumWidth(100)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editC145,1)
        row.addWidget(self.line_editD145,1)
        row.addWidget(self.line_editE145,1)
        row.addWidget(self.line_editF145,1)
        row.addWidget(self.line_editG145,1)
        form_layout.addRow(row)



        
        label1 = QLabel("LEFT")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editC146 = QLineEdit(self)
        self.line_editC146.setFont(line_edit_font)
        self.line_editC146.setMaximumWidth(100)
        self.line_editD146 = QLineEdit(self)
        self.line_editD146.setFont(line_edit_font)
        self.line_editD146.setMaximumWidth(100)
        self.line_editE146 = QLineEdit(self)
        self.line_editE146.setFont(line_edit_font)
        self.line_editE146.setMaximumWidth(100)
        self.line_editF146 = QLineEdit(self)
        self.line_editF146.setFont(line_edit_font)
        self.line_editF146.setMaximumWidth(100)
        self.line_editG146 = QLineEdit(self)
        self.line_editG146.setFont(line_edit_font)
        self.line_editG146.setMaximumWidth(100)
        row = QHBoxLayout()
        row.addWidget(label1,1)
        row.addWidget(self.line_editC146,1)
        row.addWidget(self.line_editD146,1)
        row.addWidget(self.line_editE146,1)
        row.addWidget(self.line_editF146,1)
        row.addWidget(self.line_editG146,1)
        form_layout.addRow(row)



        self.edemaB147 = QComboBox()
        self.edemaB147.setMaximumWidth(100)
        self.edemaB147.addItems(['Normal', 'Localized'])
        self.edemaB147.setStyleSheet("""
QComboBox { background-color: yellow;
    }
""")
        row = QHBoxLayout()
        row.addWidget(self.edemaB147,1)
        form_layout.addRow(row)


        label1 = QLabel("")
        row = QHBoxLayout()
        row.addWidget(label1,1)
        form_layout.addRow(row)




        label1 = QLabel("")
        label1.setStyleSheet("""
QLabel { background-color: white;
         font-weight: bold; }
""")
        label2 = QLabel("RIGHT")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")

        label3 = QLabel("LEFT")
        label3.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        label4 = QLabel("")
        label4.setStyleSheet("""
QLabel { background-color: white;
         font-weight: bold; }
""")
        row = QHBoxLayout() 
        row.addWidget(label1,1)
        row.addWidget(label2,1)
        row.addWidget(label3,1)
        row.addWidget(label4,1)
        form_layout.addRow(row)



        
        label1 = QLabel("ANTERIOR")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editC150 = QLineEdit(self)
        self.line_editC150.setFont(line_edit_font)
        # self.line_editC150.setMaximumWidth(100)
        self.line_editD150 = QLineEdit(self)
        self.line_editD150.setFont(line_edit_font)
        # self.line_editD150.setMaximumWidth(100)
        label2 = QLabel("")
        label2.setStyleSheet("""
QLabel { background-color: white;
         font-weight: bold; }
""")
        row = QHBoxLayout() 
        row.addWidget(label1,1)
        row.addWidget(self.line_editC150,1)
        row.addWidget(self.line_editD150,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)


        
        label1 = QLabel("POSTERIOR")
        label1.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        self.line_editC151 = QLineEdit(self)
        self.line_editC151.setFont(line_edit_font)
        # self.line_editC150.setMaximumWidth(100)
        self.line_editD151 = QLineEdit(self)
        self.line_editD151.setFont(line_edit_font)
        # self.line_editD150.setMaximumWidth(100)
        label2 = QLabel("UPPER")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout() 
        row.addWidget(label1,1)
        row.addWidget(self.line_editC151,1)
        row.addWidget(self.line_editD151,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)


        
        label1 = QLabel("")
        label1.setStyleSheet("""
QLabel { background-color: white;
         font-weight: bold; }
""")
        self.line_editC152 = QLineEdit(self)
        self.line_editC152.setFont(line_edit_font)
        # self.line_editC150.setMaximumWidth(100)
        self.line_editD152 = QLineEdit(self)
        self.line_editD152.setFont(line_edit_font)
        # self.line_editD150.setMaximumWidth(100)
        label2 = QLabel("LOWER")
        label2.setStyleSheet("""
QLabel { background-color: orange;
         font-weight: bold; }
""")
        row = QHBoxLayout() 
        row.addWidget(label1,1)
        row.addWidget(self.line_editC152,1)
        row.addWidget(self.line_editD152,1)
        row.addWidget(label2,1)
        form_layout.addRow(row)


        
        inputs_widget.setLayout(form_layout)
        scroll.setWidget(inputs_widget)

        left_layout.addWidget(scroll)

        # File display labels
        file_labels_layout = QHBoxLayout()
        self.pdf_label = QLabel("")
        self.excel_label = QLabel("")
        file_labels_layout.addWidget(self.pdf_label)
        file_labels_layout.addWidget(self.excel_label)
        left_layout.addLayout(file_labels_layout)


        # File choose buttons
        choose_buttons_layout = QHBoxLayout()
        btn_choose_pdf = QPushButton('Choose PDF', self)
        # btn_choose_pdf.setFixedSize(280, 40)  # Adjust button size
        btn_choose_pdf.clicked.connect(self.choose_pdf)
        btn_choose_excel = QPushButton('Choose Excel', self)
        # btn_choose_excel.setFixedSize(280, 40)  # Adjust button size
        btn_choose_excel.clicked.connect(self.choose_excel)
        choose_buttons_layout.addWidget(btn_choose_pdf)
        choose_buttons_layout.addWidget(btn_choose_excel)
        left_layout.addLayout(choose_buttons_layout)

        central_layout.addLayout(left_layout)

        middle_layout = QVBoxLayout()


        title1 = QLabel("POC:")
        title_layout1 = QHBoxLayout()
        btn_copy1 = QPushButton('ِCopy', self)
        # btn_copy1.setFixedSize(280, 40)  # Adjust button size
        btn_copy1.clicked.connect(self.copy_text1)
        title_layout1.addWidget(title1)
        title_layout1.addWidget(btn_copy1)
        middle_layout.addLayout(title_layout1)
        self.text_area1 = QTextEdit()
        middle_layout.addWidget(self.text_area1, stretch=15)


        title2 = QLabel("SUPPLY:")
        title_layout2 = QHBoxLayout()
        btn_copy2 = QPushButton('ِCopy', self)
        # btn_copy1.setFixedSize(280, 40)  # Adjust button size
        btn_copy2.clicked.connect(self.copy_text2)
        title_layout2.addWidget(title2)
        title_layout2.addWidget(btn_copy2)
        middle_layout.addLayout(title_layout2)
        self.text_area2 = QTextEdit()
        middle_layout.addWidget(self.text_area2, stretch=4)


        title3 = QLabel("PRECAUTIONS:")
        title_layout3 = QHBoxLayout()
        btn_copy3 = QPushButton('ِCopy', self)
        # btn_copy1.setFixedSize(280, 40)  # Adjust button size
        btn_copy3.clicked.connect(self.copy_text3)
        title_layout3.addWidget(title3)
        title_layout3.addWidget(btn_copy3)
        middle_layout.addLayout(title_layout3)
        self.text_area3 = QTextEdit()
        middle_layout.addWidget(self.text_area3,stretch=5)
        middle_layout.addWidget(self.progress_bar)

        # Middle: Restart and Update buttons 
        restart_update_layout = QHBoxLayout()
        # Restart button
        btn_restart = QPushButton('Restart', self)
        # btn_restart.setFixedSize(280, 40)  # Adjust button size
        btn_restart.clicked.connect(self.restart)


        # Update button
        btn_update = QPushButton('Update', self)
        # btn_update.setFixedSize(280, 40)  # Adjust button size
        btn_update.clicked.connect(self.update)
        restart_update_layout.addWidget(btn_restart)
        restart_update_layout.addWidget(btn_update)
        middle_layout.addLayout(restart_update_layout)

        central_layout.addLayout(middle_layout)


        # Right Side: Resulting paragraph area
        right_layout = QVBoxLayout()

        self.pdf_viewer = PDFViewer()
        right_layout.addWidget(self.pdf_viewer)

        # Scan and Settings buttons
        zoom_layout = QHBoxLayout()
        btn_zoom_in = QPushButton('Zoom in', self)
        # btn_zoom_in.setFixedSize(280, 40)  # Adjust button size
        btn_zoom_in.clicked.connect(self.pdf_viewer.zoom_in)
        btn_zoom_out = QPushButton('Zoom out', self)
        # btn_zoom_out.setFixedSize(280, 40)  # Adjust button size
        btn_zoom_out.clicked.connect(self.pdf_viewer.zoom_out)
        zoom_layout.addWidget(btn_zoom_out)
        zoom_layout.addWidget(btn_zoom_in)

        right_layout.addLayout(zoom_layout)

        # Scan and Settings buttons
        scan_settings_layout = QHBoxLayout()
        btn_scan = QPushButton('Scan', self)
        # btn_scan.setFixedSize(280, 40)  # Adjust button size
        btn_scan.clicked.connect(self.start_scanning)
        btn_settings = QPushButton('Settings', self)
        # btn_settings.setFixedSize(280, 40)  # Adjust button size
        btn_settings.clicked.connect(self.open_settings)
        scan_settings_layout.addWidget(btn_scan)
        scan_settings_layout.addWidget(btn_settings)
        
        right_layout.addLayout(scan_settings_layout)


        central_layout.addLayout(right_layout)
        main_layout.addLayout(central_layout)
        self.setLayout(main_layout)

    def copy_text1(self):
        self.text_area1.selectAll()
        self.text_area1.copy()
    def copy_text2(self):
        self.text_area2.selectAll()
        self.text_area2.copy()
    def copy_text3(self):
        self.text_area3.selectAll()
        self.text_area3.copy()

    def apply_dark_theme(self):
        # Set dark palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(30, 30, 30))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        # Set dark style sheet
        self.setStyleSheet("""

            QComboBox {
                background: white;
                border-radius: 5px;
                padding: 5px;
            }


            QComboBox::drop-down {
                border-radius: 5px;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: url(arrow.png); /* Replace with your own image */
                width: 10px; /* or any size that fits well in your combo box */
                height: 10px;
                border-radius: 5px;
            }

            QComboBox::down-arrow:on { /* shift the arrow when popup is open */
                border-radius: 5px;
                top: 1px;
                left: 1px;
            }

            QProgressBar {
                border: 1px solid #303030;
                border-radius: 8px;
                background-color: #232323;
            }

            QProgressBar::chunk {
                border-radius: 8px;
                background-color: #d47f00;
                width: 2px;
            }

            QTextEdit {
                border: 1px solid #303030;
                background-color: #232323;
                color: #707070;
                border-radius: 16px;

            }
            QScrollArea {
                border: 1px solid #303030;
                background-color: #232323;
            }

            QLabel{
                color:#707070;
            }

            QPushButton {
                background-color: #232323;
                color:#707070;
                padding: 5px;
                border-radius: 6px;
                outline: none;
                border: 1px solid #303030;
            }
            QLineEdit {
                padding: 5px;
                border-radius: 5px;
                outline: none;
            }
            QLabel {
                padding: 5px;
                border-radius: 5px;
                outline: none;
            }

            QPushButton:hover{
                background-color: #161616;
                color:#707070;
                padding: 5px;
                border-radius: 6px;
                outline: none;
                border: 1px solid #161616;
            }

            QPushButton:pressed {
                background-color: #d47f00;
                
                padding: 5px;
                border-radius: 6px;
                outline: none;
            }
        """)

    def choose_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose PDF", "", "PDF Files (*.pdf);;All Files (*)")
        if file_name:
            self.pdf_selected = True
            self.pdf_path = file_name
            self.pdf_label.setText(os.path.basename(file_name))
            
            # Load the PDF into the viewer
            self.load_into_viewer(file_name)

    def load_into_viewer(self, file_path):
        try:
            self.choose_pdf_thread = choosePdfThread(self.pdf_viewer, file_path)

            # Connect signals to slots for updates
            self.choose_pdf_thread.progress_signal.connect(self.update_progress)
            self.choose_pdf_thread.finished_loading.connect(lambda: print("PDF Loaded"))
            self.choose_pdf_thread.finished_rendering.connect(lambda: print("PDF Rendered"))
            self.choose_pdf_thread.request_display.connect(self.display_in_main_thread)

            self.choose_pdf_thread.start()

        except Exception as e:
            print(f"Error loading the PDF into the viewer: {e}")

    def display_in_main_thread(self):
        self.pdf_viewer.display_pdf()
        print("PDF Displayed")

    def choose_excel(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Excel", "", "Excel Files (*.xls *.xlsx *.xlsm);;All Files (*)")
        if file_name:
            self.excel_selected = True
            self.excel_path = file_name
            self.excel_label.setText(os.path.basename(file_name))

            self.choose_excel_thread = chooseExcelThread(self,file_name)

            # Connect signals to slots for updates
            self.choose_excel_thread.progress_signal.connect(self.update_progress)

            self.choose_excel_thread.start()
            self.choose_excel_thread.L4_signal.connect(self.set_paragraph_text1)
            self.choose_excel_thread.O4_signal.connect(self.set_paragraph_text2)
            self.choose_excel_thread.R4_signal.connect(self.set_paragraph_text3)

    def start_scanning(self):
        if not self.pdf_selected or not self.excel_selected or self.restarting or self.updating:
            # self.text_area.setText("Please choose both a PDF and an Excel file before scanning.")
            return
        self.scaning=True
        # Initialize and start the scanning thread
        self.scan_thread = ScanThread(self)
        self.scan_thread.progress_signal.connect(self.update_progress)
        self.scan_thread.start()
        self.scan_thread.L4_signal.connect(self.set_paragraph_text1)
        self.scan_thread.O4_signal.connect(self.set_paragraph_text2)
        self.scan_thread.R4_signal.connect(self.set_paragraph_text3)
        self.scan_thread.done.connect(self.done_scaning)
        
        # self.scan_thread.paragraph_signal.connect(self.set_paragraph_text)

    def done_scaning(self):
        self.scaning=False

    def update_progress(self, value):
        # Set the animation properties
        self.progress_animation.setDuration(500)  # Half a second for animation
        self.progress_animation.setStartValue(self.progress_bar.value())
        self.progress_animation.setEndValue(value)
        self.progress_animation.start()

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        if settings_dialog.exec_():
            self.denoising = settings_dialog.chk_denoising.isChecked()
            self.sharpen = settings_dialog.chk_sharpen.isChecked()
            self.correcting_skew = settings_dialog.chk_correcting_skew.isChecked()
            self.show_imgs = settings_dialog.chk_show_imgs.isChecked()

    def set_paragraph_text1(self, value):
        self.text_area1.setText(value)
    def set_paragraph_text2(self, value):
        self.text_area2.setText(value)
    def set_paragraph_text3(self, value):
        self.text_area3.setText(value)

    def restart(self):
        if not self.excel_selected or self.scaning or self.updating:
            return
        self.restarting=True
        # Initialize and start the scanning thread
        self.restart_thread = RestartThread(self)
        self.restart_thread.progress_signal.connect(self.update_progress)
        self.restart_thread.start()
        self.restart_thread.L4_signal.connect(self.set_paragraph_text1)
        self.restart_thread.O4_signal.connect(self.set_paragraph_text2)
        self.restart_thread.R4_signal.connect(self.set_paragraph_text3)
        self.restart_thread.done.connect(self.done_restarting)
        
        # excel_inputs={
        # 'I33': None,
        # 'I43': None,
        # 'I44': None,
        # 'I32': None,
        # 'G56': None,
        # 'G57': None,
        # 'G58': None,
        # 'G59': None,
        # 'G60': None,
        # 'G61': None,
        # 'I30': None,
        # 'F65': 6,
        # 'F53': 'Incontinent',
        # 'F14': None,
        # 'F15': None,
        # 'F16': None,
        # 'F17': None,
        # 'F18': None,
        # 'F19': None,
        # 'F20': None,
        # 'F21': None,
        # 'F22': None,
        # 'H20': None,
        # 'B65': 'Moderate',
        # 'I8': 65,
        # 'H8': None,
        # 'J55': None,
        # 'J53': None,
        # 'J54': None,
        # 'J57': None,
        # 'J56': None,
        # 'J59': None,
        # 'J58': None,
        # 'B14': 2,
        # 'E140': None,
        # 'I28': None,
        # 'I29': None,
        # 'E8': None,
        # 'E136': None,
        # 'B23': None,
        # 'J14': 'MINIMAL',
        # 'I14': None,
        # 'J13': 'IMPAIRED VISION',
        # 'I13': None,
        # 'H17': None,
        # 'I31': None,
        # 'B56': None,
        # 'C145': None,
        # 'D145': None,
        # 'E145': None,
        # 'F145': None,
        # 'G145': None,
        # 'C146': None,
        # 'D146': None,
        # 'E146': None,
        # 'F146': None,
        # 'G146': None,
        # 'B147': 'Normal',
        # 'C150': None,
        # 'C151': None,
        # 'C152': None,
        # 'D150': None,
        # 'D151': None,
        # 'D152': None,
        # 'B59': None,
        # 'C59': None,
        # 'F48': 'NAS, renal, controlled carbohydrate, ncs, low fat/cholesterol, high fiber, and nas, controlled carb, ncs, renal, high fiber diet'
        # }

    def done_restarting(self):
        self.restarting=False

    def update(self):
        if not self.excel_selected or self.scaning or self.restarting:
            return
        self.updating=True
        # Initialize and start the scanning thread
        self.update_thread = UpdateThread(self)
        self.update_thread.progress_signal.connect(self.update_progress)
        self.update_thread.start()
        self.update_thread.L4_signal.connect(self.set_paragraph_text1)
        self.update_thread.O4_signal.connect(self.set_paragraph_text2)
        self.update_thread.R4_signal.connect(self.set_paragraph_text3)
        self.update_thread.done.connect(self.done_updating)

    def done_updating(self):
        self.updating=False

from PyQt5.QtCore import QSize

class SettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set Font to Segoe UI
        font = QFont("Calibri", 14)  # Using Segoe UI font
        self.setFont(font)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout()

        self.chk_denoising = QCheckBox("Denoising", self)
        self.chk_denoising.setChecked(parent.denoising)
        self.chk_sharpen = QCheckBox("Sharpen", self)
        self.chk_sharpen.setChecked(parent.sharpen)
        self.chk_correcting_skew = QCheckBox("Correcting Skew", self)
        self.chk_correcting_skew.setChecked(parent.correcting_skew)
        self.chk_show_imgs = QCheckBox("Show images", self)
        self.chk_show_imgs.setChecked(parent.show_imgs)
        # Apply style to checkboxes
        self.apply_checkbox_style(self.chk_denoising)
        self.apply_checkbox_style(self.chk_sharpen)
        self.apply_checkbox_style(self.chk_correcting_skew)
        self.apply_checkbox_style(self.chk_show_imgs)

        layout.addWidget(self.chk_denoising)
        layout.addWidget(self.chk_sharpen)
        layout.addWidget(self.chk_correcting_skew)
        layout.addWidget(self.chk_show_imgs)

        btn_save = QPushButton("Save", self)
        btn_save.setFixedSize(320, 45)  # Adjust button size
        save_font = QFont("Calibri", 18)  # Larger font for Save button
        btn_save.setFont(save_font)
        btn_save.clicked.connect(self.accept)
        layout.addWidget(btn_save)
        self.setLayout(layout)
        self.apply_dark_theme()

    def apply_checkbox_style(self, checkbox):
        """Apply styling to the checkbox."""
        checkbox_font = QFont("Calibri", 20)  # Bigger font for checkbox
        checkbox.setFont(checkbox_font)
        checkbox.setIconSize(QSize(32, 32))  # Bigger checkbox icon

    def apply_dark_theme(self):
        # Set dark palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(30, 30, 30))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        
        # Set dark style sheet
        self.setStyleSheet("""
            QProgressBar {
                border: 1px solid #303030;
                border-radius: 8px;
                background-color: #232323;
            }

            QProgressBar::chunk {
                border-radius: 8px;
                background-color: #d47f00;
                width: 2px;
            }

            QCheckBox {
                color: #707070;
            }
            QScrollArea {
                border: 1px solid #303030;
                background-color: #232323;
                
            }

            QLabel{
                color:#707070;
                
            }
            QPushButton {
                background-color: #232323;
                color:#707070;
                padding: 5px;
                border-radius: 6px;
                outline: none;
                border: 1px solid #303030;
            }
            QPushButton:hover{
                background-color: #161616;
                color:#707070;
                padding: 5px;
                border-radius: 6px;
                outline: none;
                border: 1px solid #161616;
            }

            QPushButton:pressed {
                background-color: #d47f00;
                
                padding: 5px;
                border-radius: 6px;
                outline: none;
            }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())