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
    number = pytesseract.image_to_string(roi_image, config='--psm 7 outputbase digits').strip()
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
                # print(y1)
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

def page6(image_org,noise, plure, skewed):
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
        Image.fromarray(output_image).show()
    except:
        print("*the program was not able to continue reading the page 6 of this pdf!*")
    else:
        return excel_inputs

def page9(image_org,noise, plure, skewed):
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

def page10(image_org,noise, plure, skewed):
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
            Spiritual_resource_ROI=detect_word_location2(image_org,'Spiritual',350,75)#75 ta3 threshold lazem tkoun 75 bah ydetecti Spiritual daymen koun 80 kayen pdf ma ye9rahach 
            if len(Spiritual_resource_ROI)!=0:
                x1,y1,x2,y2=Spiritual_resource_ROI[0]
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                Spiritual_resource=extract_text_from_roi(image_org,Spiritual_resource_ROI[0],True)
                print(f"Spiritual resource: {Spiritual_resource}")
                excel_inputs['I32']=Spiritual_resource
        else:
            print("No filled button detected.")
        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 10 of this pdf!*")
    else:
        return excel_inputs

def page18(image_org,noise, plure, skewed):
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

        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 18 of this pdf!*")
    else:
        return excel_inputs

def page19(image_org,noise, plure, skewed):
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
        Image.fromarray(image_org).show()

    except:
        print("*the program was not able to continue reading the page 19 of this pdf!*")
    else:
        return excel_inputs

def page22(image_org,noise, plure, skewed):
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

def page22p2(image_org,noise, plure, skewed):
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

        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 22p2 of this pdf!*")
    else:
        return excel_inputs

def page24(image_org,noise, plure, skewed):
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


        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 24 of this pdf!*")
    else:
        return excel_inputs

def page25(image_org,noise, plure, skewed):
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


        
        # #guid2
        # output_image, squares = detect_checkboxes_p25(thresh_check,output_image)
        # filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        # if filled_check_buttons:
        #     output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
        #     print('********************************check_text*************************************************************')
        #     detected_check_text= extract_text_from_roi_checks(image_org, roi_check_coordinates)
        #     print('***************************nutritional********************************************************************')
        #     nutritional_options = ["renal", "NPO", "controlled carbohydrate", "general", "NAS"]
        #     validated_nutritional_options = validate_option(detected_check_text,nutritional_options,80)
        #     if not validated_nutritional_options:
        #         validated_nutritional_options=[]
        # else:
        #     validated_nutritional_options=[]
        #     print("No filled button detected.")

        # #other:
        # other_ROI=detect_word_location_p25(image_org,'other',410)
        # if other_ROI:
        #     x1,y1,x2,y2=other_ROI[0]
        #     roi=x1+1,y1,x2,y2
        #     # print(y2-y1)
        #     Other_text=extract_text_from_roi(image_org,roi)
        #     cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #     print('Other:',Other_text)
        # else:
        #     Other_text=''
        #     print("cannot detect the word other")
        
        # # Nutritional requirements (diet):
        # requirements_ROI=detect_phrase_location_p25(image_org, 'Nutritional requirements', 480)
        # if requirements_ROI:
        #     x1,y1,x2,y2=requirements_ROI[0]
        #     roi=x1,y1,x2,y2
        #     requirements_text=extract_text_from_roi(image_org,roi)
        #     cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #     print('Nutritional requirements (diet):',requirements_text)
        # else:
        #     requirements_ROI=detect_phrase_location_p25(image_org, 'Nutritionalrequirements', 480)
        #     if requirements_ROI:
        #         x1,y1,x2,y2=requirements_ROI[0]
        #         roi=x1,y1,x2,y2
        #         requirements_text=extract_text_from_roi(image_org,roi)
        #         cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #         print('Nutritional requirements (diet):',requirements_text)
        #     else:
        #         requirements_text=''
        #         print("cannot detect the phrase Nutritional requirements (diet):")

        # unique_options=[]

        # # Check if Other_text and requirements_text are the same using fuzz library
        # if fuzz.ratio(Other_text.lower(), requirements_text.lower()) >= 80 :
        #     # If they are the same, use only one of them
        #     unique_options = validated_nutritional_options + [Other_text.lower()]
        # else:
        #     unique_options = validated_nutritional_options + [Other_text.lower(), requirements_text.lower()]

        # # Remove duplicates and any empty strings
        # unique_options = list(filter(None, unique_options))
        # # Use join_strings function to join all the options
        # final_diet_text = join_strings(unique_options)

        # if not final_diet_text.endswith(" diet") and final_diet_text!='':
        #     final_diet_text += " diet"


        # print('nutritional status: ',final_diet_text)

        # # Update the excel_inputs dictionary for cell F48
        # excel_inputs['F48'] = final_diet_text

        Image.fromarray(output_image).show()

    except:
        print ("*the program was not able to continue reading the page 25 of this pdf!*")
    else:
        return excel_inputs

def page35(image_org,noise, plure, skewed):
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
    number = pytesseract.image_to_string(roi_image, config='--psm 7').strip()

    return number


def delet_drop_arrow_p6_with_L(thresh,img):
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
        cv2.rectangle(output, (x, y-4), (x+w-10, y+h+4), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y-4), (x, y+1), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y+h-3), (x, y+h+4), (255, 255, 255), -1)
        #khat el L
        cv2.rectangle(output, (x, y+h-5), (x+7, y+h-4), (0, 0, 0), -1)
        
        # cv2.rectangle(output, (x-2, y-4), (x, y+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y-4), (x, y+3), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-2, y+h-4), (x, y+h+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y+h-3), (x, y+h+4), (255, 255, 255), -1)

    return output, filtered_boxes

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
    k=None
    # Draw the filtered boxes on the image
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x-1, y-4), (x+w-10, y+h+4), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y-4), (x, y+1), (255, 255, 255), -1)
        cv2.rectangle(output, (x-10, y+h-1), (x, y+h+4), (255, 255, 255), -1)
        k=x-1
        # cv2.rectangle(output, (x-2, y-4), (x, y+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y-4), (x, y+3), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-2, y+h-4), (x, y+h+4), (255, 255, 255), -1)
        # cv2.rectangle(output, (x-3, y+h-3), (x, y+h+4), (255, 255, 255), -1)

    return output, k

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
    
        if fuzz.partial_ratio('No',first_word) > 80 and not fuzz.partial_ratio('Non-productive',first_word) > 80:
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
        thresh, cv2.HOUGH_GRADIENT, dp=0.1, minDist=20, param1=50, param2=20, minRadius=6, maxRadius=15# kanet dp=0.1 w kanet temchi m3a kamel les pdf li smouhoum for upwork
    )
    output = image_org.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles=[]
        for i in range(circles.shape[0]):
            #radit r=10 daymen mechi 3la hsab cha ydetecti l code kima l checkboxes radithom diameter=11 haka wlat khir fel detection ta3 filled buttons pareceque daymen nafs l size men9bel kan kayen li ydetectihom kbar kayen li sghar tema hadak el ratio s3ib bach nhadedou ida 0.5 wela 0.6 welaa....
            x, y, r = circles[i]
            if y>1200 and y<1500:
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

def validate_option_p24(detected_text, options, threshold=80):
    validated_options = []

    for text in detected_text:
        for option in options:
            # Using token sort ratio to handle out of order issues and partial matches
            if fuzz.ratio(text, option.split()[0]) > threshold:
                if option=="Ist":
                    option='1st'
                validated_options.append(option)

    if len(validated_options) == 0:
        return []
    else:

        print('validated options:', validated_options)
        return validated_options



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

def page1(image_org,noise, plure, skewed):
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
        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 1 of this pdf!*")
    else:
        return excel_inputs

def page4(image_org,noise, plure, skewed):
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

        Image.fromarray(output_image).show()


    except:
        print("*the program was not able to continue reading the page 4 of this pdf!*")
    else:
        return excel_inputs

def page6p2(image_org,noise, plure, skewed):
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
        # image_org, drop_arrow = delet_drop_arrow_p6(thresh_box,image_org)
        # output_image, drop_arrow = delet_drop_arrow_p6(thresh_box,output_image)
        image_org, xf = delet_drop_arrow_p6(thresh_box,image_org)
        output_image, xf = delet_drop_arrow_p6(thresh_box,output_image)
        thresh_code = preprocess_image(image_org,180)
        output_image, boxes = detect_code_area_p6(thresh_check,output_image)

        # Image.fromarray(image_org).show()

        if len(boxes)==0:
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines_p6(image_org,thresh_lines_ratio)
            output_image, boxes = detect_code_area_p6(thresh_check_smouth_lines,output_image)
        if len(boxes)==0:
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines_p6(image_org,170)
            output_image, boxes = detect_code_area_p6(thresh_check_smouth_lines,output_image)

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-85
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(y2-y1)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if y2-y1>36:
            y2=y2-10
            # x1=x1+3
        #     y2=y2-11
        #     y1=y1+5
        #     x2=x2+5

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # cv2.rectangle(image_org, (x2-4-10, y1+6), (x2-2-10, y2-6), (0, 0, 0), -1)
        # cv2.rectangle(image_org, (x2-4-10, y2-8), (x2+3-10, y2-6), (0, 0, 0), -1)

        # cv2.rectangle(output_image, (x2-4-10, y1+6), (x2-2-10, y2-6), (0, 0, 0), -1)
        # cv2.rectangle(output_image, (x2-4-10, y2-8), (x2+3-10, y2-6), (0, 0, 0), -1)
        # x2=x2+5


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if xf:
        #     x2=xf
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        hearing_text=extract_numbers_from_roi_p6(thresh_code,box)
        # if hearing_text not in ['0','1','2','3','4']:
        #     x1=x1+1
        #     box=x1,y1,x2,y2
        #     hearing_text=extract_numbers_from_roi(image_org,box)


        if '1' in hearing_text:
            hearing_text='1'
        if '|' in hearing_text:
            hearing_text='1'
        if '2' in hearing_text:
            hearing_text='2'
        if '3' in hearing_text:
            hearing_text='3'
        if '4' in hearing_text:
            hearing_text='1'
        if 'z' in hearing_text:
            hearing_text='2'
        if 'Z' in hearing_text:
            hearing_text='2'
        if '7' in hearing_text:
            hearing_text='2'
        if 'a' in hearing_text:
            hearing_text='2'
        if 'q' in hearing_text:
            hearing_text='0'


        if hearing_text not in ['0','1','2','3']:
            hearing_text='0'
            


        


        print('hearing_text:',hearing_text)


        if hearing_text in ['0','1','2','3']:
            hearing_number=int(hearing_text)
        else:
            hearing_number=4#bah fel dict yji "Invalid hearing_number" aya if hearing != "Invalid hearing_number": matemchich

        hearing_mapping = {
        0: "ADEQUATE",
        1: " MINIMAL",
        2: "MODERATLY IMPAIRED HEARING DIFFICULTY IN BILATERAL EARS",
        3: "HIGHLY IMPAIRED",
    }

        hearing=hearing_mapping.get(hearing_number, "Invalid hearing_number")

        print(f"hearing: {hearing}")

        if hearing != "Invalid hearing_number":
            excel_inputs['J14']=hearing

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
        if len(boxes)==0:
            # print('smouthig...')
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines_p6(image_org,170)
            output_image, boxes = detect_code_area2_p6(thresh_check_smouth_lines,output_image)

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-116
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print(y2-y1)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if y2-y1>36:
            y2=y2-10
            y1=y1+3
            # x1=x1+3
        #     x2=x2+5

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # cv2.rectangle(image_org, (x2-4-10, y1+6), (x2-2-10, y2-6), (0, 0, 0), -1)
        # cv2.rectangle(image_org, (x2-4-10, y2-8), (x2+3-10, y2-6), (0, 0, 0), -1)

        # cv2.rectangle(output_image, (x2-4-10, y1+6), (x2-2-10, y2-6), (0, 0, 0), -1)
        # cv2.rectangle(output_image, (x2-4-10, y2-8), (x2+3-10, y2-6), (0, 0, 0), -1)
        # x2=x2+5
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if xf:
        #     x2=xf
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2


        vision_text=extract_numbers_from_roi_p6(thresh_code,box)
        # print('vision_text0:',vision_text)

        if '1' in vision_text:
            vision_text='1'
        if '|' in vision_text:
            vision_text='1'
        if '2' in vision_text:
            vision_text='2'
        if '3' in vision_text:
            vision_text='3'
        if '4' in vision_text:
            vision_text='1'
        if 'z' in vision_text:
            vision_text='2'
        if 'Z' in vision_text:
            vision_text='2'
        if '7' in vision_text:
            vision_text='2'
        if 'a' in vision_text:
            vision_text='2'
        if 'q' in vision_text:
            vision_text='0'
        if 'c' in vision_text:
            vision_text='0'
        if vision_text not in ['0','1','2','3','4']:
            vision_text='0'



        if vision_text=='1':
            box=x1,y1,x2,y2
            vision_text=extract_numbers_from_roi(image_org,box)
            # print('vision_text:',vision_text)
            if vision_text.startswith('1') or vision_text[1:].startswith('1'):
                vision_text='1'
            elif vision_text.startswith('4'):
                vision_text='4'
            else:
                vision_text='1'

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
        print('vision:',vision)
        if vision:
            excel_inputs['I13']=vision


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
        Image.fromarray(output_image).show()

    except:
        print("*the program was not able to continue reading the page 6p2 of this pdf!*")
    else:
        return excel_inputs

def page5(image_org,noise, plure, skewed):
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
                IMMUNIZATIONS_status = join_strings(validated_IMMUNIZATIONS_options).upper()
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





        Image.fromarray(output_image).show()


    except:
        print("*the program was not able to continue reading the page 5 of this pdf!*")
    else:
        return excel_inputs

def page24p2(image_org,noise, plure, skewed):
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
            validated_radio_text=validate_option_p24(detected_radio_text,options)
            # Check the conditions and update excel_inputs['B56'] accordingly
            cough_status = detected_radio_yesNo.get('Cough', '')  # Default to empty string if 'Cough' is not in the dict
            if cough_status == 'No':
                pass  # excel_inputs['B56'] remains empty

            elif 'Productive' in validated_radio_text:
                excel_inputs['B56'] = 'PRODUCTIVE COUGH'
            elif 'Non-productive' in validated_radio_text:
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
                for cell in cells:
                    if cell not in excel_inputs:
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

        Image.fromarray(output_image).show()


    except:
        print("*the program was not able to continue reading the page 24p2 of this pdf!*")
    else:
        return excel_inputs

if __name__ == "__main__":
    start_time = time.time()
    excel_inputs = {}
    # for pdf_path in ['BAD_QUALITY_2.pdf','BAD_QUALITY_3.pdf','BAD_QUALITY.pdf','FILLABLES_2.pdf','FILLABLES_3.pdf','FILLABLES_4.pdf','FILLABLES_5.pdf','FILLABLES_6.pdf','FOR_UPWORK_#1.pdf','FOR_UPWORK_#2.pdf','FOR_UPWORK_#3.pdf','HIGH_QUALITY_2.pdf','HIGH_QUALITY_3.pdf','HIGH_QUALITY.pdf','MEDIUM_QUALITY.pdf','OASIS FILLABLE.pdf']:
    # Check current directory for any PDF files
    pdf_files = [f for f in os.listdir() if f.endswith('.pdf')]
    excel_files = [f for f in os.listdir() if f.endswith('.xlsm')]
    if excel_files:
        excel_path = excel_files[0]
        print('excel file:',excel_path)
    else:
        print('Error: No excel file found in the current directory!')
    if pdf_files:  # If there's at least one PDF file
        # print(pdf_files)
        for pdf_path in pdf_files:
            # pdf_path='FILLABLE-9.pdf'
            print ('PDF file:',pdf_path)
            input('press enter to start:')
            all_pages = pdf_to_images(pdf_path)

            noise=False 
            plure=False 
            skewed=True
            if pdf_path=='Noisy.pdf':
                noise=True
                plure=True

            p6_excel_inputs=page6(np.array(all_pages[5]),noise, plure, skewed)
            p9_excel_inputs=page9(np.array(all_pages[8]),noise, plure, skewed)
            p10_excel_inputs=page10(np.array(all_pages[9]),noise, plure, skewed)
            p18_excel_inputs=page18(np.array(all_pages[17]),noise, plure, skewed)
            p19_excel_inputs=page19(np.array(all_pages[18]),noise, plure, skewed)
            p22_excel_inputs=page22(np.array(all_pages[21]),noise, plure, skewed)
            p22p2_excel_inputs=page22p2(np.array(all_pages[21]),noise, plure, skewed)
            p24_excel_inputs=page24(np.array(all_pages[23]),noise, plure, skewed)
            p25_excel_inputs=page25(np.array(all_pages[24]),noise, plure, skewed)
            p35_excel_inputs=page35(np.array(all_pages[34]),noise, plure, skewed)

            # p1_excel_inputs = page1(np.array(all_pages[0]), noise, plure, skewed)
            # p4_excel_inputs = page4(np.array(all_pages[3]), noise, plure, skewed)
            # p5_excel_inputs = page5(np.array(all_pages[4]), noise, plure, skewed)
            # p6p2_excel_inputs = page6p2(np.array(all_pages[5]), noise, plure, skewed)
            # p24p2_excel_inputs = page24p2(np.array(all_pages[23]), noise, plure, skewed)

            if p6_excel_inputs:
                excel_inputs.update(p6_excel_inputs)
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
            if p25_excel_inputs:
                excel_inputs.update(p25_excel_inputs)
            if p35_excel_inputs:
                excel_inputs.update(p35_excel_inputs)

            # if p1_excel_inputs:
            #     excel_inputs.update(p1_excel_inputs)
            # if p4_excel_inputs:
            #     excel_inputs.update(p4_excel_inputs)
            # if p5_excel_inputs:
            #     excel_inputs.update(p5_excel_inputs)
            # if p6p2_excel_inputs:
            #     excel_inputs.update(p6p2_excel_inputs)
            # if p24p2_excel_inputs:
            #     excel_inputs.update(p24p2_excel_inputs)
            #******************updating the excel file*************************************
            update_excel_sheet(excel_path, excel_inputs)
    else:
        print("Error: No PDF files found in the current directory!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.2f} seconds")


    input("Press Enter to exit...")