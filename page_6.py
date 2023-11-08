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
import fitz

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
#page6
def detect_code_area(thresh,img):
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
#page6
def detect_code_area2(thresh,img):
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
#page6
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
            if ratio >= 0.8 and ratio <= 1.2 and w >= 18 and w <= 28  and y<600:  # Check if side length is at least 18 pixels
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
#page6
def detect_checkboxes2(thresh,img):
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
#page6
def detect_checkboxes3(thresh,img):
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
#page6
def detect_word_location(img, word, length, threshold=72):
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
                current_box=(x1+168, y1-7, x1+168+length, y2)#kima dert m3a Other welit nehseb l x2 men x1 w nzid valeur kima hna 161 parceque l x2 wlat taghlat 
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
        roi_x_start = x + r+6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/8/2023
        roi_x_end = x + x_ratio * r  # This can be adjusted based on expected text length
        roi_y_start = y - r- y_ratio
        roi_y_end = y + r+ y_ratio
        cv2.rectangle(output, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 0, 255), 2)
        rois_coordinates.append((roi_x_start, roi_y_start, roi_x_end, roi_y_end))
    return output, rois_coordinates
#page6
def extract_text_from_roi_checks(image_org, roi_coordinates):
    detected_options=[]
    detected_options2=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if roi_x_start<900 and roi_x_start>600 :
            detected_options.append(first_word)
        elif roi_x_start<1300 and roi_x_start>900:
            detected_options2.append(first_word)
    print('detected_options for deaf: ',detected_options)
    print('detected_options for hearing aid: ',detected_options2)
    return detected_options,detected_options2
#page6
def extract_text_from_roi_checks2(image_org, roi_coordinates):
    detected_options=[]
    detected_options2=[]
    detected_options3=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if roi_x_start<800 and roi_x_start>600:
            detected_options.append(first_word)
        elif roi_x_start<1400 and roi_x_start>1150:
            detected_options2.append(first_word)
        elif roi_x_start<1600 and roi_x_start>1400:
            detected_options3.append(first_word)
    print('detected_options for blurred: ',detected_options)
    print('detected_options for glaucoma: ',detected_options2)
    print('detected_options for cataracta: ',detected_options3)

    return detected_options,detected_options2,detected_options3
#page6
def extract_text_from_roi_checks3(image_org, roi_coordinates):
    detected_options=[]
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start+1:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        detected_options.append(first_word)
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
    print('detected_yes_no: ',detected_yes_no)

    print('detected_options: ',detected_options)
    return detected_options,detected_yes_no
#page6
def extract_text_from_roi(image_org, roi_coordinate,line=False):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    
    text = pytesseract.image_to_string(roi_image).strip()
    
    # text = pytesseract.image_to_string(roi_image, config='--psm 6 -c tessedit_char_blacklist=.-+_,|;:').strip()
    return text
#page6
def extract_numbers_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    number = pytesseract.image_to_string(roi_image, config='--psm 6 ').strip()
    return number

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

def update_excel_sheet(path,inputs):
    start_time = time.time()
    print ('opening the exel sheet...')
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
#page6
def preprocess_image_and_smouth_lines(image_org,ratio):
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
    Image.fromarray(table_segment).show()

    return table_segment
#page6
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

    return output, filtered_boxes
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
    

if __name__ == "__main__":
    start_time = time.time()
    for pdf_path in ['HIGH_QUALITY_3.pdf','BAD_QUALITY.pdf','BAD_QUALITY_3.pdf','BAD_QUALITY.pdf','FILLABLES_2.pdf','FILLABLES_3.pdf','FILLABLES_4.pdf','FILLABLES_5.pdf','FOR_UPWORK_#1.pdf','FOR_UPWORK_#2.pdf','FOR_UPWORK_#3.pdf','FOR_UPWORK_#4.pdf','HIGH_QUALITY_2.pdf','HIGH_QUALITY_3.pdf','HIGH_QUALITY.pdf','MEDIUM_QUALITY.pdf']:
# 'FILLABLE-9.pdf','BAD_QUALITY_2.pdf','BAD_QUALITY_3.pdf',
        
        input('press enter to start')
        print(pdf_path)
        excel_path = ('WORKING.xlsm')
        page=6
        noise=False# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
        plure=False
        skewed=True
        if pdf_path=='FOR_UPWORK_#4.pdf':
            noise=True# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
            plure=True
            skewed=True
        elif pdf_path=='HIGH_QUALITY_3.pdf' or pdf_path=='BAD_QUALITY_3.pdf' :
            skewed=True
        excel_inputs={}
        #***********************************valus that can be tweaked************************************************
        #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
        thresh_check_ratio=200#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_box_ratio=80
        thresh_lines_ratio=120


        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=5
        y_check_ratio=6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_radio_ratio=18
        y_radio_ratio=5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
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
        
        
        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        thresh_box = preprocess_image(image_org,thresh_box_ratio)
        image_org, drop_arrow = delet_drop_arrow(thresh_box,image_org)
        output_image, drop_arrow = delet_drop_arrow(thresh_box,output_image)
        thresh_code = preprocess_image(image_org,180)
        output_image, boxes = detect_code_area(thresh_check,output_image)

        # Image.fromarray(image_org).show()

        if len(boxes)==0:
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines(image_org,thresh_lines_ratio)
            output_image, boxes = detect_code_area(thresh_check_smouth_lines,output_image)
        
        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-85
        if y2-y1>36:
            y2=y2-10
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2
        hearing_text=extract_numbers_from_roi(thresh_code,box)
        # if hearing_text not in ['0','1','2','3','4']:
        #     x1=x1+1
        #     box=x1,y1,x2,y2
        #     hearing_text=extract_numbers_from_roi(image_org,box)

        print('hearing_text0:',hearing_text)

        if '1' in hearing_text:
            hearing_text='1'
        if '2' in hearing_text:
            hearing_text='2'
        if '3' in hearing_text:
            hearing_text='3'
        if 'c' in hearing_text:
            hearing_text='0'
        if 'z' in hearing_text:
            hearing_text='2'
        if '4' in hearing_text:
            hearing_text='1'
        if '7' in hearing_text:
            hearing_text='2'
        if 'a' in hearing_text:
            hearing_text='2'
        if 'q' in hearing_text:
            hearing_text='2'
        if hearing_text not in ['0','1','2','3']:
            hearing_text='0'
        
        
        excel_inputs={}

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
            excel_inputs['B65']=hearing

    #     #second part------------------------------------------------------------------------

        output_image, squares = detect_checkboxes(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        detected_check_text_deaf=[]
        detected_check_text_hearing=[]
        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
            
            print('********************************check_text*************************************************************')
            detected_check_text_deaf,detected_check_text_hearing= extract_text_from_roi_checks(image_org, roi_check_coordinates)

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

        output_image, boxes = detect_code_area2(thresh_check,output_image)
        # Image.fromarray(image_org).show()

        if len(boxes)==0:
            thresh_check_smouth_lines = preprocess_image_and_smouth_lines(image_org,thresh_lines_ratio)
            output_image, boxes = detect_code_area2(thresh_check_smouth_lines,output_image)

        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+36, y+36, x+w-57, y+h-116
        print(y2-y1)
        if y2-y1>36:
            y2=y2-10
            y1=y1+3
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        box=x1,y1,x2,y2

        vision_text=extract_numbers_from_roi(thresh_code,box)

        print('vision_text0:',vision_text)

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
            vision_text='4'
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
        4:"SEVERELY IMPAIRED VISION"
    }

        vision=vision_mapping.get(vision_number, "Invalid vision_number")
        
        print(f"vision: {vision}")

        if vision != "Invalid vision_number":
            excel_inputs['J13']=vision


        #fourth part---------------------------------------------------------------------------------


        output_image, squares = detect_checkboxes2(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)
        # cv2.rectangle(output_image, (1400, 900), (1600, 920), (255, 0, 255), 4)
        detected_check_text_deaf=[]
        detected_check_text_hearing=[]
        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
        
            print('********************************check_text*************************************************************')
            detected_check_text_blurred,detected_check_text_glaucoma,detected_check_text_cataracta= extract_text_from_roi_checks2(image_org, roi_check_coordinates)

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
        
        output_image, squares = detect_checkboxes3(thresh_check,output_image)

        filled_check_buttons,output_image = detect_filled_button(thresh_check,squares,output_image,filled_check_ratio)

        detected_check_text_deaf=[]
        detected_check_text_hearing=[]
        if filled_check_buttons:
            output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)

            print('********************************check_text*************************************************************')
            detected_check_text= extract_text_from_roi_checks3(image_org, roi_check_coordinates)
            print('***************************DENTURES status *************************************************************')
            DENTURES_options = ["Dentures", "Upper", "Lower", "Partial"]
            validated_DENTURES_options = validate_option(detected_check_text,DENTURES_options,80)
            if validated_DENTURES_options!=None:
                DENTURES=update_dentures_cell(validated_DENTURES_options)
                print("DENTURES: ",DENTURES)
                excel_inputs['H17'] = DENTURES

        #sixth part------------------------------------------------------------------------------------


        Educational_ROI=detect_word_location(image_org,'Educational',300)
        if Educational_ROI:
            x1,y1,x2,y2=Educational_ROI[0]
            roi=x1,y1,x2,y2
            print(y1)
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



        #******************updating the excel file*************************************
        update_excel_sheet(excel_path, excel_inputs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.2f} seconds")
# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py