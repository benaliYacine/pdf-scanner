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
#page1
def detect_checkboxes(thresh,img):
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
                print(w,h,y)
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
#page1
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
            if y1<600:
                bounding_boxes.append((x2-3, y1-10, x2+length, y2-2))

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
        roi_x_start = x + r+5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/8/2023
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
#page1
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
    number = pytesseract.image_to_string(roi_image ,config='--psm 6 outputbase digits').strip()
    return number

def extract_number_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    number = pytesseract.image_to_string(roi_image ,config='--psm 10').strip()
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
#page1   
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
#page1  
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
import os
if __name__ == "__main__":
    
    start_time = time.time()
    # for pdf_path in [f for f in os.listdir() if f.endswith('.pdf')]:
    for pdf_path in ['SCANNED-12.pdf']:
        excel_inputs={}
        print(pdf_path)
        excel_path = ('WORKING.xlsm')
        page=1
        noise=False# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
        plure=False
        skewed=False
        if pdf_path=='FOR_UPWORK_#4.pdf':
            noise=True# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
            plure=True
            skewed=True
        elif pdf_path=='HIGH_QUALITY_3.pdf' or pdf_path=='BAD_QUALITY_3.pdf' :
            skewed=True

        input()
        #***********************************valus that can be tweaked************************************************
        #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
        thresh_check_ratio=150#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=100#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

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
        # Image.fromarray(thresh_check).show()
        output_image, boxes = detect_checkboxes(thresh_check,output_image)

        # Image.fromarray(output_image).show()



        Date_ROI=detect_word_location(image_org,'DATE:',80)
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


        x, y, w, h = boxes[0]
        x1,y1,x2,y2 =x+70, y+50, x+w-80, y+h-90
        print (y2-y1)
        if y2-y1>25:
            y2=y1+25
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
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
                print(Priority_number)
                if Priority_number=='4':
                    priority_value = determine_priority_value('1')
                if priority_value:
                    print('priority:',priority_value)
                    excel_inputs["B14"] = priority_value
                else:
                    print('cannot detect the priority')
        Image.fromarray(output_image).show()
        #******************updating the excel file*************************************
        # update_excel_sheet(excel_path, excel_inputs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time of execution: {execution_time:.2f} seconds")

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py