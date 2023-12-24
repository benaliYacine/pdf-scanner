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

def pdf_to_image(pdf_path,page):
    images = convert_from_path(pdf_path)
    return np.array(images[page-1])

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
            if w >= 155 and w <= 185 and h >= 60 and h <= 75:  # Check if side length is at least 18 pixels
                print( w, h)
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

def detect_word_location(img, word, length, threshold=90):

    length=(length-126)//3
    hocr_data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr').decode('utf-8')
    bounding_boxes = []

    for line in hocr_data.splitlines():
        r=4
        if fuzz.partial_ratio(line, word) >= threshold:
            x1, y1, x2, y2 = line.split('bbox ')[1].split(';')[0].split()
            x1, y1, x2, y2 = int(x1)-r, int(y1)-r, int(x2)+r, int(y2)+r

            bounding_boxes.append((x2+126, y1-20, x2+length+126-10, y2+19))
            bounding_boxes.append((x2+length+126, y1-20, x2+length*2+126-10, y2+19))
            bounding_boxes.append((x2+length*2+126, y1-20, x2+length*3+126-10, y2+19))

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

def extract_text_from_roi(image_org, roi_coordinate,line=False):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # denoised = cv2.fastNlMeansDenoisingColored(roi_image, None,1, 1, 7, 21)
    # Image.fromarray(denoised).show()
    
    text = pytesseract.image_to_string(roi_image).strip()
    
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

if __name__ == "__main__":
    start_time = time.time()
    for pdf_path in ['BAD_QUALITY_3.pdf','HIGH_QUALITY_3.pdf','HIGH_QUALITY.pdf','MEDIUM_QUALITY.pdf']:
        
        print(pdf_path)
        excel_path = ('WORKING.xlsm')
        page=22
        noise=False# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
        plure=False
        skewed=False
        if pdf_path=='FOR_UPWORK_#4.pdf':
            noise=True# khalih daymen cha3el ynahi hadouk l ahrof random li yekhorjou ki detcti text fi blasa vide (bayda)
            plure=True
            skewed=True
        elif pdf_path=='HIGH_QUALITY_3.pdf' or pdf_path=='BAD_QUALITY_3.pdf' :
            skewed=True
        #***********************************valus that can be tweaked************************************************
        #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
        thresh_check_ratio=170  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        thresh_radio_ratio=0#kanet 120 bekri w kanet temchi m3a kamel les pdf li semouhom for upwork
        #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
        filled_radio_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023kanet 0.4 w kanet temchi m3a kamel les pdf li semouhom for upwork
        filled_check_ratio=0.6#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #ratio ta3 tol w 3ard el ROIs ta3 koul button
        x_check_ratio=30
        y_check_ratio=5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023

        x_radio_ratio=18
        y_radio_ratio=5#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        image_org = pdf_to_image(pdf_path,page)
        Image.fromarray(image_org).show()
        #kayen des pdf ki mscanyiin mayliin 
        if skewed:
            corrected = correct_skew(image_org)
            image_org=corrected
            Image.fromarray(corrected).show()


        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++9/5/2023
        #hada denoiser ynahi el noise li kayen f ba3d les pdf lakin ma ykhasarch l ketba ga3 tema ani ayrou daymen yemchi mechi ghi m3a li fihom el noise
        if noise:
            denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
            image_org=denoised
            Image.fromarray(denoised).show()
        
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
            Image.fromarray(sharpened).show()

        #hna ki kout ndiir bezaf sharpening kan ykhorjou des trace noire fel blayes elboyed tema ki n detecti text fi blasa vide yehseb kayen text tema ,tema seyit ndir denoise pour une dexieme fois bach enahi hadouk kes traces
        # denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
        # image_org=denoised
        output_image=image_org


        #++++++++++++++++++++++++++++++++++++++++++++
        thresh_check = preprocess_image(image_org,thresh_check_ratio)
        # Image.fromarray(thresh_check).show()
        output_image, boxes = detect_checkboxes(thresh_check,output_image)
        # Image.fromarray(output_image).show()

        if len(boxes)!=3:
            print('cannot detect the sites, I will now try to smouth the lines :')
            thresh_check = preprocess_image_and_smouth_lines(image_org,thresh_check_ratio)
            # Image.fromarray(thresh_check).show()
            output_image, boxes = detect_checkboxes(thresh_check,output_image)
            # Image.fromarray(output_image).show()

        if len(boxes)==3:
            x, y, w, h = boxes[0]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            box=x1,y1,x2,y2
            site1_text=extract_text_from_roi(image_org,box)
            site1_text=site1_text.replace('‘','')
            print('site1:',site1_text)


            x, y, w, h = boxes[1]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box=x1,y1,x2,y2
            site2_text=extract_text_from_roi(image_org,box)
            site2_text=site2_text.replace('‘','')
            print('site2:',site2_text)

            
            x, y, w, h = boxes[2]
            x1,y1,x2,y2 =x+2, y+3, x+w-2, y+h-3

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            box=x1,y1,x2,y2
            site3_text=extract_text_from_roi(image_org,box)
            site3_text=site3_text.replace('‘','')
            print('site3:',site3_text)



            # Assuming you've already gotten the values for site1_text, site2_text, and site3_text
            PAIN_ASSESSMENT = format_site_texts(site1_text, site2_text, site3_text)

            # Update the Excel sheet
            excel_inputs = {}
            print(f"PAIN ASSESSMENT: {PAIN_ASSESSMENT}")
            excel_inputs['H20'] = PAIN_ASSESSMENT
        else:
            print('cannot detect the three sites')

        Image.fromarray(output_image).show()

        #******************updating the excel file*************************************
        # update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py