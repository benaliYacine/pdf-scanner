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

def detect_radio_buttons(thresh, image_org):
    circles = cv2.HoughCircles(
        # thresh, cv2.HOUGH_GRADIENT, dp=1.35, minDist=30, param1=50, param2=25, minRadius=8, maxRadius=11
        # thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=50, param2=25, minRadius=20, maxRadius=30
        #**********************************************************************************************
        thresh, cv2.HOUGH_GRADIENT, dp=0.1, minDist=20, param1=50, param2=25, minRadius=9, maxRadius=15
    )
    output = image_org.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    return output, circles

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
            bounding_boxes.append((x2+2, y1, x2+length, y2))

    return bounding_boxes

def detect_phrase_location(img,phrase,length):
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
        roi_x_start = x + r+2
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

def extract_text_from_roi_radios(image_org, roi_coordinates):
    detected_options=[]
    detected_yes_no={}
    for roi_x_start, roi_y_start, roi_x_end, roi_y_end in roi_coordinates :
        roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
        #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
        first_word=text.split(' ')[0]# Extracting the first word
        if 'No' in first_word:
            roi_no = image_org[roi_y_start:roi_y_end, roi_x_start-200:roi_x_start-30]
            text_no = pytesseract.image_to_string(roi_no, config='--psm 6').strip().split(' ')[-1]
            cv2.rectangle(image_org, (roi_x_start-300, roi_y_start), (roi_x_start-30, roi_y_end), (0, 0, 255), 2)
            # Image.fromarray(image_org).show()
            detected_yes_no[text_no]='No'
        elif 'Yes' in first_word:
            roi_yes = image_org[roi_y_start:roi_y_end, roi_x_start-310:roi_x_start-100]
            text_yes = pytesseract.image_to_string(roi_yes, config='--psm 6').strip().split(' ')[-1]
            cv2.rectangle(image_org, (roi_x_start-390, roi_y_start), (roi_x_start-110, roi_y_end), (0, 0, 255), 2)
            # Image.fromarray(image_org).show()
            detected_yes_no[text_yes]='Yes'
        else:
            detected_options.append(first_word)
    print('detected_yes_no: ',detected_yes_no)

    print('detected_options: ',detected_options)
    return detected_options,detected_yes_no

def extract_text_from_roi(image_org, roi_coordinate):
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_coordinate 
    roi_image = image_org[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    text = pytesseract.image_to_string(roi_image, config='--psm 6').strip()  
    #ida kanet l kelma no wela yes n3amro dictionaire detected_yes_no
    return text

def validate_option(detected_text,options):
    validated_options=[]
    for text in detected_text:
        for option in options:
            if text in option:
                validated_options.append(option)
    if len(validated_options)==0:
        return None
    else :
        print('validated options;',validated_options)
        return validated_options

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
    pdf_path = (r'guide\FOR_UPWORK_#1.pdf')
    excel_path = (r'guide\WORKING.xlsm')
    page=10
    #***********************************valus that can be tweaked************************************************
    #ratio ta3 dettection ta3 les radio buttons w check boxes modifier 3liha
    thresh_check_ratio=180
    thresh_radio_ratio=120
    thresh_OCR_ratio=200
    #ratio ta3 dettection ta3 li filled modifier 3liha lima tehtej 0.1 ma3naha yel9a ghi 10% mel button black y acceptih filled. 0.9 ma3nah lawem 90% mel button black bah y acceptih filled
    filled_radio_ratio=0.5
    filled_check_ratio=0.6
    #ratio ta3 tol w 3ard el ROIs ta3 koul button
    x_check_ratio=13
    y_check_ratio=0

    x_radio_ratio=13
    y_radio_ratio=3
    image_org = pdf_to_image(pdf_path,page)
    Image.fromarray(image_org).show()
    denoised = cv2.fastNlMeansDenoisingColored(image_org, None, 10, 10, 7, 21)
    Image.fromarray(denoised).show()
    image_org=denoised
    thresh_check = preprocess_image(image_org,thresh_check_ratio)
    thresh_radio = preprocess_image(image_org,thresh_radio_ratio)
    output_image, squares = detect_checkboxes(thresh_check,image_org)
    output_image, circles = detect_radio_buttons(thresh_radio,output_image)
    filled_check_buttons,output_image = detect_filled_button(thresh_radio,squares,output_image,filled_check_ratio)
    filled_radio_buttons,output_image = detect_filled_button(thresh_radio,circles,output_image,filled_radio_ratio)
    img_for_OCR= preprocess_image(image_org,thresh_OCR_ratio)
    img_for_OCR2= preprocess_image(image_org,140)
    if filled_radio_buttons or filled_check_buttons:
        output_image, roi_check_coordinates = extract_text_roi(output_image, filled_check_buttons,x_check_ratio,y_check_ratio)
        output_image, roi_radio_coordinates = extract_text_roi(output_image, filled_radio_buttons,x_radio_ratio,y_radio_ratio)

        print('********************************radio_text*************************************************************')
        detected_radio_text,detected_radio_yesNo= extract_text_from_roi_radios(img_for_OCR, roi_radio_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
        print('********************************check_text*************************************************************')
        detected_check_text= extract_text_from_roi_checks(img_for_OCR, roi_check_coordinates) # detrt image_org fel fct bah ya9ra txt men img li ma rsamnach fiha lakhaterch ki rsamna ghatina 3la l harf lewel mel kelma
        excel_inputs={}
        # ***************************marital status *************************************************************
        print('***************************marital status *************************************************************')
        marital_options = ["Single", "Married", "Divorced", "Widower"]
        validated_marital_options = validate_option(detected_radio_text,marital_options)
        if validated_marital_options!=None:
            marital_status = validated_marital_options[0]
            print(f"marital_status: {marital_status}")
            excel_inputs['I33']=marital_status
        else :
            print('nothing selected for marital status')

        # ***************************Feelings/emotions*************************************************************
        print('***************************Feelings/emotions***********************************************************')
        Feelings_emotions_options = ["N/A - Nothing reported","Angry","Fear","Sadness","Discouraged","Lonely","Depressed","Helpless","Content","Happy","Hopeful","Motivated"]
        validated_Feelings_options = validate_option(detected_check_text,Feelings_emotions_options)
        #Other:
        other_ROI=detect_word_location(img_for_OCR,'ther:',580)
        x1,y1,x2,y2=other_ROI[1]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        Other_text=extract_text_from_roi(img_for_OCR,other_ROI[1])
        print('Other:',Other_text)
        if validated_Feelings_options!=None:
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
        validated_inability_options = validate_option(detected_check_text,inability_options)
        if validated_inability_options!=None:
            inability= join_strings(validated_inability_options)
            print(f"inability: {inability}")
            excel_inputs['I44']=inability
        else:
            print('nothing selected for inability')

        # ***************************Spiritual_resource*************************************************************
        print('***************************Spiritual_resource***********************************************************')
        Spiritual_resource_ROI=detect_phrase_location(img_for_OCR2,'Spiritual resource',820)
        x1,y1,x2,y2=Spiritual_resource_ROI[0]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        Spiritual_resource=extract_text_from_roi(img_for_OCR,Spiritual_resource_ROI[0])
        print(f"Spiritual resource: {Spiritual_resource}")
        excel_inputs['I32']=Spiritual_resource
    else:
        print("No filled button detected.")
    Image.fromarray(output_image).show()
    #******************updating the excel file*************************************
    update_excel_sheet(excel_path, excel_inputs)

# convert the code into .exe:
# C:\Users\benal\AppData\Roaming\Python\Python311\Scripts\pyinstaller.exe --onefile C:\slash\work\project_with_abd\python_pdf_exel\radio_button\5.detecting_the_text_and_update_the_exel_sheet_succesfully.py