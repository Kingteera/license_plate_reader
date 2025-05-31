import cv2
import numpy as np
# import Levenshtein
from PIL import Image
import json
import requests
import io
from difflib import SequenceMatcher
import time

plate_details = ["alpha","number","province"] 

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# def find_similar(a, b):
#     levenshtein_distance = Levenshtein.distance(a, b)
#     total_length = len(a) + len(b)
#     return 1 - (levenshtein_distance / total_length)

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # control Contrast by 1.5 
    alpha = 1.2 
    # control brightness by 50 
    beta = 5
    image2 = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta) 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    sharpened_image = cv2.filter2D(image2, -1, kernel) 
    # Sharpen the image 
    #blurred = cv2.GaussianBlur(image2, (5, 5), 0)
    #th2 = cv2.adaptiveThreshold(sharpened_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,15,5)
    #equalized = cv2.equalizeHist(sharpened_image)

    # Apply Otsu's thresholding
    #_, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise
    #kernel = np.zeros((3, 3), np.uint8)
    #processed_image = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    
    return sharpened_image

def point_in_quad(px, py, quad):
    """Check if a point is inside a quadrilateral using OpenCV."""
    quad_np = np.array(quad, dtype=np.int32)
    return cv2.pointPolygonTest(quad_np, (px, py), False) >= 0  
    
def find_best_province(text):
    if text:
        
        score = []
        for province in provinces:
            score.append(similar(text, province))

        return [str(provinces[score.index(max(score))])]
    else:
        return ["cant ocr province"]
        

def click_and_collect_points(event, x, y, flags, param):
    global points, lenpoints
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        if len(points) < lenpoints:  # Limit the number of points to lenpoints
            points.append((x, y))  # Store the point
            print(f"Point {len(points)}: ({x}, {y})")

points = []
lenpoints = 4 

# read province name 
with open("provinces.txt", "r", encoding="utf-8") as file:
    provinces = file.read().strip().splitlines() 
#print(provinces)

def create_empty_image(quality=20):
    image = Image.new('RGB', (1, 1), (0, 0, 0))  # Red color in RGB format
    img_byte_arr = io.BytesIO()
    # Save the image
    image.save(img_byte_arr, format="JPEG", quality=quality)
    img_byte_arr.seek(0)
    return img_byte_arr

def compress_and_resize_img(img, width=600,quality=20):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Get the original dimensions of the image
    original_width, original_height = img_pil.size
    
    # Calculate the new height to maintain the aspect ratio
    new_height = int((width / original_width) * original_height)
    
    # Resize the image while maintaining the aspect ratio
    img_resized = img_pil.resize((width, new_height))
    
    
    img_byte_arr = io.BytesIO()
    
    # Save the resized and compressed image to the buffer (without saving to disk)
    img_resized.save(img_byte_arr, format="JPEG", quality=quality)
    
    # Seek to the beginning of the BytesIO object so it can be read
    img_byte_arr.seek(0)
    
    # Load the image again from the byte array into a PIL Image object
    #compressed_img = Image.open(img_byte_arr)
    
    # Return the compressed PIL image object
    return img_byte_arr
    
def read_data(url,codename):
    data = {
        "codename": codename
    }
    try:
        response = requests.post(url, data=data)
        response_json = response.json()
        status = response_json.get("status", None)
        status_bool = True if status == "running" else False
        instance_id = response_json.get("instance_id",None)
        instance_config = response_json.get("instance_config", {})
        point_region = instance_config.get("point_region", None)

        if point_region:
            point_region = [tuple(p) for p in point_region]

        url = instance_config.get("url", "")
        #print(json.dumps(response_json, indent=4))
        
        return status_bool,url,point_region,instance_id
    except Exception as e:
        print(f"Error in read data function {e}")

    # print(json.dumps(response_json, indent=4))
    # try:
    #     # Open and read the JSON file
    #     with open(file_path, 'r') as f:
    #         data = json.load(f)  # Parse JSON content
    #         # Extract the status
    #         status = data.get("status", None)
    #         url = data.get("camera_url", None)
    #         point = data.get("point" , None)
    #         point = [tuple(p) for p in point]
            
    #         return status,url,point
    # except FileNotFoundError:
    #     print(f"Error: {file_path} not found.")
    # except json.JSONDecodeError:
    #     print("Error: Failed to decode JSON. Please check the file format.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    # return None

    
def send_data():
    pass
    
def send_to_server(url,instance_id,codename,vehicle_img,plate_img,obj_id,dataplate):
    if plate_img == False:
        plate = False
    else:
        plate = True
    data_dict = {
        "instance_id": instance_id,
        "dataplate": {
            "type": str(dataplate["type"]),
            "alpha": str("".join(dataplate["alpha"])),
            "number": str("".join(dataplate["number"])),
            "province": str("".join(dataplate["province"])),
            "plate": plate,
        }
    }

    # Convert dictionary to JSON string
    data = {
        "data": json.dumps(data_dict),
        "token":codename
    }
    if plate_img == False:
        files = {
            "car": (f"car_{obj_id}.jpg", vehicle_img, "image/jpeg")
        }
    else:
        files = {
            "plate": (f"plate_{obj_id}.jpg", plate_img, "image/jpeg"),
            "car": (f"car_{obj_id}.jpg", vehicle_img, "image/jpeg")
        }

    #print(dataplate)
    
   
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print(f"Successfully uploaded! {obj_id}")
    else:
        print(f"Failed to upload. Status code: {response.status_code}")
    
    print("Response text:", response.text)
        
        
def split_ocr_text(ocr_results,vehicle_type,class_name):
    dataplate = {}
    for img_idx, result in enumerate(ocr_results):
        texts = [text for _, text, _ in result]  # ดึงเฉพาะข้อความ
        if texts==[]:
            texts = ["-"]
        if img_idx==0:
            if vehicle_type == "car":
                if len(texts) >= 2:
                    dataplate["alpha"] = texts[0]
                    dataplate["number"] = texts[1]
                else:
                    dataplate["alpha"] = texts
                    dataplate["number"] = "-"
            elif vehicle_type == "bike":
                dataplate["alpha"] = texts
        elif img_idx==1:
            #print("ocr province",texts)
            texts = find_best_province(''.join(texts))
            dataplate["province"]=texts
        elif img_idx==2:
            dataplate["number"] = ''.join(texts)
            
    #dataplate["instance_id"]=instance_id  
    dataplate["type"]=class_name       
        #print(f"Line {img_idx+1}: {''.join(texts)}")
    
    #print(dataplate)
    
    return dataplate
    
def denormalize_points(norm_points, frame_width, frame_height):
   
    return [(int(x * frame_width), int(y * frame_height)) for x, y in norm_points]


def check_video(video_path):
    print(f"check video URL: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    if ret:
        print("Check video URL: PASS ✅")
        return True
    else:
        print("Check video URL: FAILED ❌")
        print("Try again in 3 sec.")
        # time.sleep(5)
        return False
    
def get_point_expand(x1, y1, width, height, frame_width,frame_height,expand_ratio=0.5):
    

    expand_w = int(width * expand_ratio)
    expand_h = int(height * expand_ratio)

    new_x1 = max((x1 - expand_w),0)
    new_y1 = max((y1 - expand_h),0)
    new_x2 = min((x1 + width + expand_w),frame_width)
    new_y2 = min((y1 + height + expand_h),frame_height)

    return new_x1, new_y1, new_x2, new_y2

def get_plate(plate_cls,keypoints,cropped_vehicle,plate_label,dst_w,dst_h):
    x1, y1 = keypoints[0][0]
    x2, y2 = keypoints[0][2]
    x3, y3 = keypoints[0][7]
    x4, y4 = keypoints[0][5]
    
    dst_w, dst_h = dst_w[plate_cls], dst_h[plate_cls]
    
    # Apply perspective transformation
    pts_src = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts_dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    matrix, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 4)
    warped = cv2.warpPerspective(cropped_vehicle, matrix, (dst_w, dst_h))

    return warped

def ocr_type_car(plate_img,plate_cls,dst_h,reader,thai_allowlist):
    ocr_obj = []
    preprocessing = preprocess_for_ocr(plate_img)
    # 70:30 vertical split for normal_car
    split_height_1 = int(dst_h[plate_cls] * 0.7)
    
    # Split into two parts: top 70% and bottom 30%
    top_part = preprocessing[:split_height_1, :]  # Top 70%
    province_part = preprocessing[split_height_1:, :]  # Bottom 30%
    #top_part = preprocess_for_ocr(top_part)
    #province_part = preprocess_for_ocr(province_part)
    
    ocr_obj.extend([top_part, province_part])
    ocr_results = []
    for idx, img in enumerate(ocr_obj):
        if idx == 0:  # บน อักษร + ตัวเลข
            result = reader.readtext(img,allowlist='กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789')
        elif idx == 1:  # จังหวัด
            result = reader.readtext(img)

        ocr_results.append(result)
    return ocr_results

def ocr_type_bike(plate_img,plate_cls,dst_h,reader,thai_allowlist):
    ocr_obj = []
    preprocessing = preprocess_for_ocr(plate_img)
                            
    # 33:33:33 vertical split for normal_bike
    split_height_1 = int(dst_h[plate_cls] * 0.33)
    split_height_2 = int(dst_h[plate_cls] * 0.66)

    # Split into three parts: top 33%, middle 33%, bottom 33%
    top_part = preprocessing[:split_height_1, :]  # Top 33%
    province_part = preprocessing[split_height_1:split_height_2, :]  # Middle 33%
    bottom_part = preprocessing[split_height_2:, :]  # Bottom 33%
    
    ocr_obj.extend([top_part,province_part, bottom_part])
    ocr_results = []
    
    for idx, img in enumerate(ocr_obj):
        
        if idx == 0:  #  บน
            result = reader.readtext(img, allowlist="กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ")
        elif idx == 1:  # จังหวัด
            result = reader.readtext(img, allowlist=thai_allowlist)
        elif idx == 2:  # เลข
            result = reader.readtext(img, allowlist="0123456789")
        
        #print(result)
        ocr_results.append(result)
    
    return ocr_results

def resize_for_display(img,fix_height):
    h, w = img.shape[:2]  # Get original height and width
    new_w = int(fix_height*(w/h))  
    display = cv2.resize(img,(new_w, fix_height))
    return display

def load_settings(filename="setting.json"):
    with open(filename, "r") as f:
        return json.load(f)
    
def get_class_name(class_id):
    """Return the class name for a given class ID."""
    class_map = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }
    return class_map.get(class_id, "Unknown")