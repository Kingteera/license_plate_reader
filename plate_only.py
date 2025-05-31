from ultralytics import YOLO
import torch
import os
import easyocr
import threading
from plate_ocr_func import *
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")


def runyolo(cv2_display,url,plate_model_path):
    
    try:
        global thread_running
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Using device: {device}")

        # Load  license plate detection
        plate_model = YOLO(plate_model_path).to(device)  # License plate detection model
        
        reader = easyocr.Reader(['th', 'en'],gpu=False)
        plate_label = {
                0: "normal_car",
                1: "normal_bike",
                2: "long_bike",
                3: "truck_car",
                4: "small_bike"
            }
            # Define expected license plate sizes based on real-world dimensions
        dst_h = {"normal_car": 300, "normal_bike": 344, "long_bike": 280, "truck_car": 440, "small_bike": 440}
        dst_w = {"normal_car": 680, "normal_bike": 440, "long_bike": 680, "truck_car": 800, "small_bike": 400}

        cap = cv2.VideoCapture(url)
        frame_width = int(cap.get(3))  # Get video width
        frame_height = int(cap.get(4))  # Get video height    
        print(f"Resolution input: {frame_width}x{frame_height}")
        thai_allowlist = []
        cap = cv2.VideoCapture(url)
        while cap.isOpened() and thread_running:
            
            ret, frame = cap.read()
            cap.grab()  # Grab frame to reduce lag in RTSP streams

            if not ret:
                break  # Exit loop if no frame is read

            frame_ = frame.copy()  # Keep original frame for cropping
            plate_results = plate_model(frame, verbose=False, conf=0.4, iou=0.1)[0]

            if len(plate_results.boxes.data) > 0: # check if has plate
                if len(plate_results.keypoints.data[0]) == 8: # check plate has 8 keypoint
                    keypoints = plate_results.keypoints.data.cpu().numpy()
                    try:
                        # get plate class
                        plate_cls = plate_label[int(plate_results.boxes.cls[0])]
                        # use keypoint to find plate
                        plate_img = get_plate(plate_cls,keypoints,frame,plate_label,dst_w,dst_h)             
                        # can find plate --> store id in saved_track_ids
                        # saved_track_ids.add(tk_id)
                            
                        if "car" in plate_cls:
                            # call ocr func for vehicle type car
                            ocr_results = ocr_type_car(plate_img,plate_cls,dst_h,reader,thai_allowlist)
                            # send data to server
                            plate_compress = compress_and_resize_img(plate_img,200) # compress img with size 200
                            # vehicle_compress = compress_and_resize_img(vehicle_expand[i],600) # compress img with size 600
                            data = split_ocr_text(ocr_results,"car","car") # data format to send server
                            print(data)
                            # send_to_server(send_data_url,instance_id,codename,vehicle_compress,plate_compress,tk_id,data)
                            
                        elif "bike" in plate_cls:
                            ocr_results = ocr_type_bike(plate_img,plate_cls,dst_h,reader,thai_allowlist)
                            # send data to server
                            plate_compress = compress_and_resize_img(plate_img,200)  # compress img with size 200
                            # vehicle_compress = compress_and_resize_img(vehicle_expand[i],150) # compress img with size 150
                            data = split_ocr_text(ocr_results,"bike","bike")
                            print(data)
                            # send_to_server(send_data_url,instance_id,codename,vehicle_compress,plate_compress,tk_id,data)

                        # if cv2show mode
                        if cv2_display:
                            # display img 
                            # cropped_vehicle_display = resize_for_display(cropped_vehicle,400)
                            # cv2.imshow("cropped_vehicle",cropped_vehicle_display)
                            plate_display = resize_for_display(plate_img,250)
                            cv2.imshow("plate",plate_display)
                            # vehicle_expand_display  = resize_for_display(vehicle_expand[i],400)
                            # cv2.imshow("vehicle_expand_display",frame_)

                    except Exception as e:
                        print(f"Error in plate transformation: {e}")

                        continue

                else:
                    print(f"8 Expect keypoint : found {len(plate_results.keypoints.data[0])}")
            else:
                # if model cant find plate --> send vehicle img without plate img
                # vehicle_class = "car" if "car" in plate_cls else "bike" 
                # data = {"type": vehicle_class, "alpha": "-","number": "-","province":"-"} # plate data --> "-"
                # vehicle_compress = compress_and_resize_img(vehicle_expand[i],150)
                # plate_img = False # dont send plate img
                # send_to_server(send_data_url,instance_id,codename,vehicle_compress,plate_img,tk_id,data)
                print("vehicle dont have plate")
                # saved_track_ids.add(tk_id) # save track id
            if cv2_display:
                display_frame = resize_for_display(frame,400)
                cv2.imshow("Vehicle detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
        thread_running = False
    except Exception as e:
        print(e)


# if __name__ == "__main__":
global thread_running

thread_running = False

while True:

    
    # replace with your trigger
    with open("plate_setting.json", "r") as f:
        status = json.load(f)["read_plate"]
    print(status) # status of plate read trigger
    # --------------------------------------



    if status == True and not thread_running: 
        print("plate read start")
        cv2_display = True
        url = "test_vdo.mp4"
        plate_model_path = "test_plate.pt"
        thread_running = True
        # runyolo(cv2_display,url,plate_model_path)
        thread_run_yolo = threading.Thread(target=runyolo,args=(cv2_display,url,plate_model_path))
        thread_run_yolo.start()
        # thread_running = True
    elif status == True and thread_running:
        pass # already running
    elif status == False and thread_running:
        print("plate read stop requested")
        thread_running = False
    elif status == False and not thread_running:
        print("plate read already stopped")
    time.sleep(5)
