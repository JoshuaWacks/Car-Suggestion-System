import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
import time
import GPUtil

class TestingLogicD2:
    parent_folder_train = '../../../data/Car Detection/Dataset_2/Data/Data/train/'
    parent_folder_test = '../../../data/Car Detection/Dataset_2/Data/Data/test/'
    num_samples_train = 269
    num_samples_train = 57


    def __resize_image_no_padding(self, image, model_type = 'Yolo'):
        if model_type == 'Yolo':
            yolo_image_size = (640,640)
            return cv2.resize(image,yolo_image_size)
        
    def __resize_with_padding(self, image, model_type = 'Yolo', padding_color=(0, 0, 0)):
        """
        Resizes an image to a target size with padding, maintaining aspect ratio.

        Args:
            image (np.array): The input image.
            padding_color (tuple): The color to use for padding (B, G, R).

        Returns:
            np.array: The resized and padded image.
        """
        h_orig, w_orig = image.shape[:2]
        if model_type == 'Yolo':
            yolo_image_size = (640,640)
        target_w, target_h = yolo_image_size

        # Calculate scaling factor
        scale_w = target_w / w_orig
        scale_h = target_h / h_orig
        scale = min(scale_w, scale_h)

        # Calculate new dimensions after scaling
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)

        # Resize the image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate padding
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Add padding
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=padding_color)
        return padded_image

    def __get_box_co_ords(self, tree, output_format = 'tuple'):
        # TODO Assumptions for now
        # There is always a car
        # The first car has largest area

        root = tree.getroot()
        temp_obj = root.find('object')
        if temp_obj == None:
            return -1
        
        bnd_box = temp_obj.find('bndbox')
        if bnd_box == None:
            return -1
        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        if output_format == 'tuple':
            return (x_min,y_min),(x_max,y_max)
        else:
            return [x_min,y_min,x_max, y_max]
    
    def __get_area_from_box(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width*height

    def __get_model_pred(self, results, model_type = 'Yolo', metric='confidence'):
        if model_type == 'Yolo':
            car_preds = [pred for pred in results.pred[0].cpu().numpy() if pred[-1] ==2]
            
        elif model_type == "Ultra_Yolo":
            car_preds = [pred for pred in results[0].boxes.data.cpu().numpy() if pred[-1] ==2]

        if len(car_preds) == 0:
            return -1,1
        if metric == 'confidence':
            best_car_pred = car_preds[0]
            return best_car_pred[0:4], best_car_pred[4]
        if metric == 'area':
            biggest_area = 0
            best_car_pred = car_preds[0]
            for car_pred in car_preds:
                area = self.__get_area_from_box(car_pred)
                if area > biggest_area:
                    biggest_area = area
                    best_car_pred = car_pred

            return best_car_pred[0:4], best_car_pred[4]

    def __calculate_iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (list or np.array): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
            box2 (list or np.array): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].

        Returns:
            float: The IoU value.
        """
        if type(box2) == int and box2 == -1:
            if type(box1) == int and box1 == -1:
                return 1
            else:
                return 0

        # Determine the coordinates of the intersection rectangle
        x_min_intersection = max(box1[0], box2[0])
        y_min_intersection = max(box1[1], box2[1])
        x_max_intersection = min(box1[2], box2[2])
        y_max_intersection = min(box1[3], box2[3])

        # Calculate the area of the intersection
        intersection_width = max(0, x_max_intersection - x_min_intersection)
        intersection_height = max(0, y_max_intersection - y_min_intersection)
        intersection_area = intersection_width * intersection_height

        # Calculate the area of each bounding box
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate the union area
        union_area = area_box1 + area_box2 - intersection_area

        # Handle cases with no overlap or zero-area boxes
        if union_area == 0:
            return 0.0
        else:
            iou = intersection_area / union_area
            return iou
        
    def __get_metrics(self,iou_results, confidence_results, exec_time):
        gpu = GPUtil.getGPUs()[0]

        return {
            # "iou_results": iou_results,
            # "confidence_results": confidence_results,
            "avg_iou_results": np.sum(iou_results)/len(iou_results),
            "avg_confidence_results": np.sum(confidence_results)/len(confidence_results),
            "exec_time":exec_time,
            "GPU_memory": gpu.memoryUsed
        }
    
    def test_model(self, 
                   model,
                   params,
                   experiment_tags):

        model_type = params['model_type']
        resize_images = params['resize_images']
        resize_with_padding = params['resize_with_padding']
        car_choice_metric = params['car_choice_metric']

        dataset_to_use = params['dataset']
        
        iou_results = []
        confidence_results = []

        parent_folder = self.parent_folder_train 
        if dataset_to_use == 'test' :
            parent_folder = self.parent_folder_test 
        
        start_time = time.time()

        for i,file in enumerate(os.listdir(parent_folder)):
            if i % 2==0:
                full_image_path = os.path.join(parent_folder,file)
            else:
                full_label_path = os.path.join(parent_folder,file)

                image = cv2.imread(full_image_path)
                if resize_images:
                    if resize_with_padding:
                        image = self.__resize_with_padding(image, model_type)
                    else:
                        image = self.__resize_image_no_padding(image, model_type)

                tree = ET.parse(full_label_path)
                true_box = self.__get_box_co_ords(tree, output_format='array')

                results = model(image)

                pred_box, confidence = self.__get_model_pred(results,model_type,car_choice_metric)
                iou = self.__calculate_iou(box1=true_box,box2=pred_box)

                iou_results.append(iou)
                if confidence == 1:
                    if iou == 1:
                        confidence_results.append(confidence)
                    else:
                        confidence_results.append(0)
                else:
                    confidence_results.append(confidence)
                

                
        end_time = time.time()
        exec_time = np.round(end_time-start_time,5)
        print(len(iou_results))
        print(len(confidence_results))


        return self.__get_metrics(iou_results, confidence_results,exec_time)