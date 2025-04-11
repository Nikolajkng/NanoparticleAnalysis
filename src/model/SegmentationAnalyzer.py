import os
import cv2
import numpy as np
import csv

from shared.ScaleInfo import ScaleInfo
class SegmentationAnalyzer():

    def get_connected_components(self, image):
        num_labels, labels, area_stats, centroids= cv2.connectedComponentsWithStats(image)
        return num_labels, labels, area_stats, centroids
    
    
    def write_stats_to_txt(self, stats, scale_info, particle_count):
        try:
            scale_factor = scale_info.real_scale_length / scale_info.image_width if scale_info else 1
            scaled_areas = self.__get_pixel_areas(stats) * scale_factor
            scaled_diameters = self.__get_diameters(stats) * scale_factor
        
            base_dir = os.path.dirname(__file__)
            txtfile = os.path.join(base_dir, "..", "data", "statistics", "statistics.txt")
            with open(txtfile, "w", newline="", encoding="utf-8") as txtfile:          
                writer = csv.writer(txtfile, delimiter="\t")    
                writer.writerow(["Label", "Area", "Diameter"])
                for label_idx in range(1, particle_count):
                    label = str(label_idx)
                    area = scaled_areas[label_idx-1]
                    diameter = scaled_diameters[label_idx-1]
                    writer.writerow([label, f"{area:.6f}", f"{diameter:.6f}"])
                writer.writerow("#" * 20)
                writer.writerow(["Total count" ,"Mean Area", "Mean Diameter", "Max Area", "Max Diameter", "Min Area", "Min Diameter"])
                writer.writerow([particle_count, f"{np.mean(scaled_areas):.6f}", f"{np.mean(scaled_diameters):.6f}",
                                f"{np.max(scaled_areas):.6f}", f"{np.max(scaled_diameters):.6f}",
                                f"{np.min(scaled_areas):.6f}", f"{np.min(scaled_diameters):.6f}"])
        except Exception as e:
            print("Error in writing statistics to txt file: ", e)
    
    def add_annotations(self, image, centroids, min_distance=10, max_offset_attempts=5):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            font_scale = 0.3
            thickness = 1
            used_positions = []

            for i in range(1, len(centroids)):
                cX, cY = int(centroids[i][0]), int(centroids[i][1])
                label = str(i)

                # Try offsetting label if it's too close to previous ones
                final_x, final_y = self.check_particle_distance(cX, cY, used_positions, min_distance, max_offset_attempts)
                used_positions.append((final_x, final_y))

                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = final_x - text_width // 2
                text_y = final_y + text_height // 2
                cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
                cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, lineType=cv2.LINE_AA)

            return image_rgb

        except Exception as e:
            print("Error in add_annotations: ", e)
            return image
    
    def check_particle_distance(self, cX, cY, used_positions, min_distance=10, max_offset_attempts=5):
        offset_attempt = 0
        final_x, final_y = cX, cY

        while offset_attempt < max_offset_attempts:
            too_close = False
            for prev_x, prev_y in used_positions:
                distance = ((final_x - prev_x) ** 2 + (final_y - prev_y) ** 2) ** 0.5
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                return final_x, final_y

            offset_attempt += 1
            offset_amount = offset_attempt * 5 
            final_x = cX + offset_amount
            final_y = cY - offset_amount
        return final_x, final_y

    
    def __get_diameters(self, stats: np.ndarray):
        diameters = np.empty(stats.shape[0]-1)
        for label_idx in range(1, stats.shape[0]):
            width, height = stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]
            diameter = np.mean([width, height])  
            diameters[label_idx-1] = diameter
        return diameters
    
    def __get_pixel_areas(self, stats: np.ndarray):
        return stats[1:, cv2.CC_STAT_AREA] 
    

    def format_table_data(self, stats: np.ndarray, scale_info: ScaleInfo, particle_count: int):
        scale_factor = scale_info.real_scale_length / scale_info.image_width if scale_info else 1
        scaled_areas = self.__get_pixel_areas(stats) * scale_factor
        scaled_diameters = self.__get_diameters(stats) * scale_factor
        
        area_mean = np.mean(scaled_areas).round(2)
        area_max = np.max(scaled_areas).round(2)
        area_min = np.min(scaled_areas).round(2)
        area_std = np.std(scaled_areas).round(2)
        
        diameter_mean = np.mean(scaled_diameters).round(2)
        diameter_max = np.max(scaled_diameters).round(2)
        diameter_min = np.min(scaled_diameters).round(2)
        diameter_std = np.std(scaled_diameters).round(2)

        table_data = {
        "Count":    [particle_count, particle_count, particle_count, particle_count],  
        "Area":    [area_mean, area_min, area_max, area_std],  
        "Diameter":    [diameter_mean, diameter_min, diameter_max, diameter_std] 
        }

        return table_data
