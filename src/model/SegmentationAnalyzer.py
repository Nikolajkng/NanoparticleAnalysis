import cv2
import numpy as np

from shared.ScaleInfo import ScaleInfo
class SegmentationAnalyzer():

    def get_connected_components(self, image):
        num_labels, labels, area_stats, centroids= cv2.connectedComponentsWithStats(image)
        return num_labels, labels, area_stats, centroids
    
    def add_annotations(self, image, centroids):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for i in range(1, len(centroids)):
            x, y = int(centroids[i][0]), int(centroids[i][1])

            cv2.circle(image_rgb, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
        return image_rgb
    
    def __get_diameters(self, stats: np.ndarray):
        diameters = np.empty(stats.shape[0]-1)
        for label_idx in range(1, stats.shape[0]):
            width, height = stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]
            diameter = np.mean([width, height])  # TODO: Find better approximation of diameter
            diameters[label_idx-1] = diameter
        return diameters
    
    def __get_pixel_areas(self, stats: np.ndarray):
        return stats[1:, cv2.CC_STAT_AREA] 
    

    def format_table_data(self, stats: np.ndarray, scale_info: ScaleInfo):
        scale_factor = scale_info.real_scale_length / scale_info.image_width
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
        "Area":    [area_mean, area_min, area_max, area_std],  
        "Diameter":    [diameter_mean, diameter_min, diameter_max, diameter_std] 
        }

        return table_data
