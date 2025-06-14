import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from src.shared.FileInfo import FileInfo
class SegmentationAnalyzer():

    def get_connected_components(self, image):
        num_labels, labels, area_stats, centroids= cv2.connectedComponentsWithStats(image)
        return num_labels, labels, area_stats, centroids
    
    def save_histogram_as_image(self, fig):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hist_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "histogram", "diameter_histogram.png")
        os.makedirs(os.path.dirname(hist_image_path), exist_ok=True)
        fig.savefig(hist_image_path)
        plt.close()
        histogram_image = Image.open(hist_image_path)
        return histogram_image
    
    def create_histogram(self, stats, file_info: FileInfo):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            scaled_areas, scaled_diameters = self._get_scaled_meassurements(stats, file_info)
            histogram_data = {
                "Area": scaled_areas,
                "Diameter": scaled_diameters
            }

            rice_rule_steps = int(np.ceil(2 * len(histogram_data["Diameter"]) ** (1 / 3))) 

            fig, ax = plt.subplots()            
            ax.hist(
                histogram_data["Diameter"], 
                bins=rice_rule_steps, 
                label="Diameter", 
                edgecolor='black'
                )
            ax.set_title("Particle Diameter Histogram")
            ax.set_xlabel("Diameter"+" ["+file_info.unit+"]")
            ax.set_ylabel("Frequency")
            ax.legend(title=f"Steps: {rice_rule_steps} (Rice-rule)")
            
            self.save_histogram_as_image(fig)
            return fig           
        except Exception as e:
            print("Error in creating histogram: ", e)
            return None
    
    def write_stats_to_txt(self, stats, file_info: FileInfo, particle_count, output_folder):
        try:
            scaled_areas, scaled_diameters = self._get_scaled_meassurements(stats, file_info)
            txtfile_path = os.path.join(output_folder, f"{file_info.file_name}_statistics.txt")
            os.makedirs(os.path.dirname(txtfile_path), exist_ok=True)
            
            with open(txtfile_path, "w", encoding="utf-8") as txtfile:
                txtfile.write("##############################\n")
                txtfile.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                txtfile.write("##############################\n")
                txtfile.write(f"{'Particle No.':<12}{'Area [' + file_info.unit + '²]':>20}{'Diameter [' + file_info.unit + ']':>20}\n")
                
                for label_idx in range(1, particle_count + 1):
                    area = scaled_areas[label_idx - 1]
                    diameter = scaled_diameters[label_idx - 1]
                    txtfile.write(f"{label_idx:<12}{area:>20.6f}{diameter:>20.6f}\n")
                
                txtfile.write("_______________________________\n")
                txtfile.write(f"{'Total count':<12}{'Mean Area [' + file_info.unit + '²]':>20}{'Mean Diameter [' + file_info.unit + ']':>20}"
                            f"{'Max Area [' + file_info.unit + '²]':>20}{'Max Diameter [' + file_info.unit + ']':>20}"
                            f"{'Min Area [' + file_info.unit + '²]':>20}{'Min Diameter [' + file_info.unit + ']':>20}\n")
                txtfile.write(f"{particle_count:<12}"
                            f"{np.mean(scaled_areas):>20.6f}{np.mean(scaled_diameters):>20.6f}"
                            f"{np.max(scaled_areas):>20.6f}{np.max(scaled_diameters):>20.6f}"
                            f"{np.min(scaled_areas):>20.6f}{np.min(scaled_diameters):>20.6f}\n")
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
                final_x, final_y = self.check_particle_distance(cX, cY, used_positions, min_distance, max_offset_attempts)
                used_positions.append((final_x, final_y))
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = final_x - text_width // 2
                text_y = final_y + text_height // 2
                cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

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
    

    
    def _get_scaled_meassurements(self, stats: np.ndarray, file_info: FileInfo):
        pixel_area = file_info.pixel_width * file_info.pixel_height
        scaled_areas = self.__get_pixel_areas(stats) * file_info.downsize_factor * pixel_area
        scaled_diameters = self.__get_diameters(stats) * file_info.downsize_factor * file_info.pixel_width 
        return scaled_areas, scaled_diameters

    def format_table_data(self, stats: np.ndarray, file_info: FileInfo, particle_count: int):
        if particle_count == 0:
            return {
                "Count":    [0, 0, 0, 0],  
                "Area":     [0, 0, 0, 0],  
                "Diameter": [0, 0, 0, 0]
            }
            
        scaled_areas, scaled_diameters = self._get_scaled_meassurements(stats, file_info)
        
        area_mean = np.mean(scaled_areas).round(2)
        area_max = np.max(scaled_areas).round(2)
        area_min = np.min(scaled_areas).round(2)
        area_std = np.std(scaled_areas).round(2)
        
        diameter_mean = np.mean(scaled_diameters).round(2)
        diameter_max = np.max(scaled_diameters).round(2)
        diameter_min = np.min(scaled_diameters).round(2)
        diameter_std = np.std(scaled_diameters).round(2)


        unit = " "+file_info.unit
        table_data = {
        "Count":    [particle_count, particle_count, particle_count, particle_count],  
        "Area":    [str(area_mean)+unit+"²", str(area_min)+unit+"²", str(area_max)+unit+"²", str(area_std)+unit+"²"],
        "Diameter":    [str(diameter_mean)+unit, str(diameter_min)+unit, str(diameter_max)+unit, str(diameter_std)+unit]
        }

        return table_data
