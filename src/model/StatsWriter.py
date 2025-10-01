from src.shared.FileInfo import FileInfo
import os

class StatsWriter:
    def __init__(self):
        pass
    
    def __get_diameters(self, stats):
        import numpy as np
        from cv2 import CC_STAT_HEIGHT, CC_STAT_WIDTH
        diameters = np.empty(stats.shape[0]-1)
        for label_idx in range(1, stats.shape[0]):
            width, height = stats[label_idx, CC_STAT_WIDTH], stats[label_idx, CC_STAT_HEIGHT]
            diameter = np.mean([width, height])  
            diameters[label_idx-1] = diameter
        return diameters
    
    def __get_pixel_areas(self, stats):
        from cv2 import CC_STAT_AREA
        return stats[1:, CC_STAT_AREA] 
    
    def _get_scaled_meassurements(self, stats, file_info: FileInfo):
        pixel_area = file_info.pixel_width * file_info.pixel_height
        scaled_areas = self.__get_pixel_areas(stats) * file_info.downsize_factor * pixel_area
        scaled_diameters = self.__get_diameters(stats) * file_info.downsize_factor * file_info.pixel_width 
        return scaled_areas, scaled_diameters


    def write_stats_to_txt(self, stats, file_info: FileInfo, particle_count, output_folder):
        """Write statistics for a single image to a separate TXT file."""
        try:
            scaled_areas, scaled_diameters = self._get_scaled_meassurements(stats, file_info)
            txtfile_path = os.path.join(output_folder, f"{file_info.file_name}_statistics.txt")
            os.makedirs(os.path.dirname(txtfile_path), exist_ok=True)

            with open(txtfile_path, "w", encoding="utf-8") as txtfile:
                self._write_header(txtfile, file_info.unit)
                self._write_particle_data(txtfile, scaled_areas, scaled_diameters, particle_count, file_info.unit)
        except Exception as e:
            print("Error in writing statistics to txt file: ", e)

    def write_all_stats_to_txt(self, stats_list, file_info_list, output_folder, output_filename="all_statistics.txt"):
        """Write statistics for ALL images combined into one TXT file.
        Naively assumes that all images use the same unit."""
        import numpy as np
        try:
            all_areas = []
            all_diameters = []
            txtfile_path = os.path.join(output_folder, output_filename)
            os.makedirs(os.path.dirname(txtfile_path), exist_ok=True)
            global_unit = file_info_list[0].unit
            with open(txtfile_path, "w", encoding="utf-8") as txtfile:
                self._write_header(txtfile, global_unit)

                global_particle_idx = 1
                for stats, file_info in zip(stats_list, file_info_list):
                    scaled_areas, scaled_diameters = self._get_scaled_meassurements(stats, file_info)

                    for area, diameter in zip(scaled_areas, scaled_diameters):
                        txtfile.write(
                            f"{global_particle_idx:<12}{area:>20.6f}{diameter:>20.6f}    (Image: {file_info.file_name})\n"
                        )
                        all_areas.append(area)
                        all_diameters.append(diameter)
                        global_particle_idx += 1

                # Write global summary
                txtfile.write("_______________________________\n")
                txtfile.write(
                f"{'Total count':<12}{'Mean Area [' + global_unit + '²]':>20}{'Mean Diameter [' + global_unit + ']':>20}"
                f"{'Max Area [' + global_unit + '²]':>20}{'Max Diameter [' + global_unit + ']':>20}"
                f"{'Min Area [' + global_unit + '²]':>20}{'Min Diameter [' + global_unit + ']':>20}\n"
                )
                txtfile.write(
                    f"{len(all_areas):<12}"
                    f"{np.mean(all_areas):>20.6f}{np.mean(all_diameters):>20.6f}"
                    f"{np.max(all_areas):>20.6f}{np.max(all_diameters):>20.6f}"
                    f"{np.min(all_areas):>20.6f}{np.min(all_diameters):>20.6f}\n"
                )
        except Exception as e:
            print("Error in writing combined statistics to txt file: ", e)

    def _write_header(self, txtfile, unit):
        from datetime import datetime
        txtfile.write("##############################\n")
        txtfile.write("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        txtfile.write("##############################\n")
        txtfile.write(f"{'Particle No.':<12}{'Area [' + unit + '²]':>20}{'Diameter [' + unit + ']':>20}\n")

    def _write_particle_data(self, txtfile, scaled_areas, scaled_diameters, particle_count, unit):
        import numpy as np
        for label_idx in range(1, particle_count + 1):
            area = scaled_areas[label_idx - 1]
            diameter = scaled_diameters[label_idx - 1]
            txtfile.write(f"{label_idx:<12}{area:>20.6f}{diameter:>20.6f}\n")

        txtfile.write("_______________________________\n")
        txtfile.write(
            f"{'Total count':<12}{'Mean Area [' + unit + '²]':>20}{'Mean Diameter [' + unit + ']':>20}"
            f"{'Max Area [' + unit + '²]':>20}{'Max Diameter [' + unit + ']':>20}"
            f"{'Min Area [' + unit + '²]':>20}{'Min Diameter [' + unit + ']':>20}\n"
        )
        txtfile.write(
            f"{particle_count:<12}"
            f"{np.mean(scaled_areas):>20.6f}{np.mean(scaled_diameters):>20.6f}"
            f"{np.max(scaled_areas):>20.6f}{np.max(scaled_diameters):>20.6f}"
            f"{np.min(scaled_areas):>20.6f}{np.min(scaled_diameters):>20.6f}\n"
        )
