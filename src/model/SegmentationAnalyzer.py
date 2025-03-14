import cv2

class SegmentationAnalyzer():


    def get_connected_components(self, image):
        num_labels, labels, area_stats, centroids= cv2.connectedComponentsWithStats(image)
        return num_labels, labels, area_stats, centroids
    
    def add_annotations(self, image, centroids):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for i in range(1, len(centroids)):
            x, y = int(centroids[i][0]), int(centroids[i][1])

            cv2.circle(image_rgb, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        return image_rgb