import cv2
import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt


 
# def load_gaze_data(csv_path): 
#     return pd.read_csv(csv_path)


 
def generate_heatmap(frame, gaze_points): 
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

    #add gaze points to the heatmap
    for x, y in gaze_points:
        # x,y = int(row['x']), int(row['y'])
        # if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]: 
        #     heatmap_img[y,x] += 1
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            heatmap[y, x] += 1

   
    #normalize heatmap -- for visualization 
    heatmap = cv2.GaussianBlur(heatmap, (25,25), 0)       #i think 99 was too aggressive
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)
        
    #color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # cv2.imwrite(output_path, heatmap_img)
    # print(f'Heatmap saved to {output_path}')

    return heatmap




# def plot_gaze_distribution(gaze_data):
#     plt.figure(figsize=(10,6))
#     sns.scatterplot(x=gaze_data['x'], y=gaze_data['y'], alpha=0.5)
#     plt.gca().invert_yaxis()
#     plt.title("Gaze Point Distribution")    
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate ") 
#     plt.show()  



# def main():
#     # csv_path = "../data/gaze_data.csv"
#     output_heatmap = "gaze_heatmap.jpg"
#     image_shape = (720, 1280, 3)

#     # gaze_data = load_gaze_data(csv_path=csv_path)
#     # generate_heatmap(gaze_data=gaze_data, image_shape=image_shape, output_heatmap=output_heatmap)
#     # plot_gaze_distribution(gaze_data)

#     heatmap = generate_heatmap(frame=np.zeros(image_shape, dtype=np.uint8), gaze_points=gaze_data[['x', 'y']].values)
#     cv2.imwrite(output_heatmap, heatmap)


# if __name__ == "__main__":
#     main()