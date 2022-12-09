import cv2
import numpy as np
import pyrealsense2 as rs

from centersnap import CenterSnap
from centersnap.utils import load_img_NOCS

model_path = "models/CenterSnap_sim.onnx"

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,640,360,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,640,360,rs.format.z16,30)

pipe.start(config)

try:
    while True:
        frames = pipe.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if depth is None or color is None:
            continue

        depth_image = np.asarray(depth.get_data())
        color_image = np.asarray(color.get_data())
        depth_norm = np.array(depth_image, dtype=np.float32)/255.0
        #rgb_img, depth_norm, actual_depth = load_img_NOCS(img_path, depth_path)

        # Initialize pose estimator
        poseEstimator = CenterSnap(model_path)

        # Update pose estimator
        poseEstimator(color_image, depth_norm)

        # Draw object position heat map
        combined_img = poseEstimator.draw_heatmap(color_image, alpha=0.5)

        #cv2.imwrite("heatmap.png", combined_img)

        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        cv2.imshow("detection", combined_img)
        #cv2.imshow("depth", depth_norm)
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

    exit(0)

except Exception as e:
    print(e)
    pass

finally:
    pipe.stop()
    cv2.destroyAllWindows()