import cv2
import numpy as np
import pyrealsense2 as rs

from centersnap import CenterSnap

model_path = "models/CenterSnap_sim.onnx"
poincloud_estimator_path = "models/CenterSnapAE_sim.onnx"

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
        poseEstimator = CenterSnap(model_path, poincloud_estimator_path, min_conf=0.5)

        # Update pose estimator
        ret = poseEstimator(color_image, depth_norm)

        if ret:

            # Draw projected points and boxes into the rgb image
            combined_img = poseEstimator.draw_points_2d(color_image)

            #cv2.imwrite("pose2d.png", combined_img)

            cv2.namedWindow("Projected points", cv2.WINDOW_NORMAL)
            cv2.imshow("Projected points", combined_img)
            key = cv2.waitKey(0)
        
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