import rclpy
import message_filters
import cv2
import numpy as np
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge


class Demo3(Node):
    def __init__(self):
        super().__init__('demo3')
        self.get_logger().info('Starting Demo 3 Module')

        self.bridge = CvBridge()
        self.move_pub = self.create_publisher(TwistStamped, '/cmd_vel', 1)

        self.image = None

        self.stopped_for_beacon = False
        self.stop_start_time = None

        self.slow_until = 0.0
        self.pause_until = 0.0

        self.last_lane_centre = None
        self.last_seen_both = 0.0

        colour_image_topic = "/camera/image_raw"
        depth_image_topic = "/camera/depth"

        colour_sub = message_filters.Subscriber(self, Image, colour_image_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_image_topic)

        self.camera_sync = message_filters.ApproximateTimeSynchronizer(
            [colour_sub, depth_sub], 10, 0.1
        )
        self.camera_sync.registerCallback(self.callback_camera)

        self.get_logger().info('Waiting for camera images...')

        while rclpy.ok():
            if self.image is not None:
                cv2.imshow("Demo 3 Robot View", self.image)
                cv2.waitKey(30)

            rclpy.spin_once(self, timeout_sec=1.0)

    def stop_robot(self):
        twist = TwistStamped()
        twist.twist.linear.x = 0.0
        twist.twist.angular.z = 0.0
        self.move_pub.publish(twist)

    def clean_mask(self, mask):
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def find_colour_centroid(self, mask, colour_image, draw_colour):
        h, w, d = colour_image.shape

        search_top = int(h * 0.25)
        mask[0:search_top, :] = 0

        M = cv2.moments(mask)

        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(colour_image, (cx, cy), 10, draw_colour, -1)

            return True, cx, cy

        return False, 0, 0

    def callback_camera(self, colour_msg, depth_msg):
        colour_image = self.bridge.imgmsg_to_cv2(
            colour_msg,
            desired_encoding='bgr8'
        )

        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding='passthrough'
        )

        h, w, d = colour_image.shape
        dh, dw = depth_image.shape
        now = time.time()

        hsv = cv2.cvtColor(colour_image, cv2.COLOR_BGR2HSV)

        # =====================================================
        # 1. PEDESTRIAN / OBSTACLE STOPPING
        # =====================================================
        front_region = depth_image[
            int(dh * 0.45):int(dh * 0.70),
            int(dw * 0.40):int(dw * 0.60)
        ]

        valid_depth = front_region[np.isfinite(front_region)]

        if valid_depth.size > 50:
            obstacle_depth = np.nanmin(valid_depth)

            if obstacle_depth < 0.55:
                self.stop_robot()
                self.image = colour_image

                self.pause_until = now + 1.0
                self.slow_until = now + 4.0

                self.get_logger().info(
                    f'Pedestrian/obstacle detected: {obstacle_depth:.2f} m - STOPPING'
                )
                return

        if now < self.pause_until:
            self.stop_robot()
            self.image = colour_image
            self.get_logger().info('Waiting briefly after obstacle cleared')
            return

        if now < self.slow_until:
            forward_speed = 0.06
        else:
            forward_speed = 0.10

        # =====================================================
        # 2. BLUE BEACON / STOP SIGN
        # =====================================================
        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([140, 255, 255])

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_mask = self.clean_mask(blue_mask)

        blue_pixels = cv2.countNonZero(blue_mask)

        blue_found, bx, by = self.find_colour_centroid(
            blue_mask,
            colour_image,
            (255, 0, 0)
        )

        self.get_logger().info(f'Blue pixels: {blue_pixels}')

        if blue_found:
            self.get_logger().info(f'Blue beacon detected at ({bx}, {by})')

        if blue_pixels > 800 and not self.stopped_for_beacon:
            self.stopped_for_beacon = True
            self.stop_start_time = now
            self.slow_until = now + 8.0

            self.get_logger().info(
                'Blue beacon detected - starting 5 second stop'
            )

        if self.stopped_for_beacon:
            elapsed = now - self.stop_start_time

            if elapsed < 5.0:
                self.stop_robot()
                self.image = colour_image

                self.get_logger().info(
                    f'Stopping at beacon: {elapsed:.1f}/5.0 seconds'
                )
                return

        # =====================================================
        # 3. RED + YELLOW LANE FOLLOWING
        # =====================================================
        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([40, 255, 255])

        red_lower_1 = np.array([0, 80, 80])
        red_upper_1 = np.array([10, 255, 255])

        red_lower_2 = np.array([170, 80, 80])
        red_upper_2 = np.array([180, 255, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_mask = self.clean_mask(yellow_mask)

        red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)

        red_mask = red_mask_1 + red_mask_2
        red_mask = self.clean_mask(red_mask)

        yellow_found, yellow_x, yellow_y = self.find_colour_centroid(
            yellow_mask,
            colour_image,
            (0, 255, 255)
        )

        red_found, red_x, red_y = self.find_colour_centroid(
            red_mask,
            colour_image,
            (0, 0, 255)
        )

        self.get_logger().info(
            f'red={red_found}, yellow={yellow_found}, blue={blue_found}'
        )

        twist = TwistStamped()

        if red_found and yellow_found:
            lane_centre = (red_x + yellow_x) / 2.0
            lane_y = int((red_y + yellow_y) / 2.0)

            self.last_lane_centre = lane_centre
            self.last_seen_both = now

            cv2.circle(
                colour_image,
                (int(lane_centre), lane_y),
                10,
                (0, 255, 0),
                -1
            )

            err = lane_centre - w / 2.0

            twist.twist.linear.x = forward_speed
            twist.twist.angular.z = -float(err) / 650.0

            self.move_pub.publish(twist)

            self.get_logger().info(
                f'Following lane centre - speed={forward_speed:.2f}, err={err:.1f}'
            )

        elif yellow_found and not red_found:
            if self.last_lane_centre is not None and now - self.last_seen_both < 1.0:
                err = self.last_lane_centre - w / 2.0

                twist.twist.linear.x = 0.04
                twist.twist.angular.z = -float(err) / 500.0

                self.get_logger().info(
                    'Only yellow found - using last lane centre'
                )
            else:
                twist.twist.linear.x = 0.02
                twist.twist.angular.z = 0.30

                self.get_logger().info(
                    'Only yellow found - searching left for red'
                )

            self.move_pub.publish(twist)

        elif red_found and not yellow_found:
            if self.last_lane_centre is not None and now - self.last_seen_both < 1.0:
                err = self.last_lane_centre - w / 2.0

                twist.twist.linear.x = 0.04
                twist.twist.angular.z = -float(err) / 500.0

                self.get_logger().info(
                    'Only red found - using last lane centre'
                )
            else:
                twist.twist.linear.x = 0.02
                twist.twist.angular.z = -0.30

                self.get_logger().info(
                    'Only red found - searching right for yellow'
                )

            self.move_pub.publish(twist)

        else:
            self.stop_robot()
            self.get_logger().info('No lane lines found - STOPPING')

        self.image = colour_image


def main(args=None):
    rclpy.init(args=args)
    node = Demo3()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()