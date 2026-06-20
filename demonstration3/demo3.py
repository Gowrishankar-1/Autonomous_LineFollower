import rclpy
import message_filters
import cv2
import numpy as np
import math
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class Demo3(Node):
    def __init__(self):
        super().__init__('demo3')
        self.get_logger().info('Starting Demo 3 Module')

        self.bridge = CvBridge()
        self.move_pub = self.create_publisher(TwistStamped, '/cmd_vel', 1)

        self.image = None

        # Beacon (stop sign) state
        self.stopped_for_beacon = False
        self.beacon_stop_done = False
        self.stop_start_time = None

        # Post-event timers
        self.slow_until = 0.0
        self.pause_until = 0.0

        # Lane memory
        self.last_lane_centre = None
        self.last_seen_both = 0.0

        # Robot pose from /odom: (x, y, yaw_radians) or None until first message
        self.robot_pose = None

        # Telemetry throttle counter
        self.log_counter = 0

        # Thresholds
        self.OBSTACLE_STOP_DEPTH = 0.55
        self.BEACON_TRIGGER_DEPTH = 0.70
        self.BEACON_STOP_DURATION = 5.0

        self.create_subscription(Odometry, '/odom', self.callback_odom, 10)

        colour_sub = message_filters.Subscriber(self, Image, "/camera/image_raw")
        depth_sub = message_filters.Subscriber(self, Image, "/camera/depth")
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

    def callback_odom(self, msg):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.robot_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
        )

    def stop_robot(self):
        twist = TwistStamped()
        twist.twist.linear.x = 0.0
        twist.twist.angular.z = 0.0
        self.move_pub.publish(twist)

    def clean_mask(self, mask):
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def find_lane_line(self, mask, colour_image, draw_colour):
        h, w, _ = colour_image.shape
        search_top = int(h * 0.60)
        search_bot = int(h * 0.85)
        mask[0:search_top, :] = 0
        mask[search_bot:h, :] = 0

        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(colour_image, (cx, cy), 10, draw_colour, -1)
            return True, cx, cy
        return False, 0, 0

    def find_full_image_centroid(self, mask, colour_image, draw_colour):
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(colour_image, (cx, cy), 10, draw_colour, -1)
            return True, cx, cy
        return False, 0, 0

    def sample_depth(self, depth_image, x, y, win=8):
        dh, dw = depth_image.shape
        x0 = max(0, x - win)
        x1 = min(dw, x + win + 1)
        y0 = max(0, y - win)
        y1 = min(dh, y + win + 1)
        patch = depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return float('inf')
        return float(np.nanmin(valid))

    def callback_camera(self, colour_msg, depth_msg):
        colour_image = self.bridge.imgmsg_to_cv2(colour_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        h, w, _ = colour_image.shape
        dh, dw = depth_image.shape
        now = time.time()

        hsv = cv2.cvtColor(colour_image, cv2.COLOR_BGR2HSV)

        # 1. BLUE BEACON
        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_mask = self.clean_mask(blue_mask)

        blue_found, bx, by = self.find_full_image_centroid(
            blue_mask, colour_image, (255, 0, 0)
        )

        beacon_depth = float('inf')
        if blue_found:
            bx_d = int(bx * dw / w)
            by_d = int(by * dh / h)
            beacon_depth = self.sample_depth(depth_image, bx_d, by_d, win=8)

        if (
            blue_found
            and beacon_depth < self.BEACON_TRIGGER_DEPTH
            and not self.stopped_for_beacon
            and not self.beacon_stop_done
        ):
            self.stopped_for_beacon = True
            self.stop_start_time = now
            self.get_logger().info('Blue beacon is close - starting 5 second stop')

        if self.stopped_for_beacon:
            elapsed = now - self.stop_start_time
            if elapsed < self.BEACON_STOP_DURATION:
                self.stop_robot()
                self.image = colour_image
                self.get_logger().info(
                    f'Stopping at beacon: {elapsed:.1f} / {self.BEACON_STOP_DURATION:.0f} seconds'
                )
                return
            else:
                self.stopped_for_beacon = False
                self.beacon_stop_done = True
                self.slow_until = now + 3.0
                self.get_logger().info('Finished beacon stop - continuing')

        # 2. OBSTACLE / PEDESTRIAN
        y0_d, y1_d = int(dh * 0.45), int(dh * 0.70)
        x0_d, x1_d = int(dw * 0.40), int(dw * 0.60)
        front_depth = depth_image[y0_d:y1_d, x0_d:x1_d]

        y0_c, y1_c = int(h * 0.45), int(h * 0.70)
        x0_c, x1_c = int(w * 0.40), int(w * 0.60)
        front_blue = blue_mask[y0_c:y1_c, x0_c:x1_c]

        if front_blue.shape != front_depth.shape:
            front_blue = cv2.resize(
                front_blue,
                (front_depth.shape[1], front_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        keep = (front_blue == 0) & np.isfinite(front_depth)
        valid_depth = front_depth[keep]

        if valid_depth.size > 50:
            obstacle_depth = float(np.nanmin(valid_depth))
            if obstacle_depth < self.OBSTACLE_STOP_DEPTH:
                self.stop_robot()
                self.image = colour_image
                self.pause_until = now + 1.0
                self.slow_until = now + 4.0
                self.get_logger().info(
                    f'Obstacle detected at {obstacle_depth:.2f} m - stopping'
                )
                return

        if now < self.pause_until:
            self.stop_robot()
            self.image = colour_image
            self.get_logger().info('Waiting briefly after obstacle cleared')
            return

        forward_speed = 0.06 if now < self.slow_until else 0.10

        # 3. LANE FOLLOWING
        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([40, 255, 255])
        red_lower_1 = np.array([0, 80, 80])
        red_upper_1 = np.array([10, 255, 255])
        red_lower_2 = np.array([170, 80, 80])
        red_upper_2 = np.array([180, 255, 255])

        yellow_mask = self.clean_mask(cv2.inRange(hsv, yellow_lower, yellow_upper))
        red_mask = self.clean_mask(
            cv2.inRange(hsv, red_lower_1, red_upper_1)
            + cv2.inRange(hsv, red_lower_2, red_upper_2)
        )

        yellow_found, yx, yy = self.find_lane_line(
            yellow_mask, colour_image, (0, 255, 255)
        )
        red_found, rx, ry = self.find_lane_line(
            red_mask, colour_image, (0, 0, 255)
        )

        twist = TwistStamped()
        err = 0.0

        if red_found and yellow_found:
            lane_centre = (rx + yx) / 2.0
            lane_y = int((ry + yy) / 2.0)
            self.last_lane_centre = lane_centre
            self.last_seen_both = now

            cv2.circle(colour_image, (int(lane_centre), lane_y), 10, (0, 255, 0), -1)

            err = lane_centre - w / 2.0
            twist.twist.linear.x = forward_speed
            twist.twist.angular.z = -float(err) / 650.0
            self.move_pub.publish(twist)

        elif yellow_found and not red_found:
            if self.last_lane_centre is not None and now - self.last_seen_both < 1.0:
                err = self.last_lane_centre - w / 2.0
                twist.twist.linear.x = 0.04
                twist.twist.angular.z = -float(err) / 500.0
            else:
                twist.twist.linear.x = 0.02
                twist.twist.angular.z = 0.30
            self.move_pub.publish(twist)

        elif red_found and not yellow_found:
            if self.last_lane_centre is not None and now - self.last_seen_both < 1.0:
                err = self.last_lane_centre - w / 2.0
                twist.twist.linear.x = 0.04
                twist.twist.angular.z = -float(err) / 500.0
            else:
                twist.twist.linear.x = 0.02
                twist.twist.angular.z = -0.30
            self.move_pub.publish(twist)

        else:
            self.stop_robot()

        # 4. PER-FRAME TELEMETRY (lab-notes style, one fact per line, throttled)
        self.log_counter = (self.log_counter + 1) % 10
        if self.log_counter == 0:
            if self.robot_pose is not None:
                px, py, yaw = self.robot_pose
                self.get_logger().info(
                    f'Robot pose: x = {px:.2f} m, y = {py:.2f} m, '
                    f'heading = {math.degrees(yaw):.0f} deg'
                )
            else:
                self.get_logger().info('Robot pose: waiting for odometry')

            if yellow_found:
                self.get_logger().info(
                    f'Yellow line found at ({yx}, {yy})'
                )
            else:
                self.get_logger().info('Yellow line not found')

            if red_found:
                self.get_logger().info(
                    f'Red line found at ({rx}, {ry})'
                )
            else:
                self.get_logger().info('Red line not found')

            if red_found and yellow_found:
                self.get_logger().info(
                    f'Following lane centre at x = {int(self.last_lane_centre)}, '
                    f'steering error = {err:.0f} px'
                )

            if blue_found:
                self.get_logger().info(
                    f'Blue beacon at ({bx}, {by}), depth = {beacon_depth:.2f} m'
                )

            self.get_logger().info(
                f'Velocity command: linear.x = {forward_speed:.2f} m/s, '
                f'angular.z = {twist.twist.angular.z:.3f} rad/s'
            )

        self.image = colour_image


def main(args=None):
    rclpy.init(args=args)
    Demo3()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()