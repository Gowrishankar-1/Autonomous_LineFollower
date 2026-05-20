import rclpy
import message_filters
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.get_logger().info('Starting Line Follower Module')

        self.bridge = CvBridge()
        self.move_pub = self.create_publisher(TwistStamped, '/cmd_vel', 1)

        colour_image_topic = "/camera/image_raw"
        depth_image_topic  = "/camera/depth"

        colour_sub = message_filters.Subscriber(self, Image, colour_image_topic)
        depth_sub  = message_filters.Subscriber(self, Image, depth_image_topic)

        self.camera_sync = message_filters.ApproximateTimeSynchronizer(
            [colour_sub, depth_sub], 10, 0.1
        )
        self.camera_sync.registerCallback(self.callback_camera)

        self.get_logger().info('Waiting for camera images...')
        self.image = None

        while rclpy.ok():
            if np.any(self.image):
                cv2.imshow("Line Follower View", self.image)
                cv2.waitKey(30)
            rclpy.spin_once(self, timeout_sec=1.0)

    def callback_camera(self, colour_msg, depth_msg):
        colour_image = self.bridge.imgmsg_to_cv2(colour_msg, desired_encoding='bgr8')
        depth_image  = self.bridge.imgmsg_to_cv2(depth_msg,  desired_encoding='passthrough')

        hsv = cv2.cvtColor(colour_image, cv2.COLOR_BGR2HSV)

        # Yellow HSV range
        lower_yellow = (10, 30, 30)
        upper_yellow = (50, 255, 255)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask,  None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        h, w, d = colour_image.shape

        # Search bottom half of image
        search_top = h // 2
        mask[0:search_top][:] = 0

        M = cv2.moments(mask)

        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(colour_image, (cx, cy), 10, (0, 0, 255), -1)
            self.image = colour_image

            err = cx - w / 2

            twist = TwistStamped()
            # Slower speed so robot doesn't overshoot corners
            twist.twist.linear.x  =  0.1
            # Stronger turning gain so it corrects faster
            twist.twist.angular.z = -float(err) / 500
            self.move_pub.publish(twist)
            self.get_logger().info(f'Following line - err={err:.1f}')

        else:
            # Stop if no yellow line detected
            twist = TwistStamped()
            twist.twist.linear.x  = 0.0
            twist.twist.angular.z = 0.0
            self.move_pub.publish(twist)
            self.image = colour_image
            self.get_logger().info('No yellow line - STOPPING')

def main(args=None):
    rclpy.init(args=args)
    lf = LineFollower()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
