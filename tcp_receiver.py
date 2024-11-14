import rclpy
from rclpy.node import Node
import socket 
import PIL
import PIL.Image
import struct
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Defining Socket 
        self.host = ''
        self.port = 12345

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.sock.bind((self.host, self.port)) 
        self.sock.listen(1)
        self.conn, _ = self.sock.accept()     
        self.packet_size = 1024

        self.get_logger().info('Camera TCP publisher has been started')

        self._camera_publisher = self.create_publisher(Image, f'camera_view', 10)

        self._timer = self.create_timer(1, self.timer_callback)
        self.bridge = CvBridge()

    def timer_callback(self):
        value = self.conn.recv(8)
		# file_size = struct.unpack('!Q', value)[0]
        file_size = int(value.decode())

        print("file size: ", file_size)

        with open("./output.png", 'wb') as img_file:
            received_bytes = 0
            while received_bytes < file_size:
                print("received: ", received_bytes, "; total: ", file_size)
                bytes_to_receive = self.packet_size if ((file_size - received_bytes)//self.packet_size != 0) else (file_size - received_bytes)
                data = self.conn.recv(bytes_to_receive)
                # assert(len(data) == bytes_to_receive)
                received_bytes += len(data)
                img_file.write(data)

            img_file.close()

            img = cv2.imread("./output.png")
            cv2.imshow("screen capture", img)
            cv2.waitKey(int(1/30*1000))
            # cv2.destroyAllWindows()

        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        self._camera_publisher.publish(msg)
        self.get_logger().info('Publishing camera view')


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()