from pathlib import Path
import sys

from mcap_ros2.reader import read_ros2_messages
import rospy

from sensor_msgs import Image
from utils import get_all_mcap_files, PLAY_RATE, dir_to_stamps


class ImgPublisher:	
    def __init__(self, directory_path, interval, img_topic):
        self.msgs = ImgPublisher.get_all_msgs(directory_path, img_topic)
        print(f'There are {len(self.msgs)} imgs of {img_topic}')
        self.pub = rospy.Publisher(img_topic, Image, queue_size=1)
        self.interval = (e_stamp - s_stamp) / len(self.msgs) * interval
        self.index = 0
    
    @staticmethod
    def get_all_msgs(directory_path, img_topic):
        mcap_files = get_all_mcap_files(directory_path)
        res = []
        for file_path in mcap_files:
            for msg in read_ros2_messages(file_path, topics=[img_topic]):
                ros_msg = msg.ros_msg
                if s_stamp > ros_msg.header.stamp:
                    continue
                res.append(Image(
                    header=ros_msg.header,
                    height=ros_msg.height,
                    width=ros_msg.width,
                    encoding=ros_msg.encoding,
                    is_bigendian=ros_msg.is_bigendian,
                    step=ros_msg.step,
                    data=ros_msg.data
                ))
                if e_stamp < ros_msg.header.stamp:
                    return res
        return res
    
    def publish_loop(self):
        rate = rospy.Rate(1.0 / self.interval)
        N = len(self.msgs)
        while not rospy.is_shutdown() and self.index < N:
            try:
                msg = self.msgs[self.index]
                msg.header.stamp = rospy.Time.now()
                self.pub.publish(msg)
                self.index = (1 + self.index) % N
            except Exception as e:
                rospy.logerr(e)
            rate.sleep()


if __name__ == '__main__':
    mcap_dir, img_topic = Path(sys.argv[1]), sys.argv[2]
    s_stamp, e_stamp = dir_to_stamps(mcap_dir)
    rospy.init_node(img_topic.replace('/', '_')[1:])
    p = ImgPublisher(mcap_dir, PLAY_RATE, img_topic)
    p.publish_loop()
