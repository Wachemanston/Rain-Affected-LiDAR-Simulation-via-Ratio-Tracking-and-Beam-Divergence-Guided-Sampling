from pathlib import Path
import sys

from mcap_ros2.reader import read_ros2_messages
import rospy

from sensor_msgs import PointCloud2
from utils import get_all_mcap_files, PLAY_RATE, dir_to_stamps


class PCPublisher:	
    def __init__(self, directory_path, interval, topic, suffix=''):
        self.msgs = PCPublisher.get_all_msgs(directory_path, topic)
        print(f'There are {len(self.msgs)} imgs of {topic}')
        self.pub = rospy.Publisher(f'{topic}{suffix}', PointCloud2, queue_size=1)
        self.interval = (e_stamp - s_stamp) / len(self.msgs) * interval
        self.index = 0
    
    @staticmethod
    def get_all_msgs(directory_path, topic):
        mcap_files = get_all_mcap_files(directory_path)
        res = []
        for file_path in mcap_files:
            for msg in read_ros2_messages(file_path, topics=[topic]):
                ros_msg = msg.ros_msg
                if s_stamp > ros_msg.header.stamp:
                    continue
                res.append(PointCloud2(
                    header=ros_msg.header,
                    height=ros_msg.height,
                    width=ros_msg.width,
                    fields=ros_msg.fields,
                    is_bigendian=ros_msg.is_bigendian,
                    point_step=ros_msg.point_step,
                    row_step=ros_msg.row_step,
                    data=ros_msg.data,
                    is_dense=ros_msg.is_dense
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
    mcap_dir, topic = Path(sys.argv[1]), sys.argv[2]
    s_stamp, e_stamp = dir_to_stamps(mcap_dir)
    node_name = topic.replace('/', '_')[1:]
    suffix = '_single' if 'single' in str(mcap_dir) else ''
    rospy.init_node(f'{node_name}{suffix}')
    p = PCPublisher(mcap_dir, PLAY_RATE, topic, suffix)
    p.publish_loop()
