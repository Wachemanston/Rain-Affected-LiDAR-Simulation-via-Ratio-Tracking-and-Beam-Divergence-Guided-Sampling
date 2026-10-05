from collections import Counter
from pathlib import Path

import yaml
import numpy as np
from mcap_ros2.reader import read_ros2_messages
from pypcd4 import PointCloud
import av
from scipy.spatial.transform import Rotation


PLAY_RATE = 2.5


class Stamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = int(sec)
        self.nanosec= int(nanosec)
        
    def __lt__(self, stamp):
        if self.sec == stamp.sec:
            return self.nanosec < stamp.nanosec
        return self.sec < stamp.sec
        
    def __gt__(self, stamp):
        if self.sec == stamp.sec:
            return self.nanosec > stamp.nanosec
        return self.sec > stamp.sec
        
    def __eq__(self, stamp):
        return self.sec == stamp.sec and self.nanosec == stamp.nanosec
        
    def __sub__(self, stamp):
        return (self.sec - stamp.sec) + 1e-9 * (self.nanosec - stamp.nanosec)

    def __repr__(self):
        return f'{self.sec}.{self.nanosec}'
    

def list_topics_from_metadata(yaml_dir: Path) -> None:
    with open(yaml_dir / 'metadata.yaml', 'r') as f:
        # ROS 2 的 metadata yaml 有時候會有一些自定義標籤，用 SafeLoader 比較安全
        data = yaml.load(f, Loader=yaml.SafeLoader)
        
    print(f"{'Topic (主題)':<55} | {'Type (資料型別)':<50} | Cnt")
    print("-" * 105)
    
    # 遍歷 metadata 中記錄的所有 topics
    camera_cnt, pc2_cnt = 0, 0
    topics_info = data['rosbag2_bagfile_information']['topics_with_message_count']
    for item in topics_info:
        topic_name = item['topic_metadata']['name']
        msg_type = item['topic_metadata']['type']
        msg_cnt = item['message_count']
        print(f"{topic_name:<55} | {msg_type:<50} | {msg_cnt}")
        if msg_type == 'sensor_msgs/msg/PointCloud2':
            pc2_cnt += 1
        if msg_type == 'sensor_msgs/msg/Image':
            camera_cnt += 1
    print(f'Total {len(topics_info)} topics ({pc2_cnt} PC2, {camera_cnt} camera)')


def dir_to_stamps(suffix: Path) -> (Stamp, Stamp):
    suffix = suffix.name
    
    if suffix.endswith('rosbag2_2026_01_21-08_44_33'):  # clear north
        return Stamp(sec=1768956296), Stamp(sec=1768956312)
    elif suffix.endswith('rosbag2_2026_01_22-10_57_59'):  # clear south
        return Stamp(sec=1769050696), Stamp(sec=1769050706)
    elif suffix.endswith('rosbag2_2026_01_22-14_24_20'):  # Fog 40 south
        return Stamp(sec=1769063074), Stamp(sec=1769063092)
    elif suffix.endswith('rosbag2_2026_01_22-14_42_27'):  # Fog 80 south
        return Stamp(sec=1769064161), Stamp(sec=1769064180)
    elif suffix.endswith('rosbag2_2026_01_21-15_06_27'):  # Single 40
        return Stamp(sec=1768979187), Stamp(sec=1768979197)
    elif suffix.endswith('rosbag2_2026_01_21-15_17_37'):  # Dual 40
        return Stamp(sec=1768979857), Stamp(sec=1768979867)
    elif suffix.endswith('rosbag2_2026_01_21-15_00_48'):  # Single 80
        return Stamp(sec=1768978848), Stamp(sec=1768978858)
    elif suffix.endswith('rosbag2_2026_01_21-15_14_31'):  # Dual 80
        return Stamp(sec=1768979672), Stamp(sec=1768979682)
    elif suffix.endswith('rosbag2_2026_01_22-14_45_48'):  # Fog80 02
        return Stamp(sec=1769064359), Stamp(sec=1769064369)
    elif suffix.endswith('rosbag2_2026_01_20-10_45_22'):  # Fog80 03
        return Stamp(sec=1768877172), Stamp(sec=1768877185)
    else:
        print(f'No time range for {suffix}')
        return Stamp(sec=0), Stamp(sec=2000000000)


map_mode = {
    'single': 'single',
    'dual': 'dual',
    'double': 'double'
}


def parse_dirname(dirname: str ='2026-01-21-itri-mastertrans-1-fog40-light1600-X-no-0-dual') -> dict:
    meta = dirname.split('-')
    assert meta[3] == 'itri'
    
    weather = ''
    light_idx = 0
    for i in range(6, len(meta)):
        if meta[i].startswith('light'):
            light_idx = i
            break
        weather += f'{meta[i]}'
    
    assert meta[light_idx+1] in {'north', 'south', 'X'}
    assert meta[light_idx+2] in {'yes', 'no'}

    mode = 'single'
    if meta[light_idx+4] in {'single', 'double', 'dual'}:
        mode = map_mode[meta[light_idx+4]]
    else:
        assert float(meta[light_idx+4]) >= 0
        
    return {
        'date': f'{meta[0]}{meta[1]}{meta[2]}',
        'vehicle': f'{meta[4]}-{meta[5]}',
        'weather': weather,
        'light': int(meta[light_idx].replace('light', '')),
        'speed': float(meta[light_idx+3]),
        'mode': mode
    }


def stats_scenarios(dir_path: Path) -> None:
    weather_cnt = Counter()
    speed_cnt = Counter()
    
    for scene in dir_path.iterdir():
        if scene.is_dir():
            scene_class = parse_dirname(scene.name)
            weather_cnt[scene_class['weather']] += 1
            speed_cnt[scene_class['speed']] += 1
        else:
            print(scene)
    print(weather_cnt)
    print(speed_cnt)


def get_all_mcap_files(directory_path: Path) -> [Path]:
    mcap_files = list(sorted(directory_path.glob("*.mcap")))
    
    if not mcap_files:
        print("找不到任何 .mcap 檔案。")
        return []
    return mcap_files


def list_timestamp(directory_path: Path, target_topics: str, N=5) -> None:
    with open(directory_path / 'metadata.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    meta = data['rosbag2_bagfile_information']
    topic_infos = meta['topics_with_message_count']
    topic_to_msg_cnt = {info['topic_metadata']['name']: info['message_count'] for info in topic_infos}
    
    print('Duration', meta['duration']['nanoseconds'] / 1e9)
    print('Starting', meta['starting_time']['nanoseconds_since_epoch'] / 1e9)
        
    mcap_files = get_all_mcap_files(directory_path)
    for target_topic in target_topics:
        msg_cnt = topic_to_msg_cnt[target_topic]
        print(target_topic, '(', msg_cnt, '):')
        for file_path in mcap_files:
            cnt = 0
            for msg in read_ros2_messages(file_path, topics=[target_topic]):
                ros_msg = msg.ros_msg
                print('\t', ros_msg.header.stamp)
                cnt += 1
                if cnt >= N:
                    break
            if cnt >= N:
                break

def iter_timestamp_and_pc(directory_path: Path, target_topic: str):
    # 1. 找出並排序所有的 .mcap 分割檔
    mcap_files = get_all_mcap_files(directory_path)

    # 2. 定義 ROS 2 到 NumPy 的資料型別對應表
    # 對應 sensor_msgs/PointField 中的 DATATYPE 定義
    DTYPE_MAP = {
        1: np.int8,    2: np.uint8, 
        3: np.int16,   4: np.uint16,
        5: np.int32,   6: np.uint32, 
        7: np.float32, 8: np.float64
    }

    s_stamp, e_stamp = dir_to_stamps(directory_path)

    print(f"準備從 {len(mcap_files)} 個檔案中提取 Topic: {target_topic} / {s_stamp.sec} {e_stamp.sec}")

    # 3. 循序處理每一個 .mcap 檔案
    for file_path in mcap_files:
        print(f"--- 正在讀取: {file_path.stem} ---")
        
        try:
            # read_ros2_messages 會幫我們過濾 Topic 並解碼訊息
            for msg in read_ros2_messages(file_path, topics=[target_topic]):
                ros_msg = msg.ros_msg
                
                stamp = ros_msg.header.stamp
                if s_stamp > stamp or e_stamp < stamp:
                    continue
                
                # 4. 動態解析 PointCloud2 的欄位結構 (動態建立 NumPy 的 dtype)
                # 例如：可能會解析出 [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2')]
                numpy_dtypes = []
                for field in ros_msg.fields:
                    np_type = DTYPE_MAP.get(field.datatype, np.uint8)
                    numpy_dtypes.append((field.name, np_type))
                
                # 5. 極速轉換：將 bytes 直接映射成 NumPy 的 Structured Array
                # 這裡使用 frombuffer 避免在記憶體中複製龐大的陣列
                cloud_arr = np.frombuffer(ros_msg.data, dtype=numpy_dtypes)
                
                # 若有需要，可以將一維結構化陣列重塑 (reshape) 成 2D (例如結構化 LiDAR 掃描)
                # cloud_arr = cloud_arr.reshape((ros_msg.height, ros_msg.width))
                
                # 簡單印出結果證明轉換成功
                fields = cloud_arr.dtype.names
                types = [t for _, t in numpy_dtypes]
                
                points = []
                points = np.vstack([cloud_arr[col] for col in cloud_arr.dtype.names]).T
                
                # --- 這裡是你發揮演算法的地方 ---
                # 建議使用 yield 將 cloud_arr 傳遞出去給其他分析函數，或直接存成 .npy
                pc = PointCloud.from_points(points, fields=fields, types=types)
                yield  msg.log_time_ns, pc
                
        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤: {e}")
    

def extract_and_save_pcd(mcap_dir: Path, output_root_dir: Path, topic: str ='/sensing/lidar/ouster_top') -> None:
    pcd_dir = output_root_dir / mcap_dir.name
    pcd_dir.mkdir(exist_ok=True, parents=True)
    cnt = 0
    for nsec, pc in iter_timestamp_and_pc(mcap_dir, topic):
        pc.save(pcd_dir / f'{nsec}.pcd')
        cnt += 1
    print(f'{output_root_dir.name} has {cnt} pcds')
    

def print_topic_type(mcap_dir: Path, target_topic: str) -> None:
    mcap_files = get_all_mcap_files(mcap_dir)
    for file_path in mcap_files:
        for msg in read_ros2_messages(file_path, topics=[target_topic]):
            ros_msg = msg.ros_msg
            print(ros_msg)
            return


def transform(pc: PointCloud) -> PointCloud:
    y, p, r = -0.0032020896, 0.83862830012, 0
    rot_matrix = Rotation.from_euler('zyx', [y, p, r], degrees=False).as_matrix()
    p = pc.numpy()
    new_pos = p[:, 2:5] @ rot_matrix.T
    p[:, 2:5] = new_pos
    return PointCloud.from_points(p, fields=pc.fields, types=pc.types)


def extract_seqs(root_dir: Path, save_dir: Path, ignore_existed: bool = True) -> None:
    existed_seqs = [item.name for item in save_dir.iterdir() if item.is_dir()] if ignore_existed else []
    for seq_dir in root_dir.glob('*/*'):
        
        pcd_dir = save_dir / seq_dir.name
        if seq_dir.name in existed_seqs and not pcd_dir.exists():
            pcd_dir.mkdir(exist_ok=True, parents=True)
            extract_and_save_pcd(seq_dir, pcd_dir, topic='/sensing/lidar/ouster_top')


def print_weather_to_seq(root_dir: Path) -> None:
    for mcap in root_dir.glob('*/*/*.mcap'):
        seq_name = mcap.parent.parent.name.replace("-itri-mastertrans-1-", " ")
        print(f'{seq_name} -> {mcap.name}')


def save_imgs(mcap_dir: Path, img_dir: Path, img_topic: str ='/sensing/camera/chimei_110_f/h265') -> None:
    N = 100
    mcap_files = get_all_mcap_files(mcap_dir)
    s_stamp, e_stamp = dir_to_stamps(mcap_dir)
    res = []
    
    h265_codec = av.CodecContext.create('hevc', 'r')
    
    for file_path in mcap_files:
        for msg in read_ros2_messages(file_path, topics=[img_topic]):
            ros_msg = msg.ros_msg
            img_data = bytes(ros_msg.data)
            try:
                for packet in h265_codec.parse(img_data):
                    for frame in h265_codec.decode(packet):
                        
                        # 只有當時間大於或等於 s_stamp 時，才真正轉存成圖片
                        if not (s_stamp > ros_msg.header.stamp):
                            try:
                                ts_str = f"{ros_msg.header.stamp.sec}{ros_msg.header.stamp.nanosec:09d}"
                            except AttributeError:
                                ts_str = str(ros_msg.header.stamp)

                            img = frame.to_image()
                            img.save(Path(img_dir) / f"{ts_str}.png", format='PNG')
                            #img.save(Path(img_dir) / f"{ts_str}.pdf", format='PDF')
                            N -= 1
                            print(f'{ts_str} done - {img.size}')

            except av.error.InvalidDataError:
                # 忽略剛開始串流缺少關鍵影格 (I-frame) 所造成的錯誤
                pass
            except Exception as e:
                print(f"解碼發生其他錯誤: {e}")
            
            if e_stamp < ros_msg.header.stamp or N <= 0:
                break

