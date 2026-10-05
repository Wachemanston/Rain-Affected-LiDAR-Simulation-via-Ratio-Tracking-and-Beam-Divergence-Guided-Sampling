# Source - https://stackoverflow.com/a/10909842
# Posted by Alessandro Pezzato, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-20, License - CC BY-SA 4.0

#!/bin/bash

mcap_dir="/media/ycshih/T7 Shield/Data/20260122-weather-test/2026-01-22-itri-mastertrans-1-fog80-light1600-south-no-20-double/rosbag2_2026_01_22-14_42_27" # Fog 80
mcap_dir="/media/ycshih/T7 Shield/Data/20260122-weather-test/2026-01-22-itri-mastertrans-1-fog80-light200-south-no-20-double/rosbag2_2026_01_22-14_45_48" # Fog 80 02
mcap_dir="/media/ycshih/T7 Shield/Data/20260120-weather-test/2026-01-20-itri-mastertrans-1-fog80-light1600-south-yes-20-1/rosbag2_2026_01_20-10_45_22" # Fog 80 03

# Camera
declare -a img_topics=(
  "/sensing/camera/chimei_110_f/h265"
  "/sensing/camera/chimei_183_fl/h265"
  "/sensing/camera/chimei_110_b/h265"
  "/sensing/camera/chimei_183_bl/h265"
  "/sensing/camera/chimei_183_b/h265"
  "/sensing/camera/chimei_183_br/h265"
  "/sensing/camera/chimei_30_b/h265"
  "/sensing/camera/chimei_183_fr/h265"
  "/sensing/camera/chimei_183_f/h265"
  "/sensing/camera/chimei_30_f/h265"
)

source venv/bin/activate

for topic in "${img_topics[@]}"; do {
  echo "Topic \"$topic\" started";
  python mcap_img_publisher.py "$mcap_dir" $topic & pid=$!
  PID_LIST+=" $pid";
} done


# Point Clouds
declare -a pc_topics=(
  '/sensing/lidar/ouster_left'
  '/sensing/lidar/ouster_top'
  '/sensing/lidar/falcon_front'
  '/sensing/lidar/falcon_back'
  #'/localization/compensated_points_ouster_top'
  #'/sensing/radar/ars548_rear/detections'
  #'/sensing/radar/ars548_front/objects'
  '/sensing/lidar/ouster_right'
  #'/localization/compensated_points_falcon_front'
  #'/localization/filtered_points_ouster_top'
  '/sensing/lidar/ouster_back'
  #'/sensing/radar/ars548_front/detections'
  #'/localization/filtered_points_falcon_front'
  #'/sensing/radar/ars548_rear/objects'
)

for topic in "${pc_topics[@]}"; do {
  echo "Topic \"$topic\" started";
  python mcap_pc_publisher.py "$mcap_dir" $topic & pid=$!
  PID_LIST+=" $pid";
} done


trap "kill $PID_LIST" SIGINT

echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";

