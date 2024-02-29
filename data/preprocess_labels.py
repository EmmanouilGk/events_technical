def main():
    """
    rmv maneuver sequences <=20 frames for consistency
    """
    file = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt"

    with open(file , "r") as f:
        lines = f.readlines()[:-1]
    
    file_dstp = "/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes_preprocessed.txt"
    with open(file_dstp , "w") as f_out:
        for line in lines:
                man_start = line[21:28]
                man_end = line[28:35]
                if (int(man_end) - int(man_start))<6000:
                    f_out.write(line)


if __name__=="__main__":
    main()
    