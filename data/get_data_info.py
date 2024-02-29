from glob import glob
import numpy as np
def main():
    _delta_info = []
    with open("/home/iccs/Desktop/isense/events/intention_prediction/processed_data/detection_camera1/lane_changes.txt" , "r") as f:
        lines = f.readlines()
        lines = [x[:-1] for x in lines] #remove \n
        print(lines)
        for line in lines:
            man_start = line[21:28]
            man_end = line[28:35]
            print(man_start)
            print(man_end)
            _delta_info.append(int(man_start) - int(man_end))

    print(_delta_info)
    print(np.min(_delta_info))
    print(np.max(_delta_info))
    print(np.sum(np.abs(_delta_info) >= 30))
    
    print(np.median(_delta_info))
    print(len(_delta_info))

    print("Second max:{}".format(np.min(np.sort(_delta_info)[2:])))
    
if __name__ == "__main__":
    main()