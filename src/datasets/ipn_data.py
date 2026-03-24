from torch.utils.data import Dataset
import glob
import cv2
import os

from dotenv import load_dotenv
from random import randint
import tqdm

load_dotenv()

class IPNData(Dataset):
    def __init__(self, source_path):
        self.source_path = source_path 
        self.paths = glob.glob(f'{self.source_path}/videos/videos/*')
        self.load_data()


    def load_data(self):
        file = open(f"{self.source_path}/annotations/annotations/class_details.txt")
        text = file.read().split('\n')
        label_id_dict = dict()
        for row in text:
            info = row.split('\t')[:3]
            label_id_dict[info[0]] = info[1:] 

        file = open(f"{self.source_path}/annotations/annotations/Annot_List.txt")
        text = file.read().split('\n')
        video_info_dict = dict()
        for row in text:
            info = row.split(',')
            try:
                video_info_dict[info[0]].append(info[1:]) 
            except:
                video_info_dict[info[0]] = []
                video_info_dict[info[0]].append(info[1:]) 

        # Correct info dicts
        new_label_id_dict = {}
        new_video_info_dict = {}

        for k, v in label_id_dict.items():
            if k.isdigit() and len(v) > 0:
                new_label_id_dict[f"{int(k) - 1}"] = v

        self.label_id_dict = new_label_id_dict

        for k, v in video_info_dict.items():
            for idx, video_fragment in enumerate(v):
                label, id, t_start, t_end, frames = video_fragment
                if id.isdigit():
                    new_video_info_dict[(k, idx)] = [label, f"{int(id) - 1}", t_start, t_end, frames]

        video_info_dict = new_video_info_dict

        self.video_info = video_info_dict


    def get_video_info(self):
        return self.video_info
    
    def get_label_id_dict(self):
        return self.label_id_dict
    
    def get_num_videos(self):
        return len(glob.glob(f'{self.source_path}/videos/videos/*'))

    def get_frames(self, path, info):
        frames = []
        label, ID, start_frame, end_frame, num_frames = info
        
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        frame_no = start_frame

        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()

        # Loop over the frames
        while ret:

            if start_frame <= frame_no <= end_frame:
                frames.append(frame)

            frame_no += 1


            # Read the next frame
            ret, frame = cap.read()

        # Release the video capture
        cap.release()
        return frames
    

    def __getitem__(self, video_ix, section_ix):
        path = self.paths[video_ix]
        video_name = str(path).split('/')[-1][:-4]
        info = self.video_info[(video_name, section_ix)]
        label, ID, start_frame, end_frame, num_frames = info
        frames = self.get_frames(path, info)
        return frames, label, ID
    
    def __len__(self):
        return len(self.paths)
    
    def choose(self): return self[randint(len(self))]