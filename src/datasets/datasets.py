from torch.utils.data import Dataset
import glob
import cv2
import os
import pandas as pd
import shutil

from fnmatch import fnmatch
from dotenv import load_dotenv
from random import randint
from tqdm import tqdm
from multiprocessing import Pool
import glob

from src.datasets.dataset_template import DatasetTemplate

load_dotenv()

class IPNData(DatasetTemplate):
    def __init__(self, source_path):
        super().__init__()
        self.source_path = source_path 
        self.paths = glob.glob(f'{self.source_path}/videos/videos/*')
        self._load_raw_data()


    def _load_raw_data(self):
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
    
    def label_to_index(self):
        label_to_index = {}

        for k in self.label_id_dict:
            label, gesture = self.label_id_dict[k]
            label_to_index[label] = int(k)

        return label_to_index
    

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



class MontalbanoData(DatasetTemplate):
    def __init__(self, source_path, target_path, load_data=True):
        self.source_path = source_path
        self.target_path = target_path
        self.filtered_folder_name = "filtered_videos"
        if load_data:
            self._extract_colout_mp4_samples()
            self._create_sequence_frames()
            self._create_montalbano_segments()

    def _extract_colout_mp4_samples(self):
        pattern = "*color.mp4"
        for folder in os.listdir(self.source_path):
            if os.path.isdir(os.path.join(self.source_path, folder)):
                filtered_folder = os.path.join(self.source_path, folder, self.filtered_folder_name)
                if not os.path.exists(filtered_folder):
                    os.makedirs(filtered_folder)
                for directory, _, files in os.walk(os.path.join(self.source_path, folder)):
                    for file in files:
                        if fnmatch(file, pattern):
                            src = os.path.join(directory, file)
                            dst = os.path.join(filtered_folder, file)
                            if src != dst:
                                shutil.copy2(src=src, dst=dst)

    """
    def _create_sequence_frames(self):
        for folder in os.listdir(self.source_path): #test, train etc.
            if os.path.isdir(os.path.join(self.source_path, folder)) and os.path.join(self.source_path, folder) != self.target_path:
                target_folder = os.path.join(self.target_path, folder)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                for file in tqdm(os.listdir(os.path.join(self.source_path, folder, self.filtered_folder_name))): #Sample00021_color, Sample00022_color, etc.
                    if (file.startswith(str(""))): #well that's all of 'em
                        video = cv2.VideoCapture(os.path.join(self.source_path, folder, self.filtered_folder_name, file))
                        success, frame = video.read()
                        count = 0
                        while success:
                            #print(os.path.join(self.target_path, os.path.splitext(file)[0], str(count) + '.jpg'))
                            target_subfolder = os.path.join(self.target_path, folder, os.path.splitext(file)[0])
                            if not os.path.exists(target_subfolder):
                                os.makedirs(target_subfolder)
                            cv2.imwrite(os.path.join(target_subfolder, str(count) + '.jpg'), frame)
                            #print(cv2.imwrite(os.path.join(target_subfolder, str(count) + '.jpg'), frame))
                            success, frame = video.read()
                            #print('Read a new frame: ', success)
                            count += 1
    """
    
    def _create_sequence_frames(self):
        folders = [
            folder for folder in os.listdir(self.source_path)
            if os.path.isdir(os.path.join(self.source_path, folder))
            and os.path.join(self.source_path, folder) != self.target_path
        ]

        for folder in folders:
            target_folder = os.path.join(self.target_path, folder)
            os.makedirs(target_folder, exist_ok=True)

        num_cpus = os.cpu_count()

        for folder in folders:
            video_dir = os.path.join(self.source_path, folder, self.filtered_folder_name)
            
            files_and_folders = [
                [file, folder] for file in os.listdir(video_dir)
            ]

            with Pool(processes=num_cpus) as pool:
                pool.map(self._create_sequence_frame, files_and_folders)


    def _create_sequence_frame(self, file_and_folder):
            video_path = os.path.join(
                self.source_path,
                file_and_folder[1],          # folder
                self.filtered_folder_name,
                file_and_folder[0]           # filename
            )
            
            video = cv2.VideoCapture(video_path)
            success, frame = video.read()
            count = 0

            target_subfolder = os.path.join(
                self.target_path,
                file_and_folder[1],
                os.path.splitext(file_and_folder[0])[0]
            )
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)

            while success:
                cv2.imwrite(
                    os.path.join(target_subfolder, f"{count}.jpg"),
                    frame
                )
                success, frame = video.read()
                count += 1

            video.release() 
    
    

    def _create_montalbano_segments(self):
        df = pd.read_csv(f'{self.source_path}/montalbano_segments.csv', dtype=object, sep=",", header=None)
        for folder in os.listdir(self.target_path): #training, test, etc.
            if os.path.isdir(os.path.join(self.target_path, folder)):
                for subfolder in os.listdir(os.path.join(self.target_path, folder)): #Sample00021_color, Sample00022_color, etc.
                    if (subfolder.startswith(str(""))): #use for testing
                        print(subfolder)
                        sample_index = subfolder[6:-6]
                        segments = df[df[0]==sample_index]
                        print(len(segments))
                        if (len(segments) > 0):
                            target_path = os.path.join(self.target_path, folder, subfolder)
                            print("target_path ", target_path)
                            for index, row in segments.iterrows():
                                start_file, end_file = row[1].split()[0], row[1].split()[1]
                                os.rename(os.path.join(target_path, start_file+".jpg"), os.path.join(target_path, start_file+"-start.jpg"))
                                os.rename(os.path.join(target_path, end_file+".jpg"), os.path.join(target_path, end_file+"-end.jpg"))
                                print(row[0], row[1])