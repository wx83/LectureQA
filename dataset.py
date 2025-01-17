import torch
from torch.utils.data import Dataset, DataLoader
from helper import load_txt, save_txt
# need custom tokenizer to incorpoate new token <start-of-interview> and <end-of-interview>

class CustomDataset(Dataset):
    def __init__(self, input_dir, video_name_path, output_dir, tokenizer):
        # all video name
        self.video_name_list = load_txt(video_name_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.video_name_list) # number of video_name in this dataset

    def __getitem__(self, index):
        video_name = self.video_name_list[index]
        # load txt file
        input_txt_path = self.output_dir / video_name[0] / video_name / "input.txt"
        # load txt
        text = load_txt(input_txt_path)
        # tokenize
        encoding = self.tokenize(text)
        return encoding # get the encoding for that video name list
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    def collate_fn(self, batch):
        return self.tokenizer.pad(batch, return_tensors="pt") # return the padded batch, use the intro part first, padd until the same length of input id


# input is the GPT summary and output is the abstractive summary in the teaser part
class SummaryDataset(Dataset):
    def _init_(self, input_dir, video_name_path, output_dir, tokenizer):
        pass

    def _len_(self):
        pass

    def _getitem_(self, index):
        pass

