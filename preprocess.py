### first need to convert the input to <narrator><start-of-interview><interview summary><end-of-interview><narrator>
from helper import load_txt
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import pathlib
from pathlib import Path
@retry(wait= wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100)) # solve wait limit
def call_gpt(prompt, model_name = "gpt-3.5-turbo"):
  client = OpenAI(
      # This is the default and can be omitted
      api_key=YourID,
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model= model_name,
      seed = 42,
      max_tokens = 500,
      temperature=0.0, # probability distribution
  )

  # return the contents
  return chat_completion.choices[0].message.content

def preprocess_script(data_dir, video_name, narrator_id, output_dir):
    video_csv_path = data_dir / video_name[0] / video_name / "script_timeline.csv"
    data_frame = pd.read_csv(video_csv_path)
    final_output = []
    non_narrator_text_buffer = []
    last_non_narrator = None

    for index, row in data_frame.iterrows():
        current_speaker = row["Speaker"]
        text = row["Text"]

        if current_speaker == narrator_id:
            if non_narrator_text_buffer:
                final_output.append("<start-of-interview>")
                summarized_text = call_summarizer(" ".join(non_narrator_text_buffer))
                final_output.append(summarized_text)
                final_output.append("<end-of-interview>")
                non_narrator_text_buffer = []  # Clear the buffer
            final_output.append(text)  # Append narrator's text directly
        else:

            if current_speaker != last_non_narrator and non_narrator_text_buffer:
                # need to append special token to indicate the start of the interview
                final_output.append("<start-of-interview>")
                summarized_text = call_summarizer(" ".join(non_narrator_text_buffer)) # or other feature extraction method
                final_output.append(summarized_text)
                non_narrator_text_buffer = []  
                
                final_output.append("<end-of-interview>")
            non_narrator_text_buffer.append(text)
            last_non_narrator = current_speaker

    if non_narrator_text_buffer: # last set of interview segments
        final_output.append("<start-of-interview>")
        summarized_text = call_summarizer(" ".join(non_narrator_text_buffer))
        final_output.append(summarized_text)
        final_output.append("<end-of-interview>")
    # save the final_output to a file
    # each line is a sentence
    output_path = output_dir / video_name[0] / video_name / "input.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in final_output:
            f.write(line + "\n")

def call_summarizer(text):
    # call gpt
    prompt3 = f"summarize: {text}" # can play with prompt, RAG command for retrival
    summarized_text = call_gpt(prompt3, "gpt-4o")
    return summarized_text

def preprocess_script_scale(data_dir, video_name_path, narration_annotate_path, output_dir):
    video_name_list = load_txt(video_name_path)
    narration_annotate = pd.read_csv(narration_annotate_path)
    for video_name in video_name_list:
        print(video_name)
        try:
            narrator_id = narration_annotate[narration_annotate["video_name"] == video_name]["narrator_id"].values[0]
            preprocess_script(data_dir, video_name, narrator_id, output_dir)
        except:
            print(f"Error in processing {video_name}")
            continue

if __name__ == "__main__":
    data_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_train")
    video_name_path = Path("/home/weihanx/videogpt/whisperX/good_file.txt")
    narration_annotate_path = data_dir / f"narrator_id_by_length.csv"
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/lectureqa/train_data_main")
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_script_scale(data_dir, video_name_path, narration_annotate_path, output_dir)