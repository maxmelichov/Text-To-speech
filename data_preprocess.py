import HebrewToEnglish
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

#### THIS IS AN EXAMPLE OF HOW TO CONVERT THE ROBOSHAUL DATASET TO A LIST FOR THE MODEL ####

def process_dataset(csv_path, audio_path):
    """
    Process Hebrew text-to-speech dataset by converting Hebrew to English and creating train/val splits.
    
    Args:
        csv_path (str): Path to the CSV metadata file
        audio_path (str): Full path to the directory containing audio files
    """
    # Load the dataset
    df = pd.read_csv(csv_path, sep='|', encoding='utf-8')

    # Convert the 'text' column to a list
    text_list = df['transcript'].tolist()

    # Convert Hebrew text to English transliteration
    hebrew_to_english = HebrewToEnglish.HebrewToEnglish
    english_text_list = [hebrew_to_english(text) for text in tqdm(text_list, desc="Converting Hebrew to English")]

    list_of_path_and_text = []
    for path, text in tqdm(zip(df['file_id'], english_text_list), desc="Creating path-text pairs", total=len(df)):
        # Create full path to audio file
        full_audio_path = os.path.join(audio_path, f"{path}.wav")
        list_of_path_and_text.append((full_audio_path, text))

    # Split the data into train and validation sets
    train_texts, val_texts = train_test_split(english_text_list, test_size=0.2, random_state=42)
    
    # Create filelists directory if it doesn't exist
    os.makedirs('filelists', exist_ok=True)
    
    # Save the training set
    with open('filelists/train_list.txt', 'w', encoding='utf-8') as f:
        for path, text in tqdm(zip(df['file_id'][:len(train_texts)], train_texts), desc="Saving training set"):
            full_audio_path = os.path.join(audio_path, f"{path}.wav")
            f.write(f"{full_audio_path}|{text}\n")
    
    # Save the validation set
    with open('filelists/validation_list.txt', 'w', encoding='utf-8') as f:
        for path, text in tqdm(zip(df['file_id'][len(train_texts):], val_texts), desc="Saving validation set"):
            full_audio_path = os.path.join(audio_path, f"{path}.wav")
            f.write(f"{full_audio_path}|{text}\n")

def main():
    parser = argparse.ArgumentParser(description='Process Hebrew TTS dataset')
    parser.add_argument('--csv_path', type=str, default='data/saspeech_gold_standard/metadata_full.csv',
                       help='Path to the CSV metadata file')
    parser.add_argument('--audio_path', type=str, default='data/saspeech_gold_standard/wavs',
                       help='Full path to the directory containing audio files')
    
    args = parser.parse_args()
    
    process_dataset(args.csv_path, args.audio_path)

if __name__ == "__main__":
    main()