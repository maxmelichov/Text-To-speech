# Text-To-Speech (Robo-Shaul)

Welcome to the Robo-Shaul repository! This project enables you to train your own Robo-Shaul or use pre-trained models to convert Hebrew text into speech using the Tacotron 2 TTS framework.

Robo-Shaul was originally developed for a competition, where the winning model was trained for only 5k steps. After the competition, a more advanced model was trained for 90k steps using improved methodologies and a wider range of training data, resulting in significantly better performance.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/maxmelichov/Text-To-speech.git
    cd Text-To-speech
    ```

2. **Set up a virtual environment:**
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    activate.bat  # Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Clone required submodules and dependencies:**
    ```bash
    git clone https://github.com/maxmelichov/tacotron2.git
    git submodule init
    git submodule update
    git clone https://github.com/maxmelichov/waveglow.git
    cp waveglow/glow.py ./
    ```

---

## 📁 Project Structure

The main directories used in this project are:

```
Text-To-speech/
├── data/                  # Place the SASPEECH dataset here
├── checkpoints/           # Stores Tacotron2 model checkpoints (*.pt files)
├── waveglow_weights/      # Stores WaveGlow model checkpoint (*.pt file)
├── tacotron2/             # Tacotron2 source code (cloned as submodule)
├── waveglow/              # WaveGlow source code (cloned as submodule)
├── ...
```

- **data/**: Put your downloaded and preprocessed dataset here.
- **checkpoints/**: Save and load Tacotron2 model weights (e.g., `checkpoint_90000.pt`).
- **waveglow_weights/**: Place the WaveGlow model checkpoint file (e.g., `waveglow_256channels.pt`).

---

## 📦 Download Pre-trained Models

- **Weights for RoboShaul:** [Download](https://huggingface.co/notmax123/robo_shaul)
Waveglow is a must and the 5K steps or 90K steps is for you to choose

---

## 📚 Dataset

- Download the SASPEECH dataset from [OpenSLR](https://openslr.org/134).

---

## 🛠️ Usage

1. **Preprocess the data:**
    ```bash
    python data_preprocess.py
    ```
    After running the script, ensure you generate a `.txt` file in the same format as the examples in the `filelists` directory:

    ```
    path/to/audio.wav|transcript in Hebrew that using English letters
    ```

2. **Train the model:**
    ```bash
    python train.py
    ```

3. **Generate speech (inference):**
    ```bash
    python inference.py
    ```

---

## 💡 Demos & Resources

- **Live Demo:** [Project Site](http://www.roboshaul.com/)
- **Demo Page:** [here](https://maxmelichov.github.io/)
- **Quick Start Notebook:** [Notebook](https://github.com/maxmelichov/Text-To-speech/blob/main/Tacotron_Synthesis_Notebook_contest_notebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1heUHKqCUwXGX_NRZUeN5J9UdB9UVV32m#scrollTo=IbrwoO0A1D0b)
- **Project Podcast:** [חיות כיס episode](https://open.spotify.com/episode/7eM8KcpUGMxOk6X5WQYdh5?si=3xf0TNzwRTSHaCo8jIozOg)
- **Training & Synthesis Videos:** [Part 1](https://www.youtube.com/watch?v=b1fzyM0VhhI) | [Part 2](https://www.youtube.com/watch?v=gVqSEIr2PD4&t=284s)

---

## 📝 Model Details

- The system uses the SASPEECH dataset, a collection of unedited recordings from Shaul Amsterdamski for the 'Hayot Kis' podcast.
- The TTS system is based on Nvidia's Tacotron 2, customized for Hebrew.

**Note:** The model expects diacritized Hebrew (עברית מנוקדת). For diacritization, we recommend [Nakdimon](https://nakdimon.org) ([GitHub](https://github.com/elazarg/nakdimon)).


---

## 👥 Contact

| Maxim Melichov | Tony Hasson |
| -------------- | ----------- |
| [LinkedIn](https://www.linkedin.com/in/max-melichov/) | [LinkedIn](https://www.linkedin.com/in/tony-hasson-a14402205/) |

---

Feel free to reach out with questions or suggestions!
