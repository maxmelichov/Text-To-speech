# Text-To-speech
Welcome to the Robo-Shaul repository! Here, you'll find everything you need to train your own Robo-Shaul or use pre-trained models. Robo-Shaul is a text-to-speech system that converts Hebrew text into speech using Tacotron 2 TTS as a framework

Although the model that won the competition had a training duration of only 5k steps.

A subsequent model was developed after the competition deadline. This advanced model underwent an extensive training process of 90k steps, utilizing an enhanced training methodology that incorporated a broader spectrum of extreme cases. These novel training techniques were absent in the previous model, providing the later model with a significant advantage in terms of its capabilities and performance.

#### For a demo look [here](https://maxmelichov.github.io/)

For a quick start look at [Notebook](https://github.com/maxmelichov/Text-To-speech/blob/main/Tacotron_Synthesis_Notebook_contest_notebook.ipynb) or Open In Colab <a target="_blank" href="https://colab.research.google.com/drive/1heUHKqCUwXGX_NRZUeN5J9UdB9UVV32m#scrollTo=IbrwoO0A1D0b"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### For the חיות כיס podcast documenting the project listen [here](https://open.spotify.com/episode/7eM8KcpUGMxOk6X5WQYdh5?si=3xf0TNzwRTSHaCo8jIozOg)
#### Site for the project [link](http://www.roboshaul.com/)

The system consists of the SASPEECH dataset, which is a collection of recordings of Shaul Amsterdamski's unedited recordings for the podcast 'Hayot Kis', and a Text-to-Speech system trained on the dataset, implemented in the Tacotron 2 by Nvidia AI TTS framework.

To download the dataset for training, go to [link](https://openslr.org/134)

To download the trained models, go to [model with 90K steps](https://drive.google.com/uc?id=13B_NfAw8y-A9pg-xLcP5kQ_7dbObGc8S&export=download), [model with 5K steps](https://drive.google.com/u/0/uc?id=1iE3VgeQsyZcIgAXYmwhk-FzWktwrT2Wo&export=download)

The model expects diacritized Hebrew (עברית מנוקדת), we recommend [Nakdimon](https://nakdimon.org) by Elazar Gershuni and Yuval Pinter. The link is to a free online tool, the code and model are also available on GitHub at [https://github.com/elazarg/nakdimon](https://github.com/elazarg/nakdimon)

## Data Creation 
For a quick start look at [Notebook](https://github.com/maxmelichov/Text-To-speech/blob/main/DataCreation.ipynb)https://github.com/maxmelichov/Text-To-speech/blob/main/DataCreation.ipynb)

## How to use the training notebook and the synthesis notebook
These videos will help you to gather the data and also train the model: [Part1](https://www.youtube.com/watch?v=b1fzyM0VhhI),[Part2](https://www.youtube.com/watch?v=gVqSEIr2PD4&t=284s) 

We're using the custom Tacotron 2 that we took from Nvidia and custom notebooks.


## Information about HebrewToEnglish.py
We implemented several functions that deal with processing and converting Hebrew text into English sounds. It includes functions for breaking down numbers into Hebrew words, converting Hebrew letters into their corresponding English sounds, and converting entire Hebrew sentences into English sounds. The code also includes functions for handling numbers, punctuation marks, and special cases within the Hebrew text.



## What can be done to make this model even more robust:
1. Use the Hebrew package to create a set of all the possible Hebrew letters with Nikod in UNICODE-8.
2. Change Tacotron's 2 input letters to the set that you created in Step 1.
3. Create a new transcript algorithm that can convert Hebrew with Nikod to UNICODE-8.

### Contact Us

We are Maxim Melichov and Tony Hasson. If you have any questions or comments, please feel free to contact us using the information below.

| **Maxim Melichov**          | **Tony Hasson**         |
| ------------------------- | ------------------------- |
| <a href="https://www.linkedin.com/in/max-melichov/" target="_blank">Connect on LinkedIn</a> | <a href="https://www.linkedin.com/in/tony-hasson-a14402205/" target="_blank">Connect on LinkedIn</a> |
