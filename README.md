# 🔉 Marine Acoustic Signal Classification 
(toujours en cours)


![Python](https://img.shields.io/badge/Python-100%25-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Marine Acoustic Signal Classification is a machine learning project aimed at analyzing and classifying marine sounds such as whale calls, ship noises, and background oceanic sounds. The project leverages python libraries in audio processing and deep learning to provide accurate classification results.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sound plays a vital role in the marine ecosystem. This project focuses on the classification of marine acoustic signals to aid in marine life research, ship noise pollution studies, and other oceanographic applications. By using features extracted from audio signals, this project trains a neural network to classify sounds into predefined categories.

## Features

- **Audio Format Conversion**: Converts `.mp3` files to `.wav` format for standardized processing.
- **Feature Extraction**: Extracts MFCC (Mel-frequency cepstral coefficients) features from audio files.
- **Deep Learning Model**: Implements a neural network for multiclass classification of audio signals.
- **Pretrained Model Support**: Easily load and use existing models for predictions.

## Dataset

The dataset is structured as follows:

- **Input Base Directory**: `marine_sounds`
- **Categories**: 
  - Whale
  - Ship
  - Background

After preprocessing, the audio files are stored in the `converted_wavs` directory. Each category contains `.wav` files for training and testing.

## Usage

Follow these steps to get started:

1. Clone the repository:
   ```bash
   git clone https://github.com/KushagraSaxena77/Marine-Acoustic-Signal-Classification.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place your `.mp3` files in the `marine_sounds/<category>` directory.

4. Run the main script:
   ```bash
   python main.py
   ```

5. The script will:
   - Convert `.mp3` files to `.wav`
   - Extract features
   - Train the model (or load an existing one)
   - Save the trained model to `marine_model.keras`

6. View classification results and metrics.

## Technologies Used

- **Python Libraries**:
  - `librosa`: For audio processing and feature extraction.
  - `pydub`: For audio file conversion.
  - `matplotlib`: For visualization.
  - `TensorFlow/Keras`: For building and training the neural network.

## Contributing

We welcome contributions to enhance the project! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

For further inquiries or collaborations, please contact [Kushagra Saxena](https://github.com/KushagraSaxena77).
