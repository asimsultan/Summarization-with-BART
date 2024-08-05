
# Summarization with BART

Welcome to the Summarization with BART project! This project focuses on summarizing long texts using the BART model.

## Introduction

Summarization involves condensing long texts into shorter versions while retaining the main points. In this project, we leverage the power of BART to generate summaries using a dataset of long texts and their corresponding summaries.

## Dataset

For this project, we will use a custom dataset of long texts and their summaries. You can create your own dataset and place it in the `data/summarization_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/bart_summarization.git
cd bart_summarization

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes long texts and their summaries. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: input_text and summary.

# To fine-tune the BART model for summarization, run the following command:
python scripts/train.py --data_path data/summarization_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/summarization_data.csv
