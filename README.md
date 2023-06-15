# fine-tuning-sentence-transformers
A Very Simple Demo of Fine Tuning Sentence Transformers

This repository provides a practical demonstration of how to fine-tune a Sentence Transformer model on a custom dataset and then use the fine-tuned model to generate sentence embeddings. The scripts utilize the PyTorch library and Sentence Transformers for this purpose.

This project is a simple example of how to fine-tune a Sentence Transformer model. It is not designed for large-scale or real-world applications.

## Setup and Installation
To set up and run the example, follow these steps:

1. Clone this repository: `git clone https://github.com/adiekaye/fine-tuning-sentence-transformers.git`
2. Navigate to the project directory: cd sentence-transformer-tuning
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
- For Windows: `venv\Scripts\activate`
- For macOS/Linux: `source venv/bin/activate`
5. Install the required libraries: `pip install -r requirements.txt`
6. Run the script `01_tuning_your_model.py` to fine-tune the model: `python 01_tuning_your_model.py`
7. Run the script `02_using_your_model.py` to use the fine-tuned model: `python 02_using_your_model.py`
8. To exit the virtual environment, type `deactivate` in the terminal.

Note: The virtual environment and requirements installation steps are optional but recommended to ensure compatibility and avoid conflicts with other Python packages you may have installed.

## Contents
- `01_tuning_your_model.py`: Script that fine-tunes a Sentence Transformer model on a custom dataset.
- `02_using_your_model.py`: Script that uses the fine-tuned Sentence Transformer model to generate sentence embeddings and calculate their cosine similarity.
- `requirements.txt`: Lists the required libraries for this project.
- `README.md`: Provides instructions for setting up and running the example, and explains the contents of the repository.
- `.gitignore`: A simple Git configuration file to ignore the virtual environment directory and other non-essential files.
- `/tuned_models`: A directory to store your fine tuned models.
- `/tuned_models/.gitignore`: A gitignore file to make sure you don't accidentally commit your fine tuned model.

## Usage
The scripts `01_tuning_your_model.py` and `02_using_your_model.py` are executable as they are. However, you might need to adjust the path of the fine-tuned model in 02_using_your_model.py depending on your directory structure.

The `01_tuning_your_model.py` script will train a Sentence Transformer model using the specified training examples and then save the fine-tuned model.

The `02_using_your_model.py` script will load the original and the fine-tuned models to generate embeddings for specific sentences, and it will print the cosine similarity between the original and fine-tuned embeddings for each sentence.

## License
This project is licensed under the MIT License.
