# medicalBot
It is an generative ai project which will answer the questions related to decieses.

# End-to-end-Medical-Chatbot-Generative-AI


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository(3.10.18 in our case)

```bash
conda create -n medibot python=3.10 -y (3.10.18)
```

```bash
conda activate medibot
```
### before installing packages u need to write setup.py(Install this project itself as a package.")

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### for embeddings or else its not working due to sentense-transformers should be installed

``bash
pip install -U langchain langchain-community sentence-transformers huggingface-hub
pip install -U langchain-huggingface

```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
