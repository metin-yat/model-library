

# Project Description

Model Library is a my own library where I manually find the needed dataset, selecting the model and implementing these models to a interface where I am using Streamlit for that.

List of models that is in this repository:
- Chat with Fine-tuned Llama3 8b Instruct 4bit: There is a example for fine-tuning llama model in `llm` folder. But I could not turn the the final output to `gguf` file, that means I could not load it to Ollama. I used `unsloth` to fine tune the model and used Google Colab notebooks that has been released by Unsloth.
- Plate Detection Fine-tuned YOLOv9 with Turkish Licence Plate: I finetune YOLOv9 with ![Turkish Licence Plate Dataset](https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset?resource=download) released on Kaggle. 

#### What I learned
- I improved about doing Streamlit interfaces which I never did in any public repository. Also learned how to use different models on Streamlit interface
- Working with LLM Models since I never did before. I learned how to fine-tune LLM Models and how to use it in my local
- Learned how to work with YOLO models, how to set a folder structere and how to train.
- Working with Ollama: I worked with Ollama before but never used my own model. After fine-tuning my own LLM model, I learned how to load it to Ollama, how to use it on my code and also sharing that model with users.


#### Example from models in the library:
I will add examples from each model after push. You can see an example of the YOLOv9 model on `computer_vision\cv1\run`. I will uplaod the pictures of the interface for Model Library.

# Run in your local


To use this model, first thing you should do is install Ollama to use your fine-tuned model. After adding your fine-tuned model to your local Ollama OR downloading (with Ollama) the models that has been uploaded on Ollama's model library. (https://ollama.com/library)


```bash
git clone https://github.com/metin-yat/model-library.git
```

### Getting list of Ollama models that is in your local
To get a list for that in your terminal, execute:
```bash
  ollama list
```

I added my model as 'finetunedllama38binstruct4bit:latest'. This name is my model's name which I'll using that name on `chatbot.py`. You should change that before next step.

### Last step before running.

After changing the name of the model and doing the necessary installation in your python environment, last thing you should change is the root_dir in the yolo_detection.py. root_dir is where your interface.py is.

To run the interface on your local:

```bash
cd model-library
streamlit run interface.py
```