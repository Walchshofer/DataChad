# -*- coding: utf-8 -*-
"""YT Transformers Agent like LangChain + Tools.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HGpp1OI-o_ppHi2bHZsvV6QX9k5gsTIK
"""

!pip -q install transformers openai accelerate diffusers
!pip -q install datasets sentencepiece
!pip -q install huggingface_hub>=0.14.1

!pip show transformers

import os
import openai

os.environ["OPENAI_API_KEY"] = ""

from huggingface_hub import notebook_login
notebook_login()

import IPython
import soundfile as sf

def play_audio(audio):
    sf.write("speech_converted.wav", audio.numpy(), samplerate=16000)
    return IPython.display.Audio("speech_converted.wav")

#ini the agent
from transformers.tools import HfAgent
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
print("StarCoder is initialized 💪")

from transformers.tools import HfAgent
agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
print("OpenAssistant is initialized 💪")

"""## Testing with OpenAI"""

from transformers.tools import OpenAiAgent
agent = OpenAiAgent(model="text-davinci-003")

print("OpenAI is initialized 💪")

cat = agent.run("Generate an image of a Mainecoon grey cat sitting down resting")
cat

audio = agent.run("Read out loud the summary of techcrunch.com", return_code=True)
play_audio(audio)

"""### Chat mode

The difference between the two is relative to their memory:
- `.run` does not keep memory across runs, but performs better for multiple operations at once (such as running two, or three tools in a row from a given instruction)
- `.chat` keeps memory across runs, but performs better at single instructions.

"""

agent.chat("Show me an an image of a ginger mainecoon cat")

agent.chat("Transform the image so that the background is in the snow")



agent.prepare_for_new_chat()

"""## Tools

So far we've been using the tools that the agent has access to. These tools are the following:

- **Document question answering**: given a document (such as a PDF) in image format, answer a question on this document (Donut)
- **Text question answering**: given a long text and a question, answer the question in the text (Flan-T5)
- **Unconditional image captioning**: Caption the image! (BLIP)
- **Image question answering**: given an image, answer a question on this image (VILT)
- **Image segmentation**: given an image and a prompt, output the segmentation mask of that prompt (CLIPSeg)
- **Speech to text**: given an audio recording of a person talking, transcribe the speech into text (Whisper)
- **Text to speech**: convert text to speech (SpeechT5)
- **Zero-shot text classification**: given a text and a list of labels, identify to which label the text corresponds the most (BART)
- **Text summarization**: summarize a long text in one or a few sentences (BART)
- **Translation**: translate the text into a given language (NLLB)

We also support the following community-based tools:

- **Text downloader**: to download a text from a web URL
- **Text to image**: generate an image according to a prompt, leveraging stable diffusion
- **Image transformation**: transforms an image

We can therefore use a mix and match of different tools by explaining in natural language what we would like to do.


"""

agent.chat("who is the possible new twitter ceo based on this article at https://techcrunch.com/2023/05/11/elon-musk-says-he-has-found-a-new-ceo-for-twitter/")

agent.chat("who was the former twitter ceo based on that article")

agent.chat("who is the current twitter ceo based on that article")

agent.chat("Translate the title of that article into French")

agent.chat("Summarize that article for me")

agent.prepare_for_new_chat()

"""From HF Demo 


### Adding new tools 

We'll add a very simple tool so that the demo remains simple: we'll use the awesome cataas (Cat-As-A-Service) API to get random cats on each run.

We can get a random cat with the following code:
"""

import requests
from PIL import Image

image = Image.open(requests.get('https://cataas.com/cat', stream=True).raw)
image

"""Let's create a tool that can be used by our system!

All tools depend on the superclass Tool that holds the main attributes necessary. We'll create a class that inherits from it:
"""

from transformers import Tool

class CatImageFetcher(Tool):
    pass

"""This class has a few needs:

- An attribute name, which corresponds to the name of the tool itself. To be in tune with other tools which have a performative name, we'll name it text-download-counter.
- An attribute description, which will be used to populate the prompt of the agent.
- inputs and outputs attributes. Defining this will help the python interpreter make educated choices about types, and will allow for a gradio-demo to be spawned when we push our tool to the Hub. They're both a list of expected values, which can be text, image, or audio.
- A __call__ method which contains the inference code. This is the code we've played with above!

Here’s what our class looks like now:
"""

from transformers import Tool
from huggingface_hub import list_models


class CatImageFetcher(Tool):
    name = "cat_fetcher"
    description = ("This is a tool that fetches an actual image of a cat online. It takes no input, and returns the image of a cat.")

    inputs = []
    outputs = ["text"]

    def __call__(self):
        return Image.open(requests.get('https://cataas.com/cat', stream=True).raw).resize((256, 256))

"""We can simply use and test the tool directly:"""

tool = CatImageFetcher()
tool()

"""In order to pass the tool to the agent, we recommend instantiating the agent with the tools directly:"""

from transformers.tools import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

"""Let's try to have the agent use it with other tools!"""

agent.run("Fetch an image of a cat online and caption it for me")

