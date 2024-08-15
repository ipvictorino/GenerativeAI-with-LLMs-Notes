# Week 1

Owner: Inês
Status: Not started

[W1.pdf](Week%201/W1.pdf)

# Introduction to LLMs and the generative AI project lifecycle

Starting with how transformer networks actually work.

In 2017, the paper came out, “Attention is all You Need", and it laid out all of these fairly complex data processes which are going to happen inside the transformer architecture.

## Generative AI & LLMs

**Generative AI** 

- Machine that is capable of creating content that mimics or approximates human ability
- Subset of traditional machine learning
- Machine Learning models that underpin GenAI have learned these abilities by finding statistical patterns in massive datasets of content that has originally been created by humans

| Large Language Models (LLMs) |
| --- |
| Been trained of trillions of words over many weeks and months |
| Large amount of compute power |
| Billions of parameters |
| Interact with LLM using texts called prompts |
| Able to take natural language or human written instructions and perform tasks much as a human would |

Foundation models or Base models and their relative size in terms of parameters:

![Untitled](Week%201/Untitled.png)

### LLM syntax:

![Untitled](Week%201/Untitled%201.png)

💬 **Prompt**: the text you pass to an LLM is known as prompt

**🔍 Context Window**: The space or memory that is available to the prompt, which differs from model to model

**🎯** **Completion**: Output of the model - text contained in the original prompt, followed by the generated text

**💡 Inference**: Act of using the model to generate text

## LLM use cases and tasks

- Chat tasks (chatbots) which is based in base concept “Next word prediction”
- LLMs and generative AI are not limited to chatbots; they have a variety of applications.
- Tasks include:
    - Text generation (essays, summaries)
    - Translation (languages, natural language to code)
    - Code writing (e.g., Python for data analysis)
    - Information retrieval (e.g., named entity recognition, kind of word classification)
- Advanced use cases involve connecting LLMs to external data sources and APIs (RAG)

Larger models (hundreds of millions to billions of parameters) have better language understanding and task-solving abilities.

Smaller models can be fine-tuned for specific tasks.

The architecture of these models has driven their rapid growth in capabilities.

## Transformers

### Context

LLMs are not new. Previous generations were based on Recurrent Neural Networks (RNN)

- Limited by the amount of compute and memory needed to perform well at generative tasks
- Only uses previous words to generate next ones

The model needs more than the previous words to predict the next one. It also needs to understand the semantic meaning of the words. Understanding the language can be hard (the same word can have different meanings in the same languge).

In 2017, the paper “Attention is all you need” was published, proposing the architecture `Transformer`

- Scales efficierntly
- Parallel processing
- Attention to input meaning

### Transformers architecture

- Ability to learn the relevance and context of **all** the words in a sentence
- Applies attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input

![Untitled](Week%201/Untitled%202.png)

- Attention weights are learned through LLM training
- In the image below depicting an attention map, the word “book” is strongly connected with (paying attention) to words “teacher”, “taught”, “student” → **self-attention**

![Untitled](Week%201/Untitled%203.png)

- self-attention is a key attribute of the transformer architecture

High-Level Diagram

![Untitled](Week%201/Untitled%204.png)

- Transformer architecture is split into two distinct parts which work in conjunction with each other
    - Encoder: Encodes inputs (”prompts”) with contextual understanding and produces one vector per input token
    - Decoder: Accepts input tokens and generates new tokens
- LLMs are big statistical calculators and work with numbers, not words → That’s why you need to **tokenize** the words before passing text into the model

| Tokenizer |
| --- |
| Convert words into numbers with each number representing a position in the dictionary of all the possible words that the model can work with |
1. Tokens are passed to Embedding Layer (red)
2. Adds positional encoding to embeddings
3. Resultant vectors are passed through self-attention layer of embedder and decoder
4. After self-attention weights are applied to vectors, the output is processed through a fully-connected Feed-Forward Network
5. Outputs are logits which are passed through softmax layer where they are normalized into probability score for each word of the vocabulary

**Embedding Layer**

![Untitled](Week%201/Untitled%205.png)

- Trainable vector embedding space
- High-dimensional space where each token is represented as a vector (embedding) occupies a unique location within that space
- Each vector, representing a token-ID, and they learn to encode the meaning and context of individual tokens
- Example of encoders: word2vec

Tokenization Methods:

1. By token IDS matching to complete words

![Untitled](Week%201/Untitled%206.png)

1. Using token IDS for representing parts of words

![Untitled](Week%201/Untitled%207.png)

**Positional Encoding**

![Untitled](Week%201/Untitled%208.png)

Used to preserve the information about the word order, not losing relevance of the position of the word in the sentence

**Self-attention layer**

![Untitled](Week%201/Untitled%209.png)

- Analyses the relationships between the tokens in input sequences
- Allows model to attend / better capture the contextual dependencies between the words
- Self-attenction weights learned during training reflect the importance of each word in that input sequence to all other words in the sequence
- It does not happen just once - Multi-headed self-attention
    - Multiple heads are learned in parallel, independently from each other
    - Common numbers: 12-100
    - Each self-attention head will learn a different aspect of language (e.g. relationship between people entities, sequence activity, does it rhyme)
    
    | Head |
    | --- |
    | Single set of self-attention weights |

**Encoder**:

![Untitled](Week%201/Untitled.gif)

- Add the tokens to the input on the encoder side.
- Pass tokens through the embedding layer.
- Feed the embeddings into the multi-headed attention layers.
- Process the outputs of the attention layers through a feed-forward network to get the encoder's output.

The data leaving the encoder is a deep representation of the input sequence's structure and meaning.

**Decoder**:

![Untitled](Week%201/Untitled%201.gif)

- Insert this representation into the middle of the decoder to influence its self-attention mechanisms.
- Add a start-of-sequence token to the decoder's input.
- The decoder predicts the next token based on the encoder's contextual understanding.
- Pass the decoder's self-attention layer outputs through the feed-forward network and a final softmax output layer to get the first token.

**Iteration and completion**:

![Untitled](Week%201/Untitled%202.gif)

- Continue the loop by passing the output token back to the input to generate the next token.
- Repeat until an end-of-sequence token is predicted.
- Detokenize the final sequence of tokens into words to get the translated output.

Encoder-only Models:

- Sequence-to-sequence models
- Less-common these days
- Good for classification, sentiment analysis
- Input and Output need to be the same length
- Example LLMs: BERT

Decoder-only Models:

- Most commonly used
- Models can generalize to most tasks
- Example LLMs: GPT, BLOOM, Jurassic, Llama

Encoder-decoder Models:

- Perform well on sequence-to-sequence tasks such as translation. Also work well with general text generation tasks.
- Input sequence and the output sequence can be different lengths.
- Example LLMs: BART, T5

## Prompting and Prompt Engineering

Terminology Recap:

💬 **Prompt**: the text you pass to an LLM 

**🔍 Context Window**: The space or memory that is available to the prompt, which differs from model to model

**🎯** **Completion**: Output of the model - text contained in the original prompt, followed by the generated text

**💡 Inference**: Act of using the model to generate text

Not all the times you get the result you desire in the first try

| Prompt Engineering |
| --- |
| Develop and improve prompt so that the LLM produces better outcomes |

Strategies:

1. In-context learning (few-shot inference): Include examples if the task that you want the model to carry out inside the prompt 
    
    ![Untitled](Week%201/Untitled%2010.png)
    
    ![Untitled](Week%201/Untitled%2011.png)
    
2. Fine-tune model: Performs additional training on the model using new data to make it more capable of a task you wanted to perform

## Generative Configuration

Inference parameters

- Influences model outputs
- Different than the training parameters
- Invoked at inference time and give contral on things like on how creative the output is
- Parameters:
    - Max new tokens: Limit the number of tokens that the model will generate
    - Top-p: Sampling technique that limits the random sampling. Selects an outout using the random-weighted strategy with top-ranked consecutive results by probability with a cumulative probability ≤ p.
    - Top-k: Sampling technique that limits the random sampling. Select the top-k results after applying random-weighted strategy using the probabilities
        
        ![Untitled](Week%201/Untitled%2012.png)
        
        | Greedy decoding |
        | --- |
        | Word/Token with the highest probability is always selected |
        
        Sometimes it is good to use the greedy aproach but other controls like random sampling can create more interesting results.
        
        | Random Sampling |
        | --- |
        | Select a token using a random-weighted startegy across the probabilities of all tokens |
    - Temperature: Controls the randomness of the output (softmax). Influences the shape of the probability distribution that the model calculates for the next token. The higher the temperature, the higher the randomness

## Generative AI project lifecycle

https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/

![Untitled](Week%201/Untitled%2013.png)

Define use case:

- Essay writing
- Summarizarion
- Translation
- Information Retrieval
- Invoke APIs and actions

Choose an existing model or pretrain your own:

- Typically you start with a base model

**Cheat Sheet - Time and effort in the lifecycle**

![image.png](Week%201/image.png)

* Some of these buzz words will be explained in the upcoming sections



# LLM pre-training and scaling laws

Continuing discussion on [Generative AI project lifecycle](Week%201.md), it is important to have the following considerations while choosing the model:

- Work with existing foundation model: Pretrained LLM
- Train own model from scratch: Custom LLM

In general, you'll begin the process of developing your application using an existing foundation model.

The developers of some of the major frameworks for building generative AI applications like Hugging Face and PyTorch, have curated hubs where you can browse these models.

- **Model cards**: describe important details such as best use cases, how it was trained, known limitations
    
    ![Untitled](Week%201/Untitled%2014.png)
    

## Pre-training LLMs

- Initial **training** process for LLMs: self-supervised
- During this phase, the LLMs develop a deep statistical understanding of language
- Model learns from vast amount of unstructured textual data (GB - PB) - internalizes the patterns and structures present in the language
- Patterns then enable the model to complete its training objective
- Model weights get updated to minimize the loss of the training objective
- Encoder then generates embedding for each tokens
- There is also a filter applied (data quality asessment) when tipically only 1-3% of original tokens are used

![Untitled](Week%201/Untitled%2015.png)

Each transformer model (encoder-only, decoder-only, encoder-decoder) is trained on a different objective:

- **Encoder-Only (auto-encoding)**
    - Pre-trained using Masked Language Modeling (”denoising” objective)
        
        
        | Masked Language Modeling (MLM) |
        | --- |
        | Predict mask tokens in order to reconstruct original sentence
        
        e.g. The | teacher | <MASK> | the | student |
    - Build bidirectional representations of the input sequence: model has understanding of the full context of a token and not just of the words that come before
    - Use Cases: Sentiment Analyiss, Named entity recognition, Word classification
    - Models: BERT and ROBERTA
- **Decoder-only (autoregressive)**
    - Pre-trained using Causal Language Model
        
        
        | Causal Language Modeling (CLM) |
        | --- |
        | The training objective is to predict the next token based on the previous sequence of tokens
        Predict next token = “full language model”
        
        e.g. The | ? |  ⇒ The | teacher | ? ⇒ The | teacher | teaches |
    - Iterates over the input sequence 1-by-1 to predict the following token
    - Context is unidirectional context
        - By learning to predict the next token from a vast number of examples, the model builds up a deep statistical representation of the language
        - Use Cases: Text generation, larger decoder-only models show strong zero-shot inference abilities and can often perform general tasks well
        - Models: GPT and BLOOM
- **Encoder-decoder (sequence-to-sequence)**
    - Objective function varie from model to model
    - T5 pre-trains the encoder using Span Correction
        
        
        | Span Corruption | Sentinel Tokens |
        | --- | --- |
        | Masks random sequences of input tokens, which are then replaced with unique sentinel tokens. The objective is to reconstruct the masked token sequences autoregressively.  | Special tokens added to the vocabulary that do not correspond to any actual from input text |
    - Output is the sentinel token followed by predicted tokens
        
        ![Untitled](Week%201/Untitled%2016.png)
        
    - Use-cases: Translation, summarization, question answering
    - Models: T5, BART

Summary

![Untitled](Week%201/Untitled%2017.png)

While choosing which model to work with, besides considering the typical use cases for each LLM architecture, it's important to note that larger models are typically more capable of performing their tasks well, regardless of the architecture. For instance, if you want to work on a classification model, even though BERT as an encoder is appropriate for the use case, it would be good to consider GPT-4 due to its size and strong performance on a variety of tasks.

Training enormous models is difficult and very expensive, so much so that it may be infeasible to continuously train larger and larger models. 

## **Computational Challenges while training**

- CUDA out of memory.
    - Storing 1B parameters requires = $10^9 \times 4 \text{ bytes/parameter} = 4GB$ @32-bit full precision of GPU RAM
    - Additionally, it requires ADAM optimizers states, Gradients, Activations, and temporary memory, making a total o 24GB @32-bit full precision

**Solution: Quantization**

- Statistically projects the original FP32 into lower precision space using scaling factors
- FP16
    - Reduce memory to store by reducing precision from 32-bit floating points (FP) to 16-bit floating point (FP)
    - Statistically projects the original FP32 into lower precision space using scaling factors
    - Example:
        
        ![Untitled](Week%201/Untitled%2018.png)
        
    - Using quantization to project a FP32 into FP16 reduces the memory requirement by half
- BFLOAT16
    - Short for Brain Floating Point format
    - Developed by Google Brain
    - Hybrid between half precision FP16 and full precision FP32.
    - “Truncated” 32-bit float -has the same bits as FP32 for exponent but truncates the fraction to 7 bits instead of 23 bits
    - Example:
        
        ![Untitled](Week%201/Untitled%2019.png)
        
    - Not well suited for integer calculations
    - Most popular choice
- INT8
    - Example:
        
        ![Untitled](Week%201/Untitled%2020.png)
        
    - Pretty dramatic loss of precision
- Impact
By applying quantization, you can reduce your memory consumption required to store the model parameters down to only two gigabyte using 16-bit half precision of 50% saving and you could further reduce the memory footprint by another 50% by representing the model parameters as eight bit integers, which requires only one gigabyte of GPU RAM.
    
    ![Untitled](Week%201/Untitled%2021.png)
    


> ⚠️ This example was for 1B param models. For larger models, it is impossible to train on a single GPU, requiring distributed computing techniques.



### [Optional] Efficient Multi-GPU Compute Strategies

PyTorch’s Distributed Data Parallel (DDP)

- Distributes large datasets across multiple GPUs and process these batches of data in parallel.
- Then a sychronization step combines the results of each GPU which in turn updates the model on each GPU which is always identical across chips
    
    ![Untitled](Week%201/Untitled%2022.png)
    
- Requires model weights and other parameters (gradients, optimizer states) fit into a single GPU
- The first stage requires a full model copy on each GPU - redundant memory consumption

Model Sharding - PyTorch’s Fully Sharded Data Parallel (FDPD)

- Motivated by the [“Zero” paper](https://arxiv.org/abs/1910.02054) - zero data overlap between GPUs
- Eliminates the redundancy present in DDP by sharding the model parameters, gradients and optimizer states across GPUs instead of replicating them
- Distributes large datasets across multiple GPUs, also sharding model parameters, gradients, and optimizer states
- Requires data collection from all GPU’s before any forward or backward pass. Each GPU requests data from the other GPUs on-demand to materialize the sharded data into unsharded data for the duration of the operation.
- Then a sychronization step combines the results of each GPU and each model is updated
- Offers three optimization stages:
    - Stage 1 $P_{os}$ - Shards only “Optimizer States” across GPUs (reduces memory footprint ~4)
    - Stage 2 $P_{os+g}$ - Also shards the “Gradients” across GPUs (reduces memory footprint ~4)
    - Stage 3 $P_{os+g+p}$ -Shards all components including the model parameters across GPUs (memory reduction is linear across GPUs)
        
        ![Untitled](Week%201/Untitled%2023.png)
        
- Allows use cases where model weights and other parameters (gradients, optimizer states) don’t fit into a single GPU
- Supports offloading to CPU if needed
- Configure level of sharding via `sharding factor` where max number equals the total number of GPUs (”Full Sharding”). Any lower number makes the process “Hybrid Sharding”
    
    ![Untitled](Week%201/Untitled%2024.png)
    
- Performance
    
    ![Untitled](Week%201/Untitled%2025.png)
    

## Scaling laws and compute-optimal models

Goal: maximize model performance (minimizing loss while predicting tokens)

There are various choices to take into consideration:

- Increase dataset size
- Increase number of parameters
- Compute budget (GPUs, training time)

Important measure to take into consideration

| Petaflop/s-day | Measure of the Floating Point (FP) operations performed at a rate of 1 petaFLOP ( = 1 quadrillion FP operations per second ) for one day

8 NVIDIA V100s GPUs or 2 NVIDIA A100s chips |
| --- | --- |

![Untitled](Week%201/Untitled%2026.png)

Researchers have explored trade-offs between training dataset size, model size and compute budget:

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

For Compute Budget, it was found a Power-law relationship between Test Loss and Compute (in Petaflop/s-day). The higher the compute the better the test loss

![Untitled](Week%201/Untitled%2027.png)

For Dataset size and Model Size also show a Power-law relationship (while freezing the other two parameters, respectively)

![Untitled](Week%201/e12bf5df-b354-4fae-86e9-c9ee45d8a14c.png)

In terms of balance in terms of these three quantities, Jordan H. Sebastian B. proposed how to find the optimal figures ⇒ “Chinchilla paper”

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

- Many large models may be over-parameterized and under-trained
- Smaller models trained on more data could perform as well as large models
    
    > “Optimal training dataset size for a given model is about 20x larger than the number of parameters”
    > 
- Compute-optimal: For 70B parameter model the ideal training dataset contains ~1.4T tokens

## Pretraining for domain adaptation

Domain adaptation is needed to achieve good performance in cases where target domain uses uncommon vocabulary or language structures

BloombergGPT

[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)

- Example of a LLM that has been trained for specific finance domain (51% financial data and 49% public data)
- Started with Chincilla scaling laws for guidance (area is pink is the recommendation)
    
    ![Untitled](Week%201/Untitled%2028.png)
    

# Reading

[Reading Resources](Week%201/Reading%20Resources.md)