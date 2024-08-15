# Week 2

Owner: Inês
Status: Not started

This week focuses on two main areas: instruction tuning and parameter efficient fine-tuning (PEFT) of large language models (LLMs).

**Key Topics**

1. **Instruction Fine-Tuning**:
    - Enhances the model's ability to respond to prompts and follow instructions.
    - Addresses the issue of "catastrophic forgetting" by fine-tuning across various instruction types.
2. **Parameter Efficient Fine-Tuning (PEFT)**:
    - Methods like LoRA enable fine-tuning with smaller memory footprints and reduced compute requirements.
    - These techniques are cost-effective and suitable for developers with budget constraints.
    - PEFT allows for tuning without adjusting every parameter, by freezing original model weights or adding adaptive layers.
    - Prompting is often a starting point for developers, but they may require fine-tuning techniques like LoRA for enhanced performance.

[W2.pdf](Week%202/W2.pdf)

# Fine-tuning LLMs with instructions

## Instruction fine-tuning

Prompting has some drawbacks:

- In-context learning may not work for smaller models
- To improve the performance using few-shot prompting the examples may take up space in the context window

When prompting is not enough, it makes sense to consider **fine-tuning** the model.

- Supervised learning process (vs pre-trained LLM self-supervised)
- Dataset of labeled examples are used to update the weights of the LLM (prompt-completion pairs)
- Extends the training of the model to improve performance
- An example strategy is called **Instruction-fine tuning**

| Instruction-fine tuning |
| --- |
| Trains the model using examples that demonstrate how it should respond to a specific instruction. 
Dataset includes many pairs of prompt-completion examples for the task, each of which include an instruction (e.g. Summarize the following text:)  |

![Untitled](Week%202/Untitled.png)

- **Full fine-tuning:** When all the model weights are updated
- Process:
    1. Prepare training data
    Prompt template libraries have been assembled so that they can be used for instruction fine-tuning
    https://github.com/bigscience-workshop/promptsource/blob/main/promptsource/templates/amazon_polarity/templates.yaml
    2. Divide dataset into Training, Validation, and Test splots
    3. Fine-tuning using training and validation dataset
    For several batches and epochs:
        1. Select prompts from training dataset and pass them through LLM 
        2. Completion is generated and compared with actual label
        3. Uses Cross-Entropy function to calculate loss between two token distributions
        4. Loss updates model weights in standard back-propagation
    4. Final performance evaluation using test accuracy

### Single-task fine-tuning

Applied when there is only one task that is of interest.

- Often 500-1000 examples can result in good performance
- The process may lead to **catastrophic-forgetting,** degrading performance on other tasks. You can avoide by:
    - Evaluate cost-benefits and if it makes sense to continue
    - Fine-tuning on multiple tasks at the same time
    - Use regularization techniques to limit the amount of change that can be made to the weights of the model during training.
    - Consider **Parameter Efficient Fine-Tuning**
        
        
        | Parameter Efficient Fine-Tuning (PEFT) |
        | --- |
        | Set of techniques that preserves the weights of original LLM and trains only a small number of task-specific adaptive layers and parameters |

### Multi-task fine-tuning

Extension of single-task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks (e.g. summarize, rate this review, translate inton Python, …)

- Mixed dataset improves the model’s performance on all the tasks simultaneously
- Resolves catastrophic-forgetting
- Requires a lot of data: 50-100000 examples
- Models: FLAN models (FLAN-T5 is the FLAN instruct version of the T5 foundation model)
    
    [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
    
    | Fine-tuned Language Net (FLAN) models |
    | --- |
    | Refer to a specific set of instructions used to perform instruction fine-tuning |
    - One of the datasets used for fine-tuning is SAMsum: a dialogue dataset. One key topic is that this dataset contains several different instructions for the same end goal: summarization, helping the model generalize and perform better
    
    ![Untitled](Week%202/Untitled%201.png)
    

## Model evaluation metrics

For classic ML models, it is easy to find deterministic ways of assessing the correctness of an output

$$
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}} 
$$

For large language model applications, it is different. For different tasks you need an automated, structured way to make measurements, such as ROUGE and BLEU

| ROUGE - Recall-Oriented Understudy for Gisting Evaluation | BLEU - Bilingual Evaluation Understudy |
| --- | --- |
| - Used for text summarization
- Compares to human generated summarizations | - Used for text translations
- Compares to human generated translations |

For overall evaluation of model’s performance, consider evaluation benchmarks that have been developed by researchers.

Terminology

- **Unigram:** single word
- **Bigram:** Two-words
- N-gram: Group of n-words

### ROUGE score

**Translation Example - unigrams**

<aside>
📖 Reference (human)
It is cold outside.

Generated output:
It is very cold outside

</aside>

$$
\text{ROUGE-1 Recall} = \frac{\text{unigram matches}}{\text{unigrams in reference}} = \frac{4}{4}=1.0
$$

$$
\text{ROUGE-1 Precision} = \frac{\text{unigram matches}}{\text{unigrams in output}} = \frac{4}{5}=0.8
$$

$$
\text{ROUGE-1 F1} = 2\frac{\text{precision}\times\text{recall}}{\text{precision + recall}} = 2\frac{0.8}{1.8}=0.89
$$

**Translation Example - bigrams**

<aside>
📖 Reference (human)
It is cold outside.
It is, is cold, cold outside

Generated output:
It is very cold outside
It is, is very, very cold, cold outside

</aside>

$$
\text{ROUGE-2 Recall} = \frac{\text{bigram matches}}{\text{bigrams in reference}} = \frac{2}{3}=0.67
$$

$$
\text{ROUGE-2 Precision} = \frac{\text{bigram matches}}{\text{bigrams in output}} = \frac{2}{4}=0.5
$$

$$
\text{ROUGE-2 F1} = 2\frac{\text{precision}\times\text{recall}}{\text{precision + recall}} = 2\frac{0.335}{1.17}=0.57
$$

To determine the optimal amount of N-grams you can take a look at the longest common subsequence (LCS) from the sequence, which in the example $\text{LCS(Gen, Ref)}=2$ (it is, cold outside) ⇒ ROUGE-L

$$
\text{ROUGE-L Recall} = \frac{\text{LCS(Gen, Ref)}}{\text{unigrams in reference}} = \frac{2}{4}=0.5
$$

$$
\text{ROUGE-L Precision} = \frac{\text{LCS(Gen, Ref)}}{\text{unigrams in output}} = \frac{2}{5}=0.4
$$

$$
\text{ROUGE-L F1} = 2\frac{\text{precision}\times\text{recall}}{\text{precision + recall}} = 2\frac{0.2}{0.9}=0.44
$$

Another improvement is to clip the n-matches so that the results are not incorrectly high

**Translation Example - bigrams**

<aside>
📖 Reference (human)
It is cold outside.

Generated output:
cold cold cold cold

</aside>

<aside>
👎 Incorrect

$$
\text{ROUGE-1 Precision} = \frac{\text{unigram matches}}{\text{bigrams in output}} = \frac{4}{4}=1
$$

</aside>

<aside>
👍 Correct

$$
\text{Modified Precision} = \frac{\text{clip(unigram matches)}}{\text{bigrams in output}} = \frac{1}{4}=0.25
$$

</aside>

### BLEU score

- Calculated using the **average** precision over multiple n-gram sizes

$$
\text{BLEU} = avg(\text{precision across range of n-grams})
$$

- Compute using libraries (e.g. hugging face)

<aside>
📖 Reference (human)
I am very happy to say that I am drinking a warm cup of tea.

Generated output:
I am very happy that I am drinking a cup of tea. - BLEU 0.495
I am very happy that I am drinking a warm cup of tea. - BLEU 0.730
I am very happy to say that I’m drinking a warm tea. - BLEU 0.798

</aside>

### Evaluation benchmarks

To holistically evaluate LLMs, use pre-existing datasets and benchmarks that isolate specific skills and focus on potential risks. 

Ensure the evaluation data is unseen by the model during training for accurate assessment.

Evaluation benchmarks:

- GLUE
- SuperGLUE
- HELM
- MMLU
- BIG-bench

**GLUE - General Language Understanding Evaluation**

- Introduced in 2018
- Collection of NLP tasks such as sentiment analysis or question answering
- Motivation was to apply to models that can generalize across multiple tasks

**SuperGLUE**

- Successor to GLUE, addressing some limitations
- Series of tasks, some of which not included in GLUE or more advanced such as multi-sentence reasoning and reading comprehension
- Both benchmarks have leaderboards that can be used to evaluate and compare models

Benchmarks for massive models:

- **Massive Multitask Language Understanding (MMLU)**
    - Designed specifically for modern LLM
    - To perform well, models must prossess extensive world knowledge and problem-solving abilities
    - Models tested on elementery mathematics, law, history, computer science
- **BIG-bench**
    - 204 tasks ranging from linguistincs, childhood development, math, common sense reasoning, biology, physics, and more…
    - Come in three different sizes to keep costs achievable as running this large benchmarks can incur in large inference costs
- **Holistic Evaluation of Language Models (HELM)**
    - Aims to improve transparency of LLM and provide guidance on which models provide well for specific tasks
    - Multi-metric approach: 7 metrics across 16 core scenarios
    Metrics: Accuracy, Calibration, Robustness, Fairness, Bias, Toxicity, Efficiency
        
        ![Untitled](Week%202/Untitled%202.png)
        
    

# Parameter efficient fine-tuning (PEFT)

As we saw, training LLMs is computational intensive

As solution to the problems that appear with full fine-tuning you can use instead PEFT:

- Only update a small subset of parameters (15%-20% max), drastically reducing memory allocated to fine-tuning
- PEFT techniques:
    - Most of the parameter weights are frozen and the focus is on fine-tuning a subset of existing model parameters (e.g particular layers or components)
    - Original weights are left untouched and techniques focuses on adding a small number of new parameters or layers and fine-tune only the new components
- Less prone to catastrophic forgetting
- The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.

## PEFT Trade-Offs

- Memory Efficiency
- Parameter Efficiency
- Training Speed
- Inference Costs
- Model Performance

## PEFT Methods

**Selective**

Select subset of initial LLM parameters to fine-tune

- Identify which parameters to update:
    - Train only certain components of the model
    - Train only individual parameter types

**Reparameterization**

Reparameterize model weights using a low-rank representation of the original network weights

- Commonly used technique is LoRA -
Low-rank adaptation of Large Language Models

**Additive**

Original parameters frozen. Adds new trainable layers or parameters to model
**Adapters**
**Soft Prompts**

Adapters

- Add trainable layers to encoder or decoder, after attention or feedforward layers

Soft Prompts:

- Keeps model architecture frozen and manipulates input
    - Adding trainable parameters to prompt embeddings
    - Keeping input fixed and retraining embedding weights
- Technique: **Prompt Tuning**

### Low-rank adaptation of Large Language Models (LoRA)

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

- PEFT that falls into reparameterization category
- Updates weights of the model without having to train every single parameter again
- Usually applied to the Transformer section after embeddings are obtained and are to be applied to self-attention NN, which computes attention scores by appying scores
- Procedure

![Untitled](Week%202/Untitled%203.png)

1. Freezes most of the original LLM weights $W$
2. Injects two A, B, rank decomposition matrices alongside the original weights $A\cdot B = C => size(C) = size(\text{wgt2modify})$
    1. Matrix multiply low rank matrices $B\cdot A$
    2. Add to original weights $W +B\cdot A$
3. Trains the weights of the smaller matrices
- Applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains
- Can also be applied to the feedforward NN
- Possible to train different rank decomposition matrices for different tasks. The memory required to store these LoRA matrices is very small. So in principle, you can use LoRA to train for many tasks. Switch out the weights when you need to use them, and avoid having to store multiple full-size versions of the LLM.
- Combining LoRA with quantization techniques is calles q-LoRA

**Example**:

- Weight has dimensions 512 x 64 = 32 768 parameters
- Applying LoRA rank r = 8 =
    - A has dimension 8*64 = 512 parameters
    - B has dimension 512*8 = 4 096 parameters
- Training parameter reduction of 86%

**Evaluation**

1. Define base model: FLAN-T5
2. Set baseline score
    
    ```python
    flan_t5_base = {
    	'rouge1': 0.2334,
    	'rouge2': 0.0760,
    	'rougeL': 0.2014,
    	'rougeLsum': 0.2015
    }
    ```
    
3. Compute scores for FULL fine-tuned model
    
    ```python
    flan_t5_base_instruct_full = {
    	'rouge1': 0.4216,
    	'rouge2': 0.1814,
    	'rougeL': 0.3384,
    	'rougeLsum': 0.3384
    }
    ```
    
4. Compute scores for LoRA fine tune ROUGE
    
    ```python
    flan_t5_base_instruct_lora = {
    	'rouge1': 0.4081,
    	'rouge2': 0.1633,
    	'rougeL': 0.3251,
    	'rougeLsum': 0.3249
    }
    ```
    

**Choosing the LoRA rank**

- In the paper that first proposed LoRA, researchers at Microsoft explored how different choices of rank impacted the model performance on language generation tasks.
- Authors found that using larger LoRA matrices (>16) provides plateau in loss value, i.e. such ranks didn’t so much improved performance
- Tipically it is recommended ranges between 4-32

![Source: Hu et al. 2021, “LoRA: Low-Rank Adaptation of Large Language Models”](Week%202/Untitled%204.png)

Source: Hu et al. 2021, “LoRA: Low-Rank Adaptation of Large Language Models”

### Prompt Tuning

- Do not change the model weights at all
- Prompt tuning is **not** prompt engineering!
- Prompt tuning adds trainable “soft prompt” to inputs and leave it up to the supervised learning process to determine optimal values
- **Soft prompt:** set of trainable tokens
    - Prepended to embedded vectors that represent input text
    - Same length as embedding vectors of the language tokens
    - Amount in the range of 20-100 tokens
    
    ![Untitled](Week%202/Untitled%205.png)
    
    - NL tokens are hard because they have fixed spots in the embedding vector space. But soft prompts are different; they aren't fixed words. Instead, they are like virtual tokens that can have any value in the continuous embedding space. Through supervised learning, the model finds the best values for these virtual tokens to perform well on a task.
- The weights of the LLM are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt.
    
    ![Untitled](Week%202/Untitled%206.png)
    
- Possible to train a different set of soft prompts for each task and then easily swap them out at inference time

**Evaluation**

- According to study done by Lester et al. 2021, prompt tuning can be as effective as fill fin-tuning for larger models (model parameters = $10^{10}$)
    
    ![Source: Lester et al. 202, “The Power of Scale for Parameter-Efficient Prompt Tuning”](Week%202/Untitled%207.png)
    
    Source: Lester et al. 202, “The Power of Scale for Parameter-Efficient Prompt Tuning”
    

**Issues**

- Interpretability of soft prompts: Trained soft-prompt embedding does not correspond to a known token (<unk>) but nearest neighbor tokens shows that they form tight semantic clusters (completely, totally, entirely) suggesting the prompts are learning word-like representatio ns

# Reading

[Reading Resources](Week%202/Reading%20Resources.md)