# KL divergence

Owner: Inês
Status: Not started

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/sMBhUqOTS7a1y8Ho5vF9qQ_4ea0db1de2764787b09f681255045df1_image.png?expiry=1723507200000&hmac=NuT8jqKrVBHoz1NZgHmy0ELjdBOAXomBkHGkT5w8Opk](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/sMBhUqOTS7a1y8Ho5vF9qQ_4ea0db1de2764787b09f681255045df1_image.png?expiry=1723507200000&hmac=NuT8jqKrVBHoz1NZgHmy0ELjdBOAXomBkHGkT5w8Opk)

KL-Divergence, or Kullback-Leibler Divergence, is a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (PPO) algorithm. It is a mathematical measure of the difference between two probability distributions, which helps us understand how one distribution differs from another. In the context of PPO, KL-Divergence plays a crucial role in guiding the optimization process to ensure that the updated policy does not deviate too much from the original policy.

In PPO, the goal is to find an improved policy for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment. However, updating the policy too aggressively can lead to unstable learning or drastic policy changes. To address this, PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence.

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (Transformer Reinforcement Learning). In [this link](https://huggingface.co/blog/trl-peft) you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption). The image shows an overview of the PPO training setup in TRL