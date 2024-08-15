# Proximal Policy Optimization (PPO)

Owner: Inês
Status: Not started

| Proximal Policy Optimization (PPO) |
| --- |
| It optimizes a policy to be more aligned with human preferences |
| Over many iterations, PPO makes updates to the model. The updates are small and within a bounded region, resulting in an updated model that is close to the previous version, |
| The changes are small to result in a more stable learning |

Popularity due to good balance of complexity and performance

In the context of LLMs, after defining instruct LLM, each cycle of PPO goes over two phases

# Phase 1: Create completions

LLM is used to carry out a number of **experiments**, completing the given prompts. 

Experiments are used to assess the outcome of the current model (e.g. helpfulness, harmeless level, …) so that it can be used against the reward model in phase 2.

| Value Function $V_\theta(s)$  |
| --- |
| Separate head of the LLM that is used to estimate the expected reward for a given state $s$ |
| Based on the current sequence of tokens |
| Baseline to evaluate the quality of completions against alignment criteria |

![image.png](Proximal%20Policy%20Optimization%20(PPO)/image.png)

## Value Loss

Makes estimates for future rewards more accurate

Minimize the value loss that is the difference between the actual future total reward and its approximation to the value function

# Phase 2: Model Update

The model weights updates are guided by the prompt completion, losses, and rewards. 

PPO also ensures to keep the model updates within a certain small region called the trust region. → Proximal aspect

## Policy Loss

$$
L^{POLICY}=\min \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t |s_t)}\cdot \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t |s_t)},1-\epsilon, 1+\epsilon\right)\cdot \hat{A}_t \right)
$$

where,

$\pi_\theta$ is the model’s probability of the next token $a_t$ given the current prompt $s_t$ with the updated LLM

$\pi_{\theta_{old}}$ probability of the next token $a_t$ given the current prompt $s_t$, with the initial, frozen version of the LM.

$\hat{A}_t$ is the estimated advantage term of a given action - how much better or worse the current action is, compared to all possible actions at the same state

**First Term**: Represents the ratio of new policy to old policy, scaled by the advantage term.

**Second Term**: Clipped version of the ratio, ensuring it stays within $([1 - \epsilon, 1 + \epsilon])$.

$L^{POLICY}$ aims to maximize the advantage term while ensuring the new policy remains close to the old policy. 

In general, maximizing $\hat{A}_t$  leads to higher rewards.

- Positive values means that the suggested token is better than average
    
    ![image.png](Proximal%20Policy%20Optimization%20(PPO)/45aabd6e-4e93-45d2-86c3-f9a6d861be61.png)
    

Directly maximizing the policy loss can lead to unreliable outcomes if the old and new policies diverge too much. 

That’s why the second term of the $\min$ function comes into play - acts as a guardrail, ensuring that we do not leave the trust region

## Entropy loss

$$
L^{ENT} = \text{entropy}(\pi_theta(\cdot \mid s_t))
$$

- While policy loss moves the model towards alignment, entropy allows the model to maintain creativity
- Entropy $\neq$ Temperature → Temperature influences the model creativity at inference time and Entropy influences the model creative during training

## Objective Function

$$
L^{PPO} = L^{POLICY}+c_1L^{VF}+c_2L^{ENT}
$$

where

$c_1$ and $c_2$ are hyperparameters 

- PPO objective updates the model weights through backpropagation over several steps
- Once the model weights are updated, the PPO starts a new cycle
- For the next iteration the LLM is replaced with the updated LLM and a new PPO cycle starts
- After many iterations you arrive at the human-aligned LLM

# Other techniques used for fine-tuning the LLMs through human or AI feedback

Active area of research, with a lot of emerging developments

- RL technique: Q-Learning
- Alternative to RL: Direct preference optimization [Stanford research]