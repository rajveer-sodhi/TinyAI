# TinyAI

Title: TinyAI
Team Members: Rajveer Sodhi, Aalyaan Ali, Tito Fleming, Benjamin Peckham

## **Introduction**

Our team aims to recreate and enhance the outcome of the paper “Less is More: Recursive Reasoning with Tiny Networks” (Jolicoeur-Martineau, 2025). The paper introduced Tiny Recursive Models (TRMs) as a way to create tiny models with reasoning abilities above their weight class. The model referenced in the paper, with only 7M parameters, works on solving spatial grid-based problems such as Sudoku, and achieves comparable or higher accuracy at the task than most SOTA large language models. It achieves this by repeatedly refining its output over several reasoning cycles instead of providing an answer outright.

Contemporary transformer-based LLMs improve their accuracy by scaling the number of parameters and using a single pass of the transformer block. This architecture limits the reasoning ability that small models can achieve, while making training a financially and computationally expensive task.

Our project is inspired by the fundamental idea of TRMs - recursive refinement of an output can drastically improve the final answer of a small model. To this end, we aim to apply the principles and innovations introduced by Jolicoeur-Martineau to a transformer model. We want to test if a pipeline integrating recursion can provide better performance on simple mathematical reasoning problems - an area which, as OpenAI claims, even several top-tier LLMs can struggle with.

However, TRMs are designed using CNNs and act on spatial tasks, not sequential. Language tasks, on the other hand, require an ordered token-based approach. Hence, we will have to significantly rework the architecture provided in the original paper to adapt it for an attention-based transformer. Our goal is to give recursive reasoning abilities to a 10-20M parameter transformer model trained on grade-school math problem datasets. We expect this hybrid pipeline to perform more reliably and accurately than a single-pass transformer of the same size.

## **Related Work** 

Please read and briefly summarize (no more than one paragraph) at least one paper/article/blog relevant to your topic beyond the paper you are re-implementing/novel idea you are researching.

A closely related work is Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., NeurIPS 2023). The authors implement a test-time framework where the LLM acts as a “generator, refiner, and feedback provider” on its own output. This is an attempt to improve on the often-poor initial outputs given by an LLM without additional training. This feedback-for-refining loop was evaluated on a variety of tasks, like mathematical reasoning or dialog response generation. They found consistent improvements in performance of ~20% on task performance. At a high level, Self-Refine relates to our work as it shows the iterative process of improvement on an answer conditioned on previous outputs can significantly improve performance. This idea is not very far from our intended work with Tiny Recursive Models. 

## **Data**

Due to the tiny size of our proposed transformer model (approximately 10-20 million parameters) and time constraints, we will be training the model exclusively on popular grade-school-level math natural language question-answer datasets. There will (likely) be no pre-training done for the model to learn basic grammatical rules or extended vocabulary, making the model limited but finetuned.
Datasets that fit our requirements include OpenAI’s GSM8K and Microsoft’s Ocra-Math-Word-Problems-200K. Since the Ocra word problems set includes over 200,000 synthetic word problems created based on math datasets, including GSM8K, we can use only the Ocra dataset. However, several of the problems in this dataset are still likely to be linguistically too advanced for our tiny model to process, and we would hence have to process the dataset to extract the shortest and simplest problems in the set that our model can support. We expect the size of this refined dataset to be around 10-20% of the larger set, or 20-40K data points.
This filtration process can be largely deterministic, based on parameters tuned by us. Additionally, we can use cues from the GPT-based data point generation approach outlined in “Orca-Math: Unlocking the potential of SLMs in Grade School Math“ (Arindam Mitra et al., 2024) to edit data points or create more relevant synthetic data. Finally, we can also apply common textual data augmentation techniques on the refined dataset of elementary word problems to increase the training data for our model.

## **Methodology**

Our model architecture adapts the Tiny Recursive Model (TRM) from the original paper to work with text-based mathematical reasoning rather than spatial grid-based puzzles. Our core architecture can be thought of in 3 main components. An input embedding layer, a 2-layer transformer encoder with self-attention, and our output head. Diverging from the original TRM which uses CNNs, we replace the convolutional blocks with transformer layers to handle sequential token based inputs.

The model maintains two state variables across recursive iterations:

y (the embedded current solution) and z (a latent reasoning feature).

 During each recursion step, the model first updates z by passing it through the transformer along with the embedded question x and current answer y for n=6 iterations. Then it refines y using the updated z. This process repeats for T=3 deep recursion cycles. The first T-1 cycles run without gradient computation to iteratively improve the solution, while the final cycle backpropagates through the full recursion to update weights. 

We’re training using deep supervision with up to N_sup=16 supervision steps per training example. At each step the model will attempt to improve its answer and then we compute cross entropy loss between the predicted answer and ground truth (known answer). 
We then incorporate a adaptive computational time (ACT) mechanism to determine when to stop refining and move to the next training example. The model will be trained using AdamW optimizer with a small learning rate (1e-4), batch size of 32-64 (depending on what our computers can handle), hidden dimension of 256-512, and heavy data augmentation (such as paraphrasing) on our filtered Ocra dataset.

The hardest parts of implementation will actually just be managing the gradient flow through multiple recursive steps while also maintaining separate gradient free and gradient enabled passes at the same time. We also expect implementing the state passing mechanism between supervision steps with proper detachment from the computational graph, and adapting the ACT halting mechanism to work effectively on text-based problems rather than grid-based puzzles to be challenging. 

Our backup ideas if we encounter issues include reducing model complexity by decreasing recursion depth. And if memory becomes problematic we may replace self-attention with simpler feed forward layers if training is shown unstable. Using a pre trained embedding layer (like Word2Vec) is a consideration if learning embeddings from scratch fails with limited data. 


## **Metrics**

We are primarily focused on raw accuracy since our task is question-answering with deterministic correct answers. We will measure the exact percentage of problems where the model's output exactly matches the ground truth answer. This is appropriate because math problems have objectively correct numerical or short textual answers with no ambiguity.

We plan to run three core experiments. We will be training our recursive transformer with the full TRM-inspired architecture. Then we will train a baseline single-pass transformer with the same parameter count, so that we can use it for a direct comparison. Lastly, we will run ablation studies varying recursion depth (n and T values) to better understand the relationship between recursion cycles and accuracy. Each model will be trained on our filtered Ocra dataset subset and evaluated on a held-out test set.

The original paper measured success using test accuracy on puzzle tasks. They found that TRM with 7M parameters achieved 87% accuracy on Sudoku-Extreme, 85% on Maze-Hard, and 45% on ARC-AGI-1, substantially outperforming both baseline models and much larger LLMs. Their results also examined accuracy as a function of recursion depth, showing clear improvements with deeper recursion up to an optimal point. We will follow a similar evaluation approach, reporting test accuracy and analyzing performance across different recursion configurations.

To assess our model's performance, we will just compare test accuracy between our recursive model and the single-pass baseline. We will also track training curves to ensure both models converge properly and analyze where errors occur. We will also examine how answer quality improves across recursion steps within our model by logging intermediate predictions. Our base goal is to successfully implement the recursive architecture and achieve any measurable improvement over the single-pass baseline (even a small ~5-10% relative improvement would be enough to validate the approach). Our target goal is to achieve 1.1-1.25x relative improvement compared to the single-pass baseline. This should demonstrate that recursion provides substantial enough reasoning benefits for small models. Our stretch goal is to achieve greater than 1.3x relative improvement and successfully demonstrate that performance scales predictably with recursion depth, ideally exceeding that of larger models on the same dataset subset.

## **Ethics** 

Dependency on LLMs in the modern day is skyrocketing. For many, it is largely replacing the standard web search and revolutionizing how work is done in nearly every field. This comes with enormous computational costs that are already having serious environmental consequences.

If successful, our unique architecture will act as a proof of concept that even SLMs can perform substantial reasoning tasks without exponentially increasing the compute. At scale, this could have seriously positive implications for the environment and energy demand. If we do not see an improvement or comparable performance to the status quo, we will have at least exhausted another potential candidate.

Major stakeholders in this problem would be major AI and tech companies, as well as the average end user. Major consequences from a model with mistakes would be misinformation and disinformation being spread to the end user and potentially being perceived as true. Although it is important to note that this is already a severe issue with modern LLMs, and our model is categorized as an ‘SLM’ with a limited scope of mathematical reasoning.

As for our dataset, we will use a filtered and augmented subset of Ocra-Math-Word-Problems-200K, which could include human biases in simple ways, such as the names of characters and scenarios being more “western,” but our model concerns itself more with its mathematical reasoning ability and thus won’t compound any biases in the data itself.

## **Division of Labor**

Aalyaan: Data preprocessing and data pipeline construction 
Ben: Transformer model construction 
Rajveer: TRM implementation (Recursion added to transformer)
Richard: Training and testing script, metrics, and analysis reports



