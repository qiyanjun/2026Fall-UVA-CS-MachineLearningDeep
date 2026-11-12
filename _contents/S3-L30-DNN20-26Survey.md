---
LOrder: 390
layout: post
title: Quick survey of recent deep learning (2020-2026)
lecture: S3-L30-DNN20-26Survey
lectureVersion: current
extraContent: S3-L0-25recentLLM-extra
video: <b>M1</b> <a href="https://youtu.be/tjIJVfwSrgs" target="_blank">Original</a> · <a href="https://youtu.be/VlRvDEgGlio" target="_blank">AI-Clone</a> · <a href="https://youtu.be/ERmB14NBze4" target="_blank">Author-Voice</a>  |  <b>M2</b> <a href="https://youtu.be/Z2Pq6QokCco" target="_blank">Original</a> · <a href="https://youtu.be/RKjZQ0sVkH8" target="_blank">AI-Clone</a> · <a href="https://youtu.be/uej_07mCf3U" target="_blank">Author-Voice</a>  |  <b>M3</b> <a href="https://youtu.be/l58pIDazkQU" target="_blank">Original</a> · <a href="https://youtu.be/akDRFnuW8I8" target="_blank">AI-Clone</a> · <a href="https://youtu.be/q36MqvNWXq8" target="_blank">Author-Voice</a>  |  <b>M4</b> <a href="https://youtu.be/gjXxOVly83s" target="_blank">Original</a> · <a href="https://youtu.be/Mo0qGBHNO3k" target="_blank">AI-Clone</a> · <a href="https://youtu.be/O4BzakYswuA" target="_blank">Author-Voice</a>
notes: Goodfellow, Bengio & Courville (2016), <a href="https://www.deeplearningbook.org/" target="_blank">Deep Learning</a> · LeCun, Bengio & Hinton (2015), <a href="https://www.nature.com/articles/nature14539" target="_blank">Deep Learning (Nature review)</a>
morenotes: <a href="https://github.com/afshinea/stanford-cs-230-deep-learning"> DNN Cheatsheets </a> + <a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-L0-25recentLLM-extra.pdf"> [Recent LLM-survey] </a> 
extra: true
categories: survey
tags:
- 3Classification
- Nonlinear
- Deep
- Discriminative
- 4Unsupervised
- Generative
- RL
---

### Extra reading:
<a href="https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/Lectures/S3-L0-25recentLLM-extra.pdf"> [Recent LLM-survey] </a>

### Summary:

**Deep Learning, 2020 to 2026: What Changed at Each Stage.** This lecture walks through the "deep learning in a nutshell" pipeline and asks one question at every stage: *what changed from 2020 to 2026, and what lasted?* Each story runs from a 2020 idea, to what replaced it, to where it stands in 2026. Topics covered in earlier lectures (CNNs, RNNs, Transformers, diffusion, RL, LoRA, agents, serving) are pointers, not repeats.

- Module I: Data and Representation
  + I.1 Data: from graphs, trees, and sets as a new trend to graphs and 3D geometry in major science (2024 Nobel Prizes, AlphaFold)
    * Message passing, the core GNN idea, with a worked example; symmetry (invariance and equivariance) as a design rule for 3D data
    * AlphaFold 2 → 3: a generic diffusion module on raw atom coordinates replaced hand-built geometry
    * AI weather models (GenCast, Aurora) and the operational ECMWF AIFS
  + I.2 Representation: from "which architecture wins?" to mixed families
    * CNN vs. ViT stopped being a contest (DINOv3 distills one ViT teacher into ViT and ConvNeXt students); SAM 3 segments by text prompt
    * Mixture-of-experts: memory follows total parameters, compute follows active parameters
    * Challengers: Mamba (selective state spaces), diffusion LLMs, KANs, and why the Transformer lasted
- Module II: Tasks
  + II.1 Generation: from one model per task (super-resolution, label-to-image, faces) to one general model with a text condition
    * Video with sound (Veo 3, Sora 2); 3D with diffusion plus Gaussian splatting
    * Deepfakes then and now; marking generated content (SynthID watermarks, C2PA signed metadata); laws in force (US TAKE IT DOWN Act, EU AI Act Article 50)
  + II.2 Decisions: from AlphaGo / AlphaZero to learning a model and planning inside it
    * DreamerV3 world models, Genie 3 interactive generated worlds, robot learning and the sim-to-real gap
  + II.3 Reasoning: from Neural Turing Machines to formal proofs with RL
    * AlphaProof (RL writing Lean proofs), IMO 2025 gold-medal level in formal and natural language
    * Few-shot learning moved from meta-learning (MAML) to in-context prompting of LLMs
- Module III: Training, Losses, and Search
  + III.1 New losses: from adversarial (GAN) and reconstruction losses to denoising, flow matching, one-step generators, and joint-embedding prediction
    * R3GAN ("the GAN is dead; long live the GAN!"); flow matching as regression on a velocity target; rectified flow, consistency models, and distillation
    * Self-supervised learning: layer-wise pretraining (history), SimCLR, MAE, DINO, and JEPA / V-JEPA 2
  + III.2 Search: from neural architecture search over a defined space to scale-driven progress and agents that edit code (Darwin Gödel Machine, The AI Scientist and its limits)
- Module IV: Deployment, Trust, and Hardware
  + IV.1 Trust and measurement: from saliency, adversarial examples, and small-network verification (Reluplex) to circuit tracing inside LLMs, verification at scale (α,β-CROWN), and benchmarks that saturate in months
    * The jagged frontier, METR time horizons, and ARC-AGI-3
  + IV.2 Hardware: from GPUs, TPUs, and compression for the edge (pruning, EfficientNet) to compute as a research constraint, inference-specific chips, supply concentration, and heavily quantized large models on device
- What lasted from 2020: inductive bias, pretraining without labels then fine-tuning, residuals / attention / gradient descent, regression on a good target, checking results with a verifier or held-out set, and the pipeline itself

# Disclaimer: it is quite hard to make important topics of deep learning fit on a one-session schedule. We aim to make the content reasonably digestible in an introductory manner. We try to focus on a modularity view by introducing important variables to digest deep learning into chunks regarding data/ model architecture /tasks / training workflows and model characteristics. We think this teaching style provides students with context concerning those choices and helps them build a much deeper understanding.


# Study Guide: Deep Learning, 2020 to 2026

## The whole map: 2020 vs. 2026

| Nutshell stage | 2020 | 2026 |
|----------------|------|------|
| **Data** | Grids, sequences, some graphs | Molecules, weather, video, 3D |
| **Representation** | CNNs for grids, Transformers for language | Hybrids, sparse experts |
| **Task** | Classify, generate faces, play games | Video with sound, world models, proofs |
| **Score function (loss)** | Adversarial and reconstruction | Flow matching, joint embedding |
| **Search / optimization** | Search a defined space (NAS) | Agents edit code; scale drives progress |
| **Metrics / trust** | Accuracy, saliency, robustness | Circuits, verification, time horizons |
| **Hardware** | GPUs, TPUs, edge pruning | Inference chips, heavy quantization |

---

## Quiz: Short-Answer Questions

**Instructions:** Provide a concise answer (2-3 sentences) for each of the following questions.

1. In a message-passing GNN step, if you swap the labels of two neighbors C and D, what happens to the new values of their neighbors A and B, and why?
2. A mixture-of-experts model has 1.6 T total parameters and 49 B active per token. What share runs per token, and what does MoE save and not save?
3. Why did the "KANs beat MLPs" result change once the comparison was made fair?
4. What changed from AlphaFold 2 to AlphaFold 3?
5. A detector finds no watermark in a viral video. Can you conclude that the video is real? Why?
6. What did games offer RL that the real world does not? Why does the Lean proof language offer the same thing?
7. How is the flow matching loss like linear regression?
8. How do MAE, DINO, and JEPA each train without labels?
9. How does circuit tracing go further than a saliency map?
10. A 1 B parameter model at 16 bits takes 2 GB. How many parameters fit in 2 GB at 4 bits?
11. An agent scores about 66% on OSWorld but under 1% on ARC-AGI-3 (March 2026). How can both be true?

---

## Answer Key

1. **Nothing changes.** Each node aggregates its neighbors' messages with a sum or mean, which does not depend on the order or the names of the neighbors, and the same weights are used at every node; so relabeling nodes only relabels the outputs (permutation invariance).

2. **49 B / 1.6 T ≈ 3%, so only about 3% of the parameters run for each token.** MoE cuts compute per token, giving a large store of knowledge at a smaller compute cost, but all parameters must still be held in memory.

3. **When KANs and MLPs were matched on parameters and FLOPs, MLPs generally did better, except on symbolic formulas.** The lesson is a reading habit: when a paper says "better", ask what was held equal.

4. **AlphaFold 2 used a hand-built structure module (frames and torsion angles); AlphaFold 3 replaced it with a diffusion module that denoises raw atom coordinates.** The overall plan (a pairwise trunk, then a module that makes 3D structure) stayed the same, and the scope grew to proteins, DNA, RNA, ligands, and ions.

5. **No.** A mark detector only finds content that was marked when it was made; a generator that adds no mark, or a screenshot or re-save that strips signed metadata, leaves no mark. "No mark found" does not mean "real".

6. **Games give perfect rules and unlimited practice, so an agent can try millions of times and always gets a correct reward; the real world gives neither.** In Lean, a proof either checks or fails, so the checker gives a perfect, automatic reward, just as a game does.

7. **The network is given a point on a path from noise to data and predicts the velocity along that path, and the loss is the mean squared error between the predicted and target velocity.** It is the same recipe as linear regression: a target, a prediction, and MSE; diffusion is one special path, and straighter paths need fewer sampling steps.

8. **MAE hides about 75% of image patches and rebuilds the pixels; DINO matches a student's output to a momentum teacher's on different views; JEPA predicts the hidden content as an embedding, with no pixel decoder.** Each turns unlabeled data into a supervised problem.

9. **A saliency map only shows which input pixels or tokens affected an output.** Circuit tracing maps the internal features and paths from input to output and tests them by intervention: suppressing or injecting a feature (e.g., swapping "Texas" for "California" turns "Austin" into "Sacramento").

10. **4 bits is a quarter of 16 bits, so the same 2 GB holds about 4 B parameters.** Heavily quantized large models often beat smaller high-precision ones on device (down to about 3.5 effective bits per weight).

11. **Capability is not one number: the frontier is jagged.** OSWorld tests familiar computer-use tasks that models have been trained toward, while ARC-AGI-3 tests acquiring a novel skill in an interactive environment with no instructions, where humans solve 100% and frontier AI is below 1%.

---

## Essay Questions

**Instructions:** Prepare a detailed, essay-format response for each of the following prompts. (Answers not provided).

1. **Pick two stages of the nutshell pipeline and, for each, trace the 2020 idea, what replaced it, and where it stands in 2026.** Explain what lasted and why.

2. **Compare the losses used to train generative and self-supervised models (adversarial, denoising, flow matching, contrastive, masked reconstruction, joint-embedding prediction).** What does each ask the network to do, and why has "regression on a good target" been so successful?

3. **Discuss how we measure and trust deep learning systems in 2026.** Cover mechanistic interpretability, formal verification, saturating benchmarks, METR time horizons, ARC-AGI-3, and marking generated content.

---

## Glossary of Key Terms

| Term | Definition |
|------|------------|
| **Message Passing** | The core GNN step: each neighbor sends a message, the messages are aggregated (sum or mean), and the node's vector is updated. |
| **Invariance / Equivariance** | An invariant output is unchanged by a transformation (a molecule's energy under rotation); an equivariant output changes in step with it (its forces). |
| **Geometric Deep Learning** | Choosing the data's symmetry and designing layers that build it in. |
| **Mixture-of-Experts (MoE)** | A model that routes each token to a few expert sub-networks: compute follows active parameters, memory follows total parameters. |
| **Mamba** | A selective state-space model whose cost grows linearly with sequence length. |
| **Flow Matching** | A training objective that learns a velocity field moving noise to data, trained with mean squared error. |
| **Consistency Model** | A generator that maps noise directly to data in one (or a few) steps. |
| **JEPA** | Joint-embedding predictive architecture: predict hidden content in embedding space, with no pixel decoder. |
| **Gaussian Splatting** | Representing a 3D scene with millions of explicit learnable Gaussians, for real-time rendering and editing. |
| **Watermark / C2PA** | Marks for generated content: a watermark inside the pixels or audio (SynthID), or signed metadata attached to the file (C2PA). |
| **World Model** | A learned model of the environment, in which a policy can be trained on imagined rollouts (DreamerV3). |
| **Sim-to-Real Gap** | The drop in performance when a policy trained in simulation is deployed on a real robot. |
| **Circuit Tracing** | Mapping the internal paths that cause a model's behavior, tested by suppressing or injecting features. |
| **Bound Propagation** | Cheap outer bounds on a network's outputs, used with branch and bound for verification at scale (α,β-CROWN). |
| **Jagged Frontier** | Model capability is uneven: superhuman on some tasks, failing on others that seem easy. |
| **METR Time Horizon** | The length of task, in expert human time, that an agent completes with 50% success. |
