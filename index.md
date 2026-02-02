<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  font-size: 18px !important;
  line-height: 1.75 !important;
  max-width: 680px !important;
  margin: 0 auto !important;
  padding: 20px !important;
  color: #1a1a1a !important;
  background-color: #f9f7f3 !important;
}

h1 {
  font-size: 2.2em !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
  margin-bottom: 0.5em !important;
  max-width: 900px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  position: relative !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: 100vw !important;
  max-width: 900px !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
  box-sizing: border-box !important;
}

h1 a {
  color: #2563eb !important;
  text-decoration: none !important;
}

h1 a:hover {
  text-decoration: underline !important;
}

h2 {
  font-size: 1.5em !important;
  font-weight: 600 !important;
  margin-top: 2.5em !important;
  line-height: 1.3 !important;
}

h3 {
  font-size: 1.25em !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
  margin-top: 2em !important;
}

p, li {
  font-size: 18px !important;
  line-height: 1.75 !important;
}

blockquote p {
  font-size: 18px !important;
}

code {
  font-family: 'JetBrains Mono', 'SF Mono', Monaco, monospace !important;
  font-size: 15px !important;
}

pre code {
  font-size: 14px !important;
}

h1 code, h2 code, h3 code {
  font-size: 0.85em !important;
}

code:not([class]) {
  background-color: #f5f5f5 !important;
  color: #1a1a1a !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
  font-weight: 500 !important;
  font-size: 0.9em !important;
}
</style>

# How to finetune any multiagent system on any task

<p align="center">
  <a href="https://github.com/freephdlabor/mappa">
    <img src="https://img.shields.io/badge/try_mappa_now-black?style=for-the-badge&logo=github" alt="Try MAPPA Now!">
  </a>
  <a href="https://github.com/freephdlabor/mappa">
    <img src="https://img.shields.io/github/stars/freephdlabor/mappa?style=for-the-badge&color=gold" alt="Stars">
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv" alt="arXiv">
  </a>
</p>

**TLDR:** <u>Finetuning many agents end-to-end offers a workaround to <strong>continual learning</strong></u> since different agents can specialize without catastrophic forgetting. Yet doing so is hard due to **credit assignment** and **sample efficiency**. We found that using AI feedback as **per-action process rewards** holds promise for addressing these challenges and unlocks a new axis for scaling post-training.

<img src="figures/scaling_hand.jpg" alt="Multiagent scaling" width="100%">

*Multiagent systems sidestep catastrophic forgetting the same way mixture-of-experts does—by giving different skills different parameters.*

---

## Why bother training more than 1 agent?

Finetuning a single model on one capability often degrades others. Train extensively on one language, and performance on others may drop. This is catastrophic forgetting: all tasks compete for the same parameters. MoE architectures partially solves this by routing different inputs to different parameter subsets, creating more runway to scale (more training to be done without forgetting) in one, big model. Almost all frontier models—Gemini 2.5, Kimi K2, and Claude Opus 4.5 all use MoE designs nowadays. Multiagent systems apply the same idea at the agent-level, each agent having its own weights to be finetuned separately. Thus, if coordinated right, # of agents could be the next dimension of scaling.

## What makes 

So far, most multiagent frameworks implement specialization by assigning different personas or instructions to each agent, leaving the weights separation advantage completely untapped. This is because training all agents end-to-end faces two fundamental challenges:

**Credit assignment.** When a task succeeds/fails, which agent is responsible? A data science pipeline might fail with `FileNotFoundError`. The error may show up first when the final agent tries to access the file, when root cause is an earlier agent forgetting to save that file. Under current RL approaches, all agents share the final, outcome score regardless, and in doing so penalizing the final agent for doing the right

**Sample efficiency.** Multiagent rollouts are expensive. A single run could easily involve generating dozens of actions from different LLMs, each containing tool calls to be executed by the environment, taking minutes if not hours at a time. Yet current RL approaches only provides one training signal at the end. Making it very much like "sucking supervision from a straw."

## per-action process rewards from AI feedback

We address both challenges by having an LLM coach evaluate every action as it happens—not just the final outcome.

<img src="figures/execution_loop_hand-drawn.jpeg" alt="Execution loop with coach evaluation" width="100%">

*With per-action evaluation, every step gets feedback—not just the final outcome.*

The coach receives context that enables accurate credit assignment:
- The agent's role and what it was asked to do
- What the agent saw before acting
- What the agent generated
- Tool output: stdout, stderr, error messages

When the final agent crashes with `FileNotFoundError`, the coach checks the earlier agents' tool outputs. If no agent ever saved `X_test.pkl`, blame goes to whoever should have created it—not the agent that correctly tried to load it.

We call this approach **MAPPA**: training **M**ulti**A**gent systems with **P**er-action **P**rocess rewards from **A**I feedback.

---

## Extended example: data science pipelines

To demonstrate MAPPA on a realistic long-horizon task, we train a three-agent pipeline on Kaggle-style machine learning problems. Each task provides CSV files and requires generating predictions for held-out test data.

<img src="figures/dsbench_done.jpg" alt="DSBench pipeline with file passing" width="100%">

*Agents pass files to each other through a shared workspace—creating a paper trail the coach can examine.*

### Pipeline structure

The three agents form a sequential pipeline:

- **Data Engineer**: Explores the data, handles preprocessing, engineers features. Saves processed data as pickle files.
- **Modeler**: Loads the processed data, selects algorithms, trains models, tunes hyperparameters. Saves the trained model.
- **Analyst**: Loads the model and test data, generates predictions, formats the submission file.

Each agent can take up to 4 turns, executing Python code in a sandboxed environment. Agents communicate by reading and writing files to a shared workspace.

### How the coach assigns credit

The file-passing structure makes credit assignment tractable. When something fails, the coach examines the evidence:

```
DATAENGINEER evaluation:
- Tool output: "Saved X_train.pkl, y_train.pkl"
- No mention of X_test.pkl
- VERDICT: Failed to save required artifact
- SCORE: 3/10

MODELER evaluation:
- Received expected files from Data Engineer
- Tool output: "Saved model.pkl successfully"
- VERDICT: Completed task correctly given inputs
- SCORE: 8/10

ANALYST evaluation:
- Required file X_test.pkl was never created upstream
- Correctly attempted to load it
- VERDICT: Not at fault for the failure
- SCORE: 6/10
```

The coach reads the receipts. No counterfactual reasoning required—just checking what each agent actually produced.

### Handling messy real-world metrics

Data science evaluation is not straightforward. A model might achieve 89% accuracy but 23% F1 score. Naive averaging would call this "decent," but it actually indicates failure—the model learned to predict the majority class.

The coach understands this context:

```
Coach reasoning:
High accuracy (0.89) but very low F1 (0.23)
indicates a class imbalance problem.
The model is not learning the actual signal.
SCORE: 4/10
```

By synthesizing multiple metrics in context, the coach provides judgment that simple averaging cannot.

### Results

We train for 21 epochs on 64 tasks and evaluate on 6 held-out tasks:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Success Rate | 50.0% | 66.7% | **+16.7pp** |
| Accuracy (Fair) | 0.583 | 0.719 | **+23%** |
| F1 (Fair) | 0.126 | 0.174 | **+38%** |
| RMSE (Fair) | 24.9% | 14.6% | **-41%** |

*Fair metrics penalize failed runs rather than ignoring them.*

Training improves both success rate and quality metrics. The coach's per-action feedback translates to downstream improvements on held-out tasks.

We also validate MAPPA on competition math problems with a different multiagent configuration, achieving +5–17pp improvements on AIME and AMC benchmarks. See the paper for details.

---

## Coach biases compound

While analyzing training dynamics, we discovered something unexpected: our coach had preferences we did not program.

Regression tasks kept improving while classification stagnated. Examining the scores revealed systematic bias—regression actions were scored 0.5–1.8 points higher than equivalent classification actions.

The agents figured this out before we did. Over training, they specialized toward regression, maintaining 87.5% success on those tasks while classification dropped back to baseline.

This illustrates a key limitation: coach biases get amplified through training. If you use LLM evaluation for training, you need to monitor for exactly this kind of drift.

---

## What's next

We showed that multiagent systems can be trained end-to-end using process rewards from an LLM coach. Dense per-action feedback addresses credit assignment, improves sample efficiency, and works across domains.

The broader direction: scaling specialized agents—not just scaling single models—may be a promising path for complex tasks. A strong general model serves as coach to a team of smaller specialists that can collectively exceed what the coach could do alone.

Open questions remain:

- **Stateful coaching**: Our coach evaluates each action in isolation. A smarter coach might track its own scoring patterns and adjust for detected biases.
- **Reward backpropagation**: Instead of evaluating each action independently, trace backward from outcomes to identify root causes.
- **Beyond scalar rewards**: Coaches could generate corrected actions, not just scores—enabling hybrid RL and supervised learning approaches.

Current limitations:
- Coach quality bounds what agents can learn
- Computational cost runs ~$50–150 per training run in API calls
- Stateless evaluation misses temporal patterns

We are entering an era where AI systems increasingly involve multiple agents working together. Figuring out how to train and evaluate these systems is becoming critical. This is our first step toward making that tractable.

---

## Citation

```bibtex
@article{mappa2026,
  title={MAPPA: Scaling Multiagent LLM Systems with Process Rewards},
  author={Anonymous},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  url={https://anonymous.4open.science/r/ANONYMOUS}
}
```

---

## References

[^1]: Kim, T., et al. (2026). *Reasoning Models Generate Societies of Thought*. arXiv preprint.
