<style>
body {
  font-size: 22px !important;
  line-height: 1.9 !important;
  max-width: 900px !important;
  margin: 0 auto !important;
  padding: 20px !important;
}

h1 {
  font-size: 1.8em !important;
  line-height: 1.3 !important;
}

h2 {
  font-size: 1.4em !important;
  margin-top: 2.5em !important;
  line-height: 1.4 !important;
}

h3 {
  font-size: 1.2em !important;
  line-height: 1.5 !important;
}

p, li {
  font-size: 22px !important;
  line-height: 1.9 !important;
}

blockquote p {
  font-size: 22px !important;
}

code {
  font-size: 20px !important;
}

pre code {
  font-size: 18px !important;
}

h1 code {
  font-size: 0.9em !important;
}

h2 code {
  font-size: 0.9em !important;
}

h3 code {
  font-size: 0.9em !important;
}

code:not([class]) {
  background-color: transparent !important;
  color: inherit !important;
  padding: 0 !important;
  border: none !important;
  font-weight: bold !important;
  font-size: 1em !important;
}
</style>

# MAPPA: Scaling Multiagent LLM Systems with Process Rewards

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

**TLDR:** Finetuning many agents end-to-end offers a workaround to **continual learning** since different agents can specialize without catastrophic forgetting. Yet doing so is hard due to **credit assignment** and **sample efficiency**. Using AI feedback as **per-action process rewards**, we demonstrate this approach is feasible and led to real gains on different systems & tasks.

<img src="figures/scaling_hand.jpg" alt="Multiagent scaling" width="100%">

*Multiagent systems sidestep catastrophic forgetting the same way mixture-of-experts does—by giving different skills different parameters.*

---

## Why bother training multiple agents?

Fine-tuning a single model on one capability often degrades others. Optimize for instruction following, and open-ended generation becomes more rigid; train extensively on one language, and performance on others may drop. This is catastrophic forgetting: all skills compete for the same parameters.

Mixture-of-experts (MoE) architectures address this by routing different inputs to different parameter subsets. This insight now underpins most frontier models—Gemini 2.5, Kimi K2, and Claude Opus 4.5 all use MoE designs. Multiagent systems apply the same principle at a higher level: each agent has entirely separate weights, so improving one agent's capabilities cannot interfere with another's. Recent work[^1] suggests this kind of specialization emerges naturally—reasoning models trained purely for accuracy spontaneously develop diverse internal "personas."

## Why existing frameworks stop at prompting

So far, multiagent frameworks implement specialization only through system prompts—assigning different personas or instructions to each agent. This is because training all agents end-to-end faces two fundamental challenges:

**Credit assignment.** When a pipeline fails, which agent is responsible? A three-agent data science pipeline might fail with `FileNotFoundError: X_test.pkl not found`. The error appears in the final agent's code, but the root cause could be upstream—an earlier agent forgot to save that file. With standard outcome-based rewards, all agents receive the same penalty regardless of fault.

**Sample efficiency.** Multiagent rollouts are expensive. A single run might involve three agents, each taking multiple turns with code execution. This can take 30+ seconds and cost real money in API calls. Yet traditional RL provides only one bit of feedback at the end: success or failure. A rollout with eight good actions and one bug looks identical to one where everything failed.

## Our approach: per-action process rewards

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
