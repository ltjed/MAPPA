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

# MAPPA: Scaling Multiagent Systems with Process Rewards

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

**TLDR:** Finetuning many agents end-to-end offers a workaround to **continual learning** since different agents can specialize without catastrophic forgetting. Yet this is easier said than done due to **credit assignment** and **sample efficiency** problems. Using AI feedback as **per-action process rewards**, we show how this approach led to real gains across different systems & tasks.

<img src="figures/scaling_hand.jpg" alt="Multiagent scaling" width="100%">

*Multiagent systems sidestep catastrophic forgetting the same way mixture-of-experts does—by allowing different tasks to occupy different parameters.*

---

## Why bother training multiple agents?

Finetuning a single model on one capability often degrades others. Optimize for instruction following and open-ended generation becomes more rigid; train extensively on one language and performance on others may drop. This is catastrophic forgetting: all skills compete for the same parameters.

MoE is popular in part because it provides extra runway to scale a single model further: inputs are routed to different experts with disjoint parameters. Virtually all SOTA models—Gemini 2.5, Kimi K2, Claude Opus 4.5—use MoE. Multiagent systems add another axis for scaling: each agent has entirely separate weights, allowing specialization to occur between agents.

```
Single Model:      [All params] → Output
                   ↑ interference between skills

Multiagent:        [Agent 1: Engineer] ─┐
                   [Agent 2: Modeler]   ─┼→ Pipeline → Output
                   [Agent 3: Analyst]   ─┘
                   ↑ separate params, no interference
```

We demonstrate this with a data science pipeline where three agents collaborate to solve Kaggle-style ML tasks. Each agent has a distinct role:

- `DataEngineer`: performs exploratory analysis, preprocessing, and feature engineering
- `Modeler`: selects algorithms, trains models, and tunes hyperparameters
- `Analyst`: generates predictions and formats the final submission

Agents communicate by passing files: the Data Engineer produces preprocessed data, the Modeler consumes it and saves a trained model, and the Analyst loads both to generate predictions. Each agent can execute Python code in a shared sandbox.

<img src="figures/dsbench_done.jpg" alt="DSBench pipeline with file passing" width="100%">

*Three agents pass files through a shared workspace. The Data Engineer preprocesses CSV files into pickle artifacts; the Modeler trains and saves a model; the Analyst generates the final submission.*

---

## Two challenges

Training multiagent systems end-to-end faces two problems.

**Credit assignment.** When the pipeline fails, which agent is responsible? A `FileNotFoundError` in the Analyst's code might trace back to the Data Engineer forgetting to save a file. With a single outcome reward, all agents receive identical gradients regardless of individual contribution.

**Sample efficiency.** A single rollout involves 3 agents × up to 4 turns × code execution, taking 30+ seconds and significant compute. Yet standard RL provides only one bit of feedback—success or failure—for the entire trajectory.

```
Traditional RL:
  Rollout 1: [Good] → [Good] → [Bug] → FAIL → Reward: 0
  Rollout 2: [Good] → [Bad]  → [...]  → FAIL → Reward: 0
  Rollout 3: [Good] → [Good] → [Good] → SUCCESS → Reward: 1

  Learning signal: 1 bit from 3 expensive rollouts
```

A rollout with 8 good actions and 1 bug looks identical to one where everything failed. The training signal doesn't distinguish between them.

---

## Our approach: per-action process rewards

We address both challenges by having an LLM evaluate each agent action as it happens, rather than waiting for the final outcome.

We call this **MAPPA**: training **M**ulti**A**gent systems with **P**er-action **P**rocess rewards from **A**I feedback.

<img src="figures/execution_loop_hand-drawn.jpeg" alt="Execution loop with coach evaluation" width="100%">

*Each action receives feedback from the coach—not just the final outcome.*

The key insight: when agents execute code, stdout and stderr create a record of what actually happened. An LLM coach can examine this record to determine which agent caused a failure.

For each action, the coach receives:
- The agent's role and the overall task
- The input context the agent observed
- The action the agent generated
- Tool output: stdout, stderr, error messages

When the Analyst crashes with `FileNotFoundError: X_test.pkl not found`, the coach checks the Data Engineer's execution logs. If those logs never mention saving `X_test.pkl`, the coach assigns low scores to the Data Engineer—not the Analyst who correctly attempted to load it.

```
DATAENGINEER evaluation:
- Tool output: "Saved X_train.pkl, y_train.pkl"
- No mention of X_test.pkl
- VERDICT: Failed to save required artifact
- SCORE: 3/10

MODELER evaluation:
- Received expected files
- Tool output: "Saved model.pkl successfully"
- VERDICT: Performed correctly given inputs
- SCORE: 8/10

ANALYST evaluation:
- Required file was never created upstream
- Correctly attempted to load it
- VERDICT: Not at fault
- SCORE: 6/10
```

Credit assignment emerges from giving the coach the right context. No explicit machinery required.

### Handling multi-dimensional metrics

Data science evaluation involves multiple metrics that can conflict. A model might achieve 89% accuracy but 23% F1—high accuracy from predicting the majority class, but failure to capture the minority class signal.

Naive averaging would score this as decent:

```
Simple average: (0.89 + 0.23 + 0.78) / 3 = 0.63
```

The coach recognizes this pattern:

```
Coach reasoning:
High accuracy (0.89) but very low F1 (0.23)
= classic class imbalance problem.
This is a FAILURE despite the good-looking accuracy.
SCORE: 4/10
```

Pre-defined weighting schemes cannot anticipate every situation. The coach provides contextual judgment based on the specific metrics and task.

---

## Results

We trained the three-agent pipeline on 64 Kaggle-style tasks and evaluated on 6 held-out tasks (4 classification, 2 regression).

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Success Rate | 50% | 67% | **+17pp** |
| F1 (Fair) | 0.126 | 0.174 | **+38%** |
| RMSE (Fair) | 24.9% | 14.6% | **-41%** |

Success rate measures whether the pipeline produces valid predictions. Fair metrics penalize failures (0.5 for Accuracy, 0 for F1, 50% for RMSE), so improvements reflect both higher success rates and better quality on successful runs.

### Coach bias

While analyzing training dynamics, we observed an unexpected pattern: regression tasks continued improving while classification performance stagnated. Stratifying coach scores by task type revealed systematic bias—regression actions were scored 0.5–1.8 points higher than equivalent classification actions.

The agents learned to exploit this. Over training, they specialized toward regression, maintaining 87.5% success on those tasks while classification dropped to baseline.

This illustrates a general principle: coach biases compound through training. When using LLM evaluation for RL, behavioral metrics should be monitored to detect this kind of drift.

---

## Limitations and future work

**Current limitations:**
- Coach quality bounds what agents can learn
- Computational cost: ~$50-150 per training run in API calls
- Stateless evaluation misses temporal patterns across actions

**Open directions:**
- **Stateful coaching**: The coach evaluates each action in isolation and cannot detect its own systematic biases. A coach with access to training history could notice patterns like "I've been scoring regression higher—should I adjust?"
- **Reward backpropagation**: Rather than evaluating actions independently, trace backward from outcomes to identify root causes.
- **Beyond scalar rewards**: Coaches could generate corrected actions rather than just scores, enabling hybrid RL+SFT approaches.

---

## Conclusion

We demonstrate that multiagent systems can be trained end-to-end using per-action process rewards from an AI coach. Dense feedback addresses credit assignment, improves sample efficiency, and enables learning even from failed trajectories.

Scaling specialized agents—rather than scaling single models—represents a promising direction for complex, long-horizon tasks. A strong general model can serve as coach to smaller specialists that collectively exceed what the coach achieves alone.

---

## Citation

```bibtex
@article{mappa2026,
  title={MAPPA: Scaling Multiagent Systems with Process Rewards},
  author={Anonymous},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  url={https://anonymous.4open.science/r/ANONYMOUS}
}
```
