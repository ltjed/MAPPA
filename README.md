# MAPPA Blog

Blog post for **MAPPA: Training Multiagent Systems with Per-action Process Rewards from AI Feedback**

Live at: https://freephdlabor.github.io/mappa/

## About

This blog post describes MAPPA, a framework for training multi-agent AI systems using process rewards from an LLM coach. Key contributions:

- **Per-action feedback**: Every agent action gets evaluated, not just final outcomes
- **Credit assignment**: The coach can trace errors back to their source agent
- **End-to-end training**: Agents learn to coordinate through shared training

## Local Development

```bash
# Install Jekyll
gem install bundler jekyll

# Serve locally
bundle exec jekyll serve
```

Then open http://localhost:4000/mappa/

