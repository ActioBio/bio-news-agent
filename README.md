# Bio News Agent

[![Daily Biotech / Pharma Digest](https://github.com/ActioBio/bio-news-agent/actions/workflows/digest.yml/badge.svg)](https://github.com/ActioBio/bio-news-agent/actions/workflows/digest.yml)
[![Read latest digests](https://img.shields.io/badge/Read-latest%20digests-181717?logo=github&logoColor=white)](https://github.com/ActioBio/bio-news-agent/issues?q=label%3A%22ai-digest%22)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A short daily digest of biotech and pharma news, delivered to your inbox. Headlines are collected from trusted industry, regulatory, and research sources, deduplicated, grouped by topic, and posted as a GitHub Issue every day.

## Get the daily digest

🔔 **[Watch this repository](https://github.com/ActioBio/bio-news-agent/subscription)** with notifications set to **Issues**. GitHub will email you each new digest. A GitHub account is required for email notifications; no separate newsletter service.

[**→ Browse all digests**](https://github.com/ActioBio/bio-news-agent/issues?q=label%3A%22ai-digest%22)

## What it looks like

Top stories from **May 13, 2026**:

- **[FDA chief Makary resigns](https://www.biopharmadive.com/news/makary-fda-commissioner-resign-trump/819757/)** — BioPharma Dive
- **[Gene therapy viruses linked to tumor](https://www.statnews.com/2026/05/13/gene-therapy-cancer-risks-mps-hurler-syndrome/?utm_campaign=rss)** — STAT Biotech
- **[Isomorphic raises $2.1 billion](https://www.biospace.com/business/ai-fueled-isomorphic-bags-2-1b-the-second-largest-biotech-round-ever)** — BioSpace Business

*[Browse latest digests →](https://github.com/ActioBio/bio-news-agent/issues?q=label%3A%22ai-digest%22)*

## How it works

Every morning, an automated workflow reads the RSS feeds listed in [`feeds.json`](feeds.json), removes duplicates, groups related stories, and asks an LLM to pick the most important ones. The result is posted as one skimmable GitHub Issue.

## For developers

- [docs/development.md](docs/development.md) — setup, agent-driven mode, feed configuration
- [docs/architecture.md](docs/architecture.md) — pipeline diagrams and design notes
- [AGENTS.md](AGENTS.md) — runbook for Codex / Claude Code automation

## License

[MIT](LICENSE)
