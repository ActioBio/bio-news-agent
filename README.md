# Daily Biotech & Pharma News Digest

One short email a day with the biotech and pharma news that matters — picked from 15 trusted sources (STAT, Endpoints News, BioPharma Dive, FierceBiotech, BioSpace, FDA, EMA, and more) and readable in two minutes.

**[📬 Subscribe to the daily email](https://github.com/ActioBio/bio-news-agent/subscription)** · **[📖 Read the latest digest](https://github.com/ActioBio/bio-news-agent/issues?q=label%3A%22ai-digest%22)**

## How to subscribe (about 30 seconds)

Delivery is handled by GitHub's built-in notifications — free, no newsletter service, no signup form, no ads. You just need a free [GitHub account](https://github.com/signup).

1. Open the **[subscription page](https://github.com/ActioBio/bio-news-agent/subscription)** — it's this repository's "Watch" menu.
2. Choose **Custom**, tick **Issues**, and click **Apply**. Each digest is published here as a public daily post, and "Issues" is GitHub's name for those posts.
3. Done — new digests arrive in your inbox each day.

Normally that's one email per day — the digest itself. To stop, open the same page and choose **Unwatch**. If nothing arrives, check that email is enabled in your [notification settings](https://github.com/settings/notifications).

## What it looks like

<img src="docs/images/digest-email-preview.svg" alt="Example of the daily digest email: top stories and topic sections with linked headlines" width="680">

From the **June 12, 2026** digest:

- **[MHRA approves oral GLP-1 for weight loss](https://www.gov.uk/government/news/first-glp-1-tablet-for-weight-loss-approved-in-the-uk)** — MHRA
- **[Novartis reports Avidity dystrophy data](https://www.biospace.com/drug-development/novartis-12b-avidity-buy-pays-dividends-with-phase-1-2-muscular-dystrophy-win)** — BioSpace Drug Development
- **[WuXi AppTec sues Pentagon](https://endpoints.news/wuxi-apptec-sues-pentagon-in-challenge-over-inclusion-on-chinese-military-list/)** — Endpoints News

Each digest leads with a compact list of top stories, then groups the rest by topic — Regulatory & FDA, Clinical & Research, Deals & Finance, Company News — so you can skim straight to what interests you.

## How it works

Every day, an automated workflow reads the headlines published by the sources in [`feeds.json`](feeds.json), removes duplicates, groups related stories, and asks an AI model to pick the most important ones. The result goes up as a public daily post on this repository, and GitHub emails it to everyone watching.

## FAQ

- **Do I need to be technical?** No. If you can tick a checkbox, you can subscribe.
- **Why GitHub instead of a newsletter?** There's no mailing list and no tracking — GitHub's own notification system delivers the email, and every past digest stays publicly readable.
- **Who picks the stories?** An AI model ranks each day's headlines. The source list is public in [`feeds.json`](feeds.json), so you can see exactly where the news comes from.
- **Can I suggest a source?** Yes — open an issue with the feed you'd like added.

## For developers

[![Daily Biotech News Digest](https://github.com/ActioBio/bio-news-agent/actions/workflows/digest.yml/badge.svg)](https://github.com/ActioBio/bio-news-agent/actions/workflows/digest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- [docs/development.md](docs/development.md) — setup, agent-driven mode, feed configuration
- [docs/architecture.md](docs/architecture.md) — pipeline diagrams and design notes
- [AGENTS.md](AGENTS.md) — runbook for Codex / Claude Code automation

## License

[MIT](LICENSE)
