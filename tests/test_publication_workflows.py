"""Publication-date and cross-workflow concurrency boundary regressions."""

import json
from pathlib import Path
import subprocess

import pytest
import yaml


_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow(name):
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow):
    # PyYAML's YAML 1.1 loader treats the unquoted Actions key "on" as True.
    return workflow.get("on", workflow.get(True))


def _run_scripts(scripts, *, now, env=None, issues=None):
    """Execute workflow JavaScript with local Actions boundaries and a fixed clock."""
    harness = """
        const fixture = JSON.parse(process.argv[1]);
        const RealDate = Date;
        class FixedDate extends RealDate {
          constructor(...args) {
            super(...(args.length ? args : [fixture.now]));
          }
          static now() { return new RealDate(fixture.now).valueOf(); }
        }
        const env = {...fixture.env};
        const outputs = {};
        const exported = {};
        const core = {
          exportVariable(name, value) { env[name] = value; exported[name] = value; },
          setOutput(name, value) { outputs[name] = value; },
          notice() {}
        };
        const github = {
          async graphql() {
            return {repository: {issues: {
              nodes: fixture.issues,
              pageInfo: {hasNextPage: false, endCursor: null}
            }}};
          }
        };
        const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
        (async () => {
          for (const script of fixture.scripts) {
            await new AsyncFunction("Date", "core", "process", "github", "context", script)(
              FixedDate, core, {env}, github, {repo: {owner: "example", repo: "digest"}}
            );
          }
          console.log(JSON.stringify({env, exported, outputs}));
        })().catch(error => { console.error(error); process.exitCode = 1; });
    """
    fixture = {
        "scripts": scripts,
        "now": now,
        "env": env or {},
        "issues": issues or [],
    }
    result = subprocess.run(
        ["node", "-e", harness, json.dumps(fixture)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", ["digest.yml", "publish-digest.yml"])
def test_publication_workflows_share_a_lock_and_retain_pending_runs(name):
    concurrency = _workflow(name).get("concurrency", {})

    assert concurrency.get("group") == "digest-publish-${{ github.repository }}"
    assert concurrency.get("cancel-in-progress") is False
    assert concurrency.get("queue") == "max"


def test_receiver_requires_and_passes_the_dispatch_date_unchanged():
    workflow = _workflow("publish-digest.yml")
    date_input = _triggers(workflow)["workflow_dispatch"]["inputs"].get("digest_date", {})

    assert date_input.get("required") is True
    assert date_input.get("type") == "string"
    job = workflow["jobs"]["publish"]
    assert job["env"].get("DIGEST_DATE") == "${{ inputs.digest_date }}"
    assert all("DIGEST_DATE" not in step.get("env", {}) for step in job["steps"])


@pytest.mark.parametrize(
    ("now", "expected_date"),
    [
        ("2026-09-08T03:59:59Z", "2026-09-07"),
        ("2026-09-08T04:00:00Z", "2026-09-08"),
        ("2026-01-08T04:59:59Z", "2026-01-07"),
        ("2026-01-08T05:00:00Z", "2026-01-08"),
    ],
)
def test_legacy_workflow_exports_one_eastern_date_before_preflight_and_build(
    now, expected_date
):
    steps = _workflow("digest.yml")["jobs"]["digest"]["steps"]
    preflight_index = next(i for i, step in enumerate(steps) if step.get("id") == "preflight")
    date_steps = [
        step
        for step in steps[:preflight_index]
        if step.get("uses", "").startswith("actions/github-script@")
    ]

    assert date_steps, "The workflow must export its date before preflight and build."
    assert all("if" not in step for step in date_steps)
    result = _run_scripts(
        [step["with"]["script"] for step in date_steps], now=now
    )
    assert result["exported"].get("DIGEST_DATE") == expected_date
    assert result["env"].get("DIGEST_DATE") == expected_date


def test_legacy_preflight_uses_the_frozen_date_when_midnight_passes():
    steps = _workflow("digest.yml")["jobs"]["digest"]["steps"]
    preflight = next(step for step in steps if step.get("id") == "preflight")
    result = _run_scripts(
        [preflight["with"]["script"]],
        now="2026-09-08T04:00:01Z",
        env={"DIGEST_DATE": "2026-09-07"},
        issues=[
            {
                "number": 42,
                "createdAt": "2026-09-07T16:00:00Z",
                "labels": {"nodes": [{"name": "ai-digest"}]},
            }
        ],
    )

    assert result["outputs"]["should_skip"] == "true"


def test_publication_receiver_remains_manual_only():
    assert set(_triggers(_workflow("publish-digest.yml"))) == {"workflow_dispatch"}


def test_legacy_bio_workflow_remains_manual_only():
    assert set(_triggers(_workflow("digest.yml"))) == {"workflow_dispatch"}

