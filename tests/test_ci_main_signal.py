"""Bounded nightly main signal without expanding the per-push runner queue. No hosted CI required."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {"ci": "23 3 * * *", "runtime": "53 3 * * *"}
SUFFIX = (
    "${{ github.event_name }}-${{ github.event.pull_request.number || "
    "(github.event_name == 'push' && github.ref) || github.run_id }}"
)


def workflow(name):
    return (ROOT / ".github" / "workflows" / (name + ".yml")).read_text(encoding="utf-8")


def group(name, event, run_id, *, ref="refs/heads/main", pr=None):
    """Model the exact expression asserted below, not GitHub's scheduler or runner availability."""
    key = pr or (ref if event == "push" else None) or run_id
    return f"{name}-{event}-{key}"


class MainSignal(unittest.TestCase):
    def test_one_staggered_daily_schedule_per_workflow(self):
        for name, cron in WORKFLOWS.items():
            with self.subTest(workflow=name):
                text = workflow(name)
                self.assertEqual(re.findall(r'^    - cron: "([^"]+)"$', text, re.M), [cron])
                self.assertIn("  schedule:\n", text)
                self.assertEqual(cron.split()[2:], ["*", "*", "*"])
        self.assertNotEqual(WORKFLOWS["ci"], WORKFLOWS["runtime"])

    def test_push_pr_and_manual_triggers_remain(self):
        for name in WORKFLOWS:
            text = workflow(name)
            for trigger in ("push", "pull_request", "workflow_dispatch"):
                self.assertRegex(text, rf"(?m)^  {trigger}:")
            self.assertRegex(text, r"(?m)^    branches: \[main\]$")

    def test_existing_coalescing_expression_is_unchanged(self):
        for name in WORKFLOWS:
            self.assertIn("  group: " + name + "-" + SUFFIX + "\n", workflow(name))

    def test_only_pull_requests_cancel_running_jobs(self):
        for name in WORKFLOWS:
            self.assertIn(
                "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n",
                workflow(name),
            )

    def test_groups_keep_bursts_bounded_and_schedules_independent(self):
        for name in WORKFLOWS:
            self.assertEqual(group(name, "push", 1), group(name, "push", 2))
            self.assertEqual(group(name, "pull_request", 1, pr=771), group(name, "pull_request", 2, pr=771))
            self.assertNotEqual(group(name, "pull_request", 1, pr=771), group(name, "pull_request", 2, pr=782))
            self.assertNotEqual(group(name, "schedule", 1), group(name, "push", 1))
            self.assertNotEqual(group(name, "schedule", 1), group(name, "schedule", 2))
            self.assertNotEqual(group(name, "workflow_dispatch", 1), group(name, "schedule", 1))
        self.assertNotEqual(group("ci", "schedule", 1), group("runtime", "schedule", 1))

    def test_jobs_are_not_restricted_to_push_events(self):
        for name in WORKFLOWS:
            jobs = workflow(name).split("\njobs:\n", 1)[1]
            self.assertNotRegex(jobs, r"(?m)^\s+if:.*github\.event_name\s*==\s*'push'")
        # The Wine artifact consumer remains available to schedules, but still needs a real artifact.
        runtime = workflow("runtime")
        self.assertIn("github.event_name != 'pull_request'", runtime)
        self.assertIn("needs.windows.outputs.wine-artifact == 'success'", runtime)


if __name__ == "__main__":
    unittest.main()
