"""Prevent unattended dependency updates from violating project version ranges."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "requirements" / "requirements-macos-py313.txt"


def test_pinned_snapshot_name_and_ci_reference():
    assert PINS.is_file()
    assert not (ROOT / "requirements" / "macos-py313.txt").exists()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "requirements/requirements-macos-py313.txt" in ci


def test_pinned_direct_dependencies_honor_project_bounds():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    declared = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    parsed = {r.name.lower(): r for spec in declared if (r := Requirement(spec))}
    installed = {}
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pinned = Requirement(line)
            installed[pinned.name.lower()] = pinned
    for name, requirement in parsed.items():
        assert name in installed, f"Missing pinned direct dependency: {name}"
        exact = [s.version for s in installed[name].specifier if s.operator == "=="]
        assert len(exact) == 1, f"Missing exact pin for {name}"
        assert requirement.specifier.contains(exact[0]), (
            f"{name} pin {exact[0]} is outside pyproject range {requirement.specifier}"
        )
        for bound in requirement.specifier:
            if bound.operator in ("<", "<=", ">", ">="):
                assert any(
                    pin.operator == bound.operator and pin.version == bound.version
                    for pin in installed[name].specifier
                ), f"{name} must repeat pyproject bound {bound} in requirements"
