"""The public benchmark CLI; listing and validation never initialize instruments."""
import argparse
import json
from pathlib import Path

from ..benchmarks import available
from .config import StudyConfig, load_config


def memory_name(value):
    names = {"external-notebook": "external-notebook", "full-history": "full-history",
             "C": "external-notebook", "B": "full-history"}
    if not isinstance(value, str) or value not in names:
        raise ValueError("Choose external-notebook or full-history memory")
    return names[value]


def discoverphysics_config(memory="external-notebook", review=False, random=False,
                           requests=100, credits=1000):
    """One explicit native investigation; constructing it never runs the benchmark."""
    memory = memory_name(memory)
    if type(review) is not bool or type(random) is not bool:
        raise ValueError("Review and random must be explicit boolean options")
    name = memory + ("-review" if review else "") + ("-random" if random else "")
    value = {
        "schema_version": 1, "benchmark": "discoverphysics", "protocol": "native-trajectory-v1",
        "worlds": ["yukawa"], "seeds": [12345],
        "model": {"provider": "copilot", "id": "gpt-6.1-sol"},
        "limits": {"requests": requests, "observations": 50, "credits": credits},
        "parameters": {"checkout": ".cache/DiscoverPhysics", "cli_version": "1.0.103-0",
                       "position_scale": {"yukawa": 0.3732285059350784}},
        "arms": [{"name": name, "memory": "external" if memory == "external-notebook" else "inline",
                  "advisor": "design" if review else "off",
                  "selection": "random" if random else "active",
                  "profile": "memory-advisor-v1"}],
    }
    return StudyConfig.parse(value)


def newtonbench_config(memory="external-notebook", review=False, random=False,
                       requests=16, credits=0):
    if type(review) is not bool or type(random) is not bool:
        raise ValueError("Review and random must be explicit boolean options")
    if memory_name(memory) != "external-notebook" or review or random:
        raise ValueError("NewtonBench reference proof supports external-notebook, no review and active selection")
    return StudyConfig.parse({
        "schema_version": 1, "benchmark": "newtonbench", "protocol": "scientist-hooke-reference-v1",
        "worlds": ["vanilla_equation"], "seeds": [0],
        "model": {"provider": "baseline", "id": "reference"},
        "limits": {"requests": requests, "observations": 8, "credits": credits},
        "parameters": {"checkout": ".cache/NewtonBench"},
        "arms": [{"name": "external-notebook-reference", "memory": "external",
                  "profile": "notebook-reference-v1"}],
    })


def discovery_config(benchmark):
    if benchmark not in ("newtonbench-discovery", "made"):
        raise ValueError("Choose a registered discovery integration")
    made = benchmark == "made"
    parameters = {"checkout": ".cache/MADE" if made else ".cache/NewtonBench",
                  "cli_version": "1.0.103-0"}
    if made:
        parameters["worker_python"] = ".cache/made-analytic-env/bin/python"
    return StudyConfig.parse({
        "schema_version": 1, "benchmark": benchmark,
        "protocol": "scientist-made-analytic-v1" if made else "scientist-hooke-discovery-v1",
        "worlds": ["lj-crystal" if made else "vanilla_equation"], "seeds": [20261118],
        "model": {"provider": "copilot", "id": "claude-opus-5.5"},
        "limits": {"requests": 24, "observations": 12, "credits": 1500},
        "parameters": parameters,
        "arms": [{"name": "external-notebook", "memory": "external", "profile": "notebook-discovery-v1"}],
    })


def configure(parser):
    commands = parser.add_subparsers(dest="bench_command", required=True)
    commands.add_parser("list", help="show adapters, protocols, policies and runtime requirements")
    init = commands.add_parser("init", help="write one config; default external notebook, no dispatch")
    init.add_argument("--output", required=True, help="a NEW configuration file")
    init.add_argument("--benchmark", choices=("discoverphysics", "newtonbench", "newtonbench-discovery", "made"),
                      default="discoverphysics")
    init.add_argument("--memory", choices=("external-notebook", "full-history", "B", "C"),
                      metavar="{external-notebook,full-history}", default="external-notebook",
                      help="descriptive memory policy; legacy B/C aliases remain accepted")
    init.add_argument("--review", action="store_true")
    init.add_argument("--random", action="store_true")
    init.add_argument("--requests", type=int)
    init.add_argument("--credits", type=float)
    validate = commands.add_parser("validate", help="validate and enumerate a study without executing it")
    validate.add_argument("config")
    run = commands.add_parser("run", help="run an explicit study into a NEW output directory")
    run.add_argument("config")
    run.add_argument("--output")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--approve-credits", type=str, help="NEW combined study cap, exactly matching the config")
    report = commands.add_parser("report", help="aggregate saved results without model calls")
    report.add_argument("directory")


def execute(args):
    if args.bench_command == "list":
        return {"benchmarks": available()}
    if args.bench_command == "init":
        if args.benchmark in ("newtonbench-discovery", "made"):
            if args.review or args.random or args.memory not in ("external-notebook", "C"):
                raise ValueError("Discovery integrations currently require external-notebook, no review/random")
            config = discovery_config(args.benchmark)
            if args.requests is not None:
                config.value["limits"]["requests"] = args.requests
            if args.credits is not None:
                config.value["limits"]["credits"] = args.credits
            config = StudyConfig.parse(config.value)
            with Path(args.output).open("x", encoding="utf-8") as handle:
                json.dump(config.value, handle, indent=2, allow_nan=False)
                handle.write("\n")
            return {"config": str(args.output), "episodes": len(config.cells()), "dispatches": 0}
        native = args.benchmark == "discoverphysics"
        factory = discoverphysics_config if native else newtonbench_config
        requests = args.requests if args.requests is not None else (100 if native else 16)
        credits = args.credits if args.credits is not None else (1000 if native else 0)
        config = factory(args.memory, args.review, args.random, requests, credits)
        with Path(args.output).open("x", encoding="utf-8") as handle:
            json.dump(config.value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return {"config": str(args.output), "episodes": len(config.cells()), "dispatches": 0}
    if args.bench_command == "report":
        from ..reporting import report_study
        return report_study(args.directory)
    config = load_config(args.config)
    if args.bench_command == "validate" or args.dry_run:
        return config.describe()
    if not args.output:
        raise ValueError("bench run requires --output for a new study directory")
    from .runner import run_study
    return run_study(config, args.output, args.approve_credits)


def main(argv=None):
    """Small standalone entry point, without the historical research CLI."""
    parser = argparse.ArgumentParser(prog="science-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    configure(commands.add_parser("bench", help="configure, run and report explicit benchmark studies"))
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
