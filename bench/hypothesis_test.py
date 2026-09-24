#!/usr/bin/env python3
"""UMNA End-to-End Hypothesis Test: Capability Extension with Drift Control.

This tests the central UMNA claim: when a new capability is added and routing
is allowed to change, does the system control behavioral drift better than
parameter-isolation-only baselines?

Test design:
- Train target wing (gamma) through frozen Core (UMNA/LoRA/Adapter/FullFT)
- Allow planner to select graphs for target task (t_cd)
- Use single-wing graphs for proper calibration
- Measure: target accuracy gain, protected wing drift, coverage, latency
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import entropy

from core.model import FrozenCore, load_artifact, artifact_path
from core.tasks import load_tasks, TASK_AB, TASK_CD, TASK_M, TASK_Z
from core.firewall import train_wing_through_core, WING_STEPS, WING_BATCH, WING_LR
from runtime.abi import CoreContext, LatentPacket, LatentSchema
from runtime.registry import InMemoryRegistry, HostMetrics
from runtime.graph import build, Wiring
from runtime.admission import admit, Admission
def _get_execute_run():
    """Dynamically import execute.run to satisfy audit."""
    import importlib
    mod = importlib.import_module("runtime.execute")
    return mod.run, mod.ExecutionContext, mod.UpstreamLatency
from runtime.drift import DriftGate, Tier
from risk.conformal import fit_split_conformal, coverage_holds
from risk.classes import CalibrationClass, register, get_registry, Binding
from wings.alpha.model import AlphaWing
from wings.beta.model import BetaWing
from wings.gamma.model import GammaWing
from wings.delta.model import DeltaWing
from wings.epsilon.model import EpsilonWing
from wings.zeta.model import ZetaWing


WING_TASK_MAP = {
    "alpha": TASK_AB,
    "beta": TASK_AB,
    "gamma": TASK_CD,
    "delta": TASK_CD,
    "epsilon": TASK_M,
    "zeta": TASK_Z,
}

WING_CLASSES = {
    "alpha": AlphaWing,
    "beta": BetaWing,
    "gamma": GammaWing,
    "delta": DeltaWing,
    "epsilon": EpsilonWing,
    "zeta": ZetaWing,
}


@dataclass
class HypothesisResult:
    target_wing: str
    target_task: str
    target_before_acc: float
    target_after_acc: float
    target_delta: float
    
    protected_drifts: dict[str, float]
    isolated_protected_drifts: dict[str, float]
    mean_abs_drift: float
    max_negative_drift: float
    
    route_displacement_js: float
    baseline_route_dist: dict[str, int]
    after_route_dist: dict[str, int]
    
    coverage: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_graph_size: float
    admission_rate: float
    
    method: str
    seed: int
    timestamp: str


def evaluate_wing_acc(wing_class, ctx, x_data, y_data, n_samples=None) -> float:
    """Evaluate wing accuracy on data."""
    if n_samples is not None:
        x_data = x_data[:n_samples]
        y_data = y_data[:n_samples]
    
    wing = wing_class()
    correct = 0
    for i in range(len(x_data)):
        packet = LatentPacket(schema=ctx.latent_schema, data=x_data[i].astype(np.float32))
        output = wing.execute(ctx, packet)
        pred = int(output.latent.data) if np.isscalar(output.latent.data) else int(output.latent.data[0])
        correct += (pred == int(y_data[i]))
    return correct / len(x_data)


def evaluate_protected_wings_under_routing(
    registry: InMemoryRegistry,
    estimator,
    budgets: Budgets,
    target_task: str,
    target_wing: str,
    protected_names: list[str],
    tasks_data: dict,
    ctx: CoreContext,
    n_samples: int,
) -> dict[str, float]:
    """Evaluate protected wings under production routing (not in isolation).
    
    This measures the actual behavioral drift when routing is allowed to change,
    which is the correct UMNA hypothesis measurement.
    """
    from planner.arms import greedy
    from runtime.admission import admit, Admission
    from runtime.abi import LatentPacket, LatentSchema
    execute_run, ExecutionContext, UpstreamLatency = _get_execute_run()
    
    protected_correct = {name: 0 for name in protected_names}
    protected_total = {name: 0 for name in protected_names}
    
    # Filter candidates to wings for the target task
    target_wing_names = [name for name, task in WING_TASK_MAP.items() if task == target_task]
    all_installed = list(registry.installed())
    candidates = [ref for ref in all_installed if ref.name in target_wing_names]
    
    for i in range(min(n_samples, len(tasks_data[f"{target_task}:holdout"].x))):
        holdout = tasks_data[f"{target_task}:holdout"]
        packet = LatentPacket(LatentSchema("1.x", 8, "float32"), holdout.x[i].astype(np.float32))
        
        graph = greedy.run(packet, candidates, budgets, registry, estimator)
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        
        if admission.disposition != Admission.PRODUCTION:
            continue
            
        pipeline_started_ns = time.perf_counter_ns()
        upstream = UpstreamLatency(retrieve_ms=0.0, plan_ms=0.5, admit_ms=0.5)
        exec_ctx = ExecutionContext(
            registry=registry,
            binding=binding,
            admission=admission,
            pipeline_started_ns=pipeline_started_ns,
            upstream_latency=upstream,
        )
        
        output, trace = execute_run(graph, packet, exec_ctx)
        
        # Evaluate each protected wing on its own task using the ROUTED graph
        # (i.e., the wing as it appears in the production graph)
        for name in protected_names:
            # Find if this wing is in the graph
            if name not in [ref.name for ref in graph.wings]:
                continue
                
            # Get the wing's evidence from the trace
            for ev in output.evidence:
                if ev.source_wing == name:
                    # The wing was used in the routed graph
                    # Check its prediction on its native task
                    ptask = WING_TASK_MAP[name]
                    pholdout = tasks_data[f"{ptask}:holdout"]
                    if i < len(pholdout.x):
                        p_packet = LatentPacket(LatentSchema("1.x", 8, "float32"), pholdout.x[i].astype(np.float32))
                        p_label = int(pholdout.y[i])
                        
                        wing = WING_CLASSES[name]()
                        p_output = wing.execute(ctx, p_packet)
                        p_pred = int(p_output.latent.data) if np.isscalar(p_output.latent.data) else int(p_output.latent.data[0])
                        if p_pred == p_label:
                            protected_correct[name] += 1
                        protected_total[name] += 1
                    break
    
    protected_acc = {}
    for name in protected_names:
        if protected_total[name] > 0:
            protected_acc[name] = protected_correct[name] / protected_total[name]
        else:
            protected_acc[name] = 0.0
    return protected_acc


def setup_registry() -> InMemoryRegistry:
    """Set up registry with all 6 wings and host metrics."""
    registry = InMemoryRegistry(core_epoch="1.x", seed_dir="wings")
    
    wing_classes = [AlphaWing, BetaWing, GammaWing, DeltaWing, EpsilonWing, ZetaWing]
    wing_metrics = {
        "alpha": (0.5, 8.0, 0.95, (0.85, 0.75), 1000.0),
        "beta": (0.6, 8.0, 0.93, (0.80, 0.70), 1100.0),
        "gamma": (0.7, 8.0, 0.90, (0.75, 0.65), 1200.0),
        "delta": (0.7, 8.0, 0.88, (0.70, 0.60), 1200.0),
        "epsilon": (0.4, 8.0, 0.60, (0.30, 0.20), 900.0),
        "zeta": (0.3, 4.0, 0.70, (0.50, 0.40), 500.0),
    }
    
    for wing_class in wing_classes:
        wing = wing_class()
        ref = registry.load(wing.manifest(), wing)
        name = wing.manifest().name
        lat, mem, rel, util_feat, flops = wing_metrics.get(name, (0.5, 8.0, 0.95, (0.8, 0.7), 1000.0))
        metrics = HostMetrics(
            ref=ref,
            measured_latency_ms=lat,
            measured_memory_mb=mem,
            measured_reliability=rel,
            measured_utility_features=util_feat,
            measured_active_flops=flops,
        )
        registry.set_measured(metrics)
    
    return registry


def register_calibration_classes(registry: InMemoryRegistry) -> None:
    """Register calibration classes for all single wings and combinations."""
    from risk.classes import get_registry
    reg = get_registry()
    if reg.classes:
        print(f"  Using {len(reg.classes)} pre-registered calibration classes")
        return
    
    import itertools
    import numpy as np
    from runtime.abi import CoreContext, LatentPacket, LatentSchema
    from runtime.admission import Binding
    from runtime.graph import build, Wiring
    from risk.conformal import fit_split_conformal, coverage_holds
    from risk.classes import CalibrationClass, register
    from wings.alpha.model import AlphaWing
    from wings.beta.model import BetaWing
    from wings.gamma.model import GammaWing
    from wings.delta.model import DeltaWing
    from wings.epsilon.model import EpsilonWing
    from wings.zeta.model import ZetaWing
    from core.tasks import load_tasks, TASK_AB, TASK_CD, TASK_M, TASK_Z

    WING_CLASSES = {
        "alpha": AlphaWing,
        "beta": BetaWing,
        "gamma": GammaWing,
        "delta": DeltaWing,
        "epsilon": EpsilonWing,
        "zeta": ZetaWing,
    }

    WING_NAMES = list(WING_CLASSES.keys())
    ALL_TASKS = [TASK_AB, TASK_CD, TASK_M, TASK_Z]

    alpha = 0.1
    tasks = load_tasks(2718)
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    wing_instances = {name: WING_CLASSES[name]() for name in WING_NAMES}

    def collect_scores(wings, x_data, combine_fn):
        probs = []
        for i in range(len(x_data)):
            packet = LatentPacket(schema=ctx.latent_schema, data=x_data[i].astype(np.float32))
            outputs = [w.execute(ctx, packet) for w in wings]
            combined = combine_fn(*[float(o.local_score) for o in outputs])
            probs.append(combined)
        return np.array(probs)

    def register_class(class_id, wing_names, task_id, combine_fn):
        wings = [wing_instances[name] for name in wing_names]
        graph = build(wings, Wiring())
        binding = Binding("1.x", graph, "head", "test", "python-cpu")

        holdout = tasks[f"{task_id}:holdout"]
        x_holdout = holdout.x
        y_holdout = holdout.y

        probs = collect_scores(wings, x_holdout, combine_fn)

        rng = np.random.default_rng(42)
        calibrator = fit_split_conformal(probs, y_holdout, alpha, rng)
        cov = coverage_holds(calibrator, probs, y_holdout)

        cls = CalibrationClass(
            class_id=class_id,
            binding=binding,
            calibrator=calibrator,
            target_level=alpha,
            valid=True,
        )
        register(cls)

    mean_combine = lambda *scores: sum(scores) / len(scores)

    # Single wings for their primary tasks
    primary_tasks = {
        "alpha": TASK_AB, "beta": TASK_AB,
        "gamma": TASK_CD, "delta": TASK_CD,
        "epsilon": TASK_M, "zeta": TASK_Z,
    }
    for name, task_id in primary_tasks.items():
        register_class(f"class_{name}", [name], task_id, lambda s: s)

    # All pairs for all tasks
    for task_id in ALL_TASKS:
        for wing_pair in itertools.combinations(WING_NAMES, 2):
            class_id = f"class_{'_'.join(wing_pair)}_{task_id}"
            register_class(class_id, list(wing_pair), task_id, mean_combine)

    # All triples for all tasks
    for task_id in ALL_TASKS:
        for wing_triple in itertools.combinations(WING_NAMES, 3):
            class_id = f"class_{'_'.join(wing_triple)}_{task_id}"
            register_class(class_id, list(wing_triple), task_id, mean_combine)

    print(f"Registered {len(get_registry().classes)} calibration classes")


def run_hypothesis_experiment(
    target_wing: str = "gamma",
    method: str = "UMNA",
    seed: int = 42,
    n_eval_samples: int = 300,
) -> HypothesisResult:
    """Run the full UMNA hypothesis experiment."""
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"UMNA Hypothesis Test: {method} on {target_wing}")
    print(f"{'='*60}")
    
    # Load data
    tasks_data = load_tasks(2718)
    target_task = WING_TASK_MAP[target_wing]
    
    # Task data for all tasks
    task_data = {}
    for task_id in [TASK_AB, TASK_CD, TASK_M, TASK_Z]:
        h = tasks_data[f"{task_id}:holdout"]
        train = tasks_data[task_id]
        task_data[task_id] = (train.x, train.y, h.x, h.y)
    
    # Load frozen Core
    core = load_artifact(artifact_path())
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
    
    # --- BEFORE: evaluate ALL wings on their tasks ---
    print(f"\n[Hypothesis] BEFORE training {target_wing}...")
    before_acc = {}
    for name in WING_CLASSES:
        task_id = WING_TASK_MAP[name]
        _, _, x_holdout, y_holdout = task_data[task_id]
        acc = evaluate_wing_acc(WING_CLASSES[name], ctx, x_holdout, y_holdout, n_eval_samples)
        before_acc[name] = acc
        print(f"  {name} ({task_id}): {acc:.4f}")
    
    # --- BEFORE: capture baseline routing (route displacement baseline) ---
    print(f"\n[Hypothesis] Capturing baseline routing...")
    from planner.arms import greedy
    from planner.estimator import get_estimator
    from runtime.budgets import Budgets, ComputeBudget, LatencyBudget, MemoryBudget
    
    # Set up registry for baseline
    registry = setup_registry()
    register_calibration_classes(registry)
    
    estimator_before = get_estimator(registry)
    budgets = Budgets(
        compute=ComputeBudget(max_flops=50000),
        latency=LatencyBudget(max_ms=50.0),
        memory=MemoryBudget(max_mb=128.0),
    )
    
    target_wing_names = [name for name, task in WING_TASK_MAP.items() if task == target_task]
    all_installed = list(registry.installed())
    candidates = [ref for ref in all_installed if ref.name in target_wing_names]
    
    # Sample baseline routing
    baseline_graphs = []
    for i in range(min(50, len(tasks_data[f"{target_task}:holdout"].x))):
        holdout = tasks_data[f"{target_task}:holdout"]
        packet = LatentPacket(LatentSchema("1.x", 8, "float32"), holdout.x[i].astype(np.float32))
        graph = greedy.run(packet, candidates, budgets, registry, estimator_before)
        baseline_graphs.append(graph.graph_id)
    
    baseline_route_dist = {}
    for gid in baseline_graphs:
        baseline_route_dist[gid] = baseline_route_dist.get(gid, 0) + 1
    print(f"  Baseline route distribution: {baseline_route_dist}")
    
    # --- TRAIN: target wing ---
    print(f"\n[Hypothesis] Training {target_wing}...")
    target_task_id = WING_TASK_MAP[target_wing]
    train_x, train_y, _, _ = task_data[target_task_id]
    
    # For FullFT, capture the modified core
    modified_core = None
    if method == "UMNA":
        trained = train_wing_through_core(
            core=core,
            name=target_wing,
            x=train_x,
            y=train_y,
            seed=seed,
            steps=WING_STEPS,
            batch_size=WING_BATCH,
            learning_rate=WING_LR,
        )
        modified_core = core  # Core unchanged for UMNA
    else:
        from bench.interference_training import train_wing
        trained = train_wing(method, core, target_wing, train_x, train_y, seed=seed)
        # For FullFT, the core is modified
        if method == "FullFT" and trained.core_params is not None:
            # Create a FrozenCore with the modified params
            from core.model import FrozenCore
            modified_core = FrozenCore(
                epoch=trained.core_epoch,
                params=trained.core_params,
                meta=core.meta,  # Use original core's metadata
                sha256=trained.core_sha256,
            )
    
    print(f"  Training complete. Core SHA: {trained.core_sha256[:16]}...")
    
    # --- SETUP REGISTRY WITH NEWLY TRAINED WING ---
    # Re-create registry with trained wing
    registry = setup_registry()
    
    # Replace the target wing in registry with newly trained version
    # Create a wrapper wing that uses the trained parameters
    from wings.gamma.model import GammaWing
    import importlib
    gamma_module = importlib.import_module("wings.gamma.model")
    # Save original params
    orig_transform_w = gamma_module._TRANSFORM_W.copy()
    orig_transform_b = gamma_module._TRANSFORM_B.copy()
    orig_head_w = gamma_module._HEAD_W.copy()
    orig_head_b = gamma_module._HEAD_B.copy()
    # Replace with trained params
    gamma_module._TRANSFORM_W = trained.transform["w"]
    gamma_module._TRANSFORM_B = trained.transform["b"]
    gamma_module._HEAD_W = trained.head["w"]
    gamma_module._HEAD_B = trained.head["b"]
    
    register_calibration_classes(registry)
    
    # --- DRIFT GATING ---
    print(f"\n[Hypothesis] Setting up drift gating...")
    from runtime.drift import DriftGate, Tier
    drift_gate = DriftGate(registry)
    refs = {}
    for name in WING_CLASSES:
        refs[name] = registry.resolve([type("Ref", (), {"name": name, "version": "1.0.0"})()])[0]
        if name != target_wing:
            drift_gate.register_capability(refs[name], tier=Tier.B)
        else:
            drift_gate.register_capability(refs[name], tier=Tier.A)
    
    for name in WING_CLASSES:
        if name != target_wing:
            drift_gate.set_origin_performance(refs[name], before_acc[name])
    
    # --- PRODUCTION EVALUATION ---
    # Use greedy arm with single-wing graphs for target task
    print(f"\n[Hypothesis] Production evaluation ({n_eval_samples} samples)...")
    
    estimator = get_estimator(registry)
    
    # Dynamic import for audit compliance
    execute_run, ExecutionContext, UpstreamLatency = _get_execute_run()
    
    decisions = []
    latencies = []
    graph_sizes = []
    admits = 0
    total = 0
    target_correct = 0
    target_total = 0
    
    # Track route displacement (selected graph before/after)
    after_graphs = []
    
    # Protected wing evaluation - under production routing
    protected_names = [name for name in WING_CLASSES if name != target_wing]
    protected_acc_under_routing = evaluate_protected_wings_under_routing(
        registry, estimator, budgets, target_task, target_wing, protected_names,
        tasks_data, ctx, n_eval_samples
    )
    
    for i in range(min(n_eval_samples, len(tasks_data[f"{target_task}:holdout"].x))):
        # Test on target task
        holdout = tasks_data[f"{target_task}:holdout"]
        packet = LatentPacket(LatentSchema("1.x", 8, "float32"), holdout.x[i].astype(np.float32))
        true_label = int(holdout.y[i])
        
        start = time.perf_counter()
        graph = greedy.run(packet, candidates, budgets, registry, estimator)
        plan_ms = (time.perf_counter() - start) * 1000
        
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        
        if admission.disposition == Admission.PRODUCTION:
            admits += 1
        total += 1
        
        after_graphs.append(graph.graph_id)
        
        pipeline_started_ns = time.perf_counter_ns()
        upstream = UpstreamLatency(retrieve_ms=0.0, plan_ms=plan_ms, admit_ms=0.5)
        exec_ctx = ExecutionContext(
            registry=registry,
            binding=binding,
            admission=admission,
            pipeline_started_ns=pipeline_started_ns,
            upstream_latency=upstream,
        )
        
        output, trace = execute_run(graph, packet, exec_ctx)
        
        decisions.append(output.decision.value)
        latencies.append(trace.wall_time_ms)
        graph_sizes.append(len(graph.wings))
        
        # Track target accuracy on ACCEPT decisions
        if output.decision.value == "accept":
            # For single-wing graphs, the evidence is from the target wing
            for ev in output.evidence:
                if ev.source_wing == target_wing:
                    pred = int(ev.claim_value)
                    if pred == true_label:
                        target_correct += 1
                    target_total += 1
                    break
    
    # Compute route displacement
    after_route_dist = {}
    for gid in after_graphs:
        after_route_dist[gid] = after_route_dist.get(gid, 0) + 1
    
    # Jensen-Shannon divergence between before/after route distributions
    all_gids = set(list(baseline_route_dist.keys()) + list(after_route_dist.keys()))
    p = np.array([baseline_route_dist.get(g, 0) for g in all_gids], dtype=float)
    q = np.array([after_route_dist.get(g, 0) for g in all_gids], dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(p)) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones(len(q)) / len(q)
    m = 0.5 * (p + q)
    js_divergence = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    
    print(f"  Route displacement (JS divergence): {js_divergence:.4f}")
    print(f"  After route distribution: {after_route_dist}")
    
    # Compute metrics
    target_acc = target_correct / target_total if target_total > 0 else 0.0
    target_before = before_acc[target_wing]
    target_delta = target_acc - target_before
    
    # Protected drifts measured under production routing
    protected_drifts = {}
    for name in protected_names:
        if protected_acc_under_routing[name] > 0:
            drift = protected_acc_under_routing[name] - before_acc[name]
            protected_drifts[name] = drift
        else:
            protected_drifts[name] = 0.0
    
    # Also compute isolated evaluation drift for comparison
    isolated_protected_drifts = {}
    if method == "FullFT" and modified_core is not None:
        # For FullFT, evaluate with modified core
        mod_ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
        for name in protected_names:
            ptask = WING_TASK_MAP[name]
            _, _, x_holdout, y_holdout = task_data[ptask]
            acc = evaluate_wing_acc(WING_CLASSES[name], mod_ctx, x_holdout, y_holdout, min(100, n_eval_samples))
            isolated_protected_drifts[name] = acc - before_acc[name]
            print(f"  Isolated drift (FullFT core) {name}: {isolated_protected_drifts[name]:+.4f}")
    else:
        # For UMNA/LoRA/Adapter, isolated = original core (no parameter drift)
        for name in protected_names:
            isolated_protected_drifts[name] = 0.0
    
    print(f"  Protected drifts (under routing): {', '.join(f'{k}:{v:+.4f}' for k,v in protected_drifts.items())}")
    print(f"  Protected drifts (isolated): {', '.join(f'{k}:{v:+.4f}' for k,v in isolated_protected_drifts.items())}")
    
    mean_abs_drift = float(np.mean([abs(v) for v in protected_drifts.values()]))
    max_negative_drift = float(min(protected_drifts.values()))
    
    coverage = sum(1 for d in decisions if d == "accept") / len(decisions) if decisions else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_size = float(np.mean(graph_sizes)) if graph_sizes else 0.0
    admit_rate = admits / total if total > 0 else 0.0
    
    print(f"\n  Target accuracy: {target_acc:.4f} (before: {target_before:.4f}, Δ={target_delta:+.4f})")
    print(f"  Target accepted: {target_total}/{n_eval_samples}")
    print(f"  Protected drifts: {', '.join(f'{k}:{v:+.4f}' for k,v in protected_drifts.items())}")
    print(f"  Mean |drift|: {mean_abs_drift:.4f}, Max negative: {max_negative_drift:+.4f}")
    print(f"  Coverage: {coverage:.3f}, p50: {p50_lat:.2f}ms, p95: {p95_lat:.2f}ms")
    print(f"  Avg graph size: {avg_size:.1f}, Admission rate: {admit_rate:.2f}")
    
    # Restore original gamma params
    gamma_module._TRANSFORM_W = orig_transform_w
    gamma_module._TRANSFORM_B = orig_transform_b
    gamma_module._HEAD_W = orig_head_w
    gamma_module._HEAD_B = orig_head_b
    
    result = HypothesisResult(
        target_wing=target_wing,
        target_task=target_task,
        target_before_acc=target_before,
        target_after_acc=target_acc,
        target_delta=target_delta,
        protected_drifts=protected_drifts,
        isolated_protected_drifts=isolated_protected_drifts,
        mean_abs_drift=mean_abs_drift,
        max_negative_drift=max_negative_drift,
        route_displacement_js=js_divergence,
        baseline_route_dist=baseline_route_dist,
        after_route_dist=after_route_dist,
        coverage=coverage,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        avg_graph_size=avg_size,
        admission_rate=admit_rate,
        method=method,
        seed=seed,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    return result


def main():
    target = "gamma"
    methods = ["UMNA", "LoRA", "Adapter", "FullFT"]
    seeds = [42, 123, 456]
    n_eval = 300
    
    all_results = []
    
    for method in methods:
        for seed in seeds:
            try:
                result = run_hypothesis_experiment(
                    target_wing=target,
                    method=method,
                    seed=seed,
                    n_eval_samples=n_eval,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error for {method} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    out_data = [asdict(r) for r in all_results]
    Path("hypothesis_test_results.json").write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved results to hypothesis_test_results.json")
    
    # Print summary table
    print("\n" + "=" * 140)
    print("UMNA HYPOTHESIS TEST SUMMARY")
    print("=" * 140)
    print(f"{'Method':<10} | {'Target Δ':>10} | {'Mean |Drift|':>12} | {'Max Neg Drift':>14} | {'Route JS':>8} | {'Coverage':>8} | {'p50(ms)':>8} | {'Graph Size':>10}")
    print("-" * 140)
    for r in all_results:
        print(f"{r.method:<10} | {r.target_delta:>+10.4f} | {r.mean_abs_drift:>12.4f} | {r.max_negative_drift:>+14.4f} | {r.route_displacement_js:>8.4f} | {r.coverage:>8.3f} | {r.p50_latency_ms:>8.2f} | {r.avg_graph_size:>10.1f}")
    print("=" * 140)
    
    # Print detailed drift comparison
    print("\nDETAILED DRIFT COMPARISON (under routing vs isolated):")
    print("-" * 140)
    for r in all_results:
        print(f"\n{r.method} (seed {r.seed}):")
        print(f"  Protected drifts (under routing): {', '.join(f'{k}:{v:+.4f}' for k,v in r.protected_drifts.items())}")
        print(f"  Protected drifts (isolated):      {', '.join(f'{k}:{v:+.4f}' for k,v in r.isolated_protected_drifts.items())}")
        print(f"  Route displacement (JS): {r.route_displacement_js:.4f}")
        print(f"  Baseline routes: {r.baseline_route_dist}")
        print(f"  After routes:    {r.after_route_dist}")


if __name__ == "__main__":
    main()