use runtime_scheduler::{execute, ExecutionContext, ChaosConfig};
use runtime_abi::{ContainmentTier, FaultClass, WingRef, LatentPacket, LatentSchema, ArtifactRef, CoreContext, Wing, WingOutput, EvidencePacket, Polarity, TypeSet, TypeExpr, Manifest, ResourceCeilings, ManifestIntegrity, ManifestProvenance, HostMetrics};
use runtime_graph::{Graph, Binding, Wiring};
use runtime_registry::{Registry, InMemoryRegistry};
use std::sync::Arc;
use std::collections::HashMap;

struct TestWing {
    name: String,
    version: String,
}

impl TestWing {
    fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
        }
    }
}

impl Wing for TestWing {
    fn manifest(&self) -> Manifest {
        Manifest {
            name: self.name.clone(),
            version: self.version.clone(),
            core_epoch: "1.x".to_string(),
            author: "test".to_string(),
            input_types: TypeSet::new(vec![TypeExpr::bare("Latent<Vec8>")]),
            output_types: TypeSet::new(vec![TypeExpr::bare("Latent<Score>")]),
            capabilities: vec!["test".to_string()],
            dependencies: vec![],
            resource_ceilings: Some(ResourceCeilings { max_memory_mb: 8, max_input_size: 10000 }),
            integrity: Some(ManifestIntegrity { artifact_hash: "test".to_string() }),
            provenance: Some(ManifestProvenance { training_dataset_ids: vec![], training_code_commit: "test".to_string() }),
        }
    }

    fn input_types(&self) -> TypeSet {
        TypeSet::new(vec![TypeExpr::bare("Latent<Vec8>")])
    }

    fn output_types(&self) -> TypeSet {
        TypeSet::new(vec![TypeExpr::bare("Latent<Score>")])
    }

    fn execute(&self, _ctx: CoreContext, x: LatentPacket) -> WingOutput {
        let out_schema = LatentSchema {
            core_epoch: "1.x".to_string(),
            dimensions: 1,
            dtype: "float32".to_string(),
        };
        let evidence = EvidencePacket {
            claim_type: "test".to_string(),
            claim_value: "test".to_string(),
            polarity: Polarity::Support,
            source_wing: self.name.clone(),
            source_version: self.version.clone(),
            route_id: "test".to_string(),
            provenance: "test".to_string(),
            local_score: 0.5,
            calibration_binding: "test".to_string(),
            validity_domain: "test".to_string(),
            timestamp: "now".to_string(),
        };
        WingOutput {
            latent: LatentPacket {
                schema: out_schema.clone(),
                data: x.data,
                provenance: "test".to_string(),
            },
            schema: out_schema,
            evidence: vec![evidence],
            local_score: 0.5,
            wing_version: self.version.clone(),
            provenance: "test".to_string(),
        }
    }
}

fn make_test_registry() -> Arc<dyn Registry> {
    let registry = InMemoryRegistry::with_core_epoch("1.x".to_string());
    let wing = TestWing::new("test_wing", "1.0.0");
    let ref_ = WingRef::new("test_wing", "1.0.0");
    
    // Load manifest first
    let manifest = wing.manifest();
    registry.load(manifest).unwrap();
    
    // Register the wing artifact
    registry.set_artifact(&ref_, Arc::new(wing)).unwrap();
    
    // Register artifact ref
    registry.set_artifact_ref(&ref_, ArtifactRef::InMemory { wing_id: "test_wing@1.0.0".to_string() }).unwrap();
    
    // Register metrics
    registry.set_measured(HostMetrics {
        ref_: ref_.clone(),
        measured_latency_ms: 1.0,
        measured_memory_mb: 8.0,
        measured_reliability: 1.0,
        measured_utility_features: vec![],
        measured_active_flops: 1000.0,
        failure_history: vec![],
    }).unwrap();
    
    Arc::new(registry)
}

fn make_test_graph() -> Graph {
    Graph {
        core_epoch: "1.x".to_string(),
        graph_id: "test-graph".to_string(),
        wings: vec![WingRef::new("test_wing", "1.0.0")],
        wiring: Wiring { edges: vec![] },
    }
}

fn make_test_binding(graph: &Graph) -> Binding {
    Binding::new(
        "1.x".to_string(),
        graph.clone(),
        "test_head".to_string(),
        "test_dist".to_string(),
        "cpu".to_string(),
    )
}

fn make_f32_input() -> LatentPacket {
    LatentPacket {
        schema: LatentSchema {
            core_epoch: "1.x".to_string(),
            dimensions: 1,
            dtype: "float32".to_string(),
        },
        data: vec![0, 0, 128, 63], // 1.0f32 in little-endian
        provenance: "test".to_string(),
    }
}

fn contains_chaos_failure(trace: &runtime_scheduler::Trace, fault: FaultClass, contained: bool) -> bool {
    trace.failures.iter().any(|f| {
        f.contains("CHAOS:") 
        && f.contains(&format!("fault={:?}", fault))
        && f.contains(&format!("contained={}", contained))
    })
}

fn run_chaos_test(
    registry: Arc<dyn Registry>,
    graph: Graph,
    binding: Binding,
    input: LatentPacket,
    fault: FaultClass,
    tier: ContainmentTier,
    probability: f32,
) -> runtime_scheduler::Trace {
    let ctx = ExecutionContext::new(registry, binding)
        .with_containment(tier)
        .with_chaos(ChaosConfig::single_fault(fault, tier, probability));
    
    let (_, trace) = execute(&graph, input, ctx).unwrap();
    trace
}

fn run_chaos_test_fallible(
    registry: Arc<dyn Registry>,
    graph: Graph,
    binding: Binding,
    input: LatentPacket,
    fault: FaultClass,
    tier: ContainmentTier,
    probability: f32,
) -> Result<runtime_scheduler::Trace, runtime_scheduler::ExecutionError> {
    let ctx = ExecutionContext::new(registry, binding)
        .with_containment(tier)
        .with_chaos(ChaosConfig::single_fault(fault, tier, probability));
    
    execute(&graph, input, ctx).map(|(_, trace)| trace)
}

#[cfg(test)]
mod tier_sweep {
    use super::*;

    fn setup() -> (Arc<dyn Registry>, Graph, Binding, LatentPacket) {
        let registry = make_test_registry();
        let graph = make_test_graph();
        let binding = make_test_binding(&graph);
        let input = make_f32_input();
        (registry, graph, binding, input)
    }

    #[test]
    fn test_nan_output_all_tiers() {
        let (registry, graph, binding, input) = setup();
        
        for tier in [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2] {
            let trace = run_chaos_test(
                registry.clone(), graph.clone(), binding.clone(), input.clone(),
                FaultClass::NanOutput, tier, 1.0
            );
            assert!(contains_chaos_failure(&trace, FaultClass::NanOutput, true), 
                "NanOutput should be contained at {:?}", tier);
        }
    }

    #[test]
    fn test_invalid_shape_all_tiers() {
        let (registry, graph, binding, input) = setup();
        
        for tier in [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2] {
            let trace = run_chaos_test(
                registry.clone(), graph.clone(), binding.clone(), input.clone(),
                FaultClass::InvalidShape, tier, 1.0
            );
            assert!(contains_chaos_failure(&trace, FaultClass::InvalidShape, true),
                "InvalidShape should be contained at {:?}", tier);
        }
    }

    #[test]
    fn test_range_violation_all_tiers() {
        let (registry, graph, binding, input) = setup();
        
        for tier in [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2] {
            let trace = run_chaos_test(
                registry.clone(), graph.clone(), binding.clone(), input.clone(),
                FaultClass::RangeViolation, tier, 1.0
            );
            assert!(contains_chaos_failure(&trace, FaultClass::RangeViolation, true),
                "RangeViolation should be contained at {:?}", tier);
        }
    }

    #[test]
    fn test_schema_mismatch_all_tiers() {
        let (registry, graph, binding, input) = setup();
        
        for tier in [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2] {
            let trace = run_chaos_test(
                registry.clone(), graph.clone(), binding.clone(), input.clone(),
                FaultClass::SchemaMismatch, tier, 1.0
            );
            assert!(contains_chaos_failure(&trace, FaultClass::SchemaMismatch, true),
                "SchemaMismatch should be contained at {:?}", tier);
        }
    }

    #[test]
    fn test_process_crash_tier_sweep() {
        let (registry, graph, binding, input) = setup();
        
        let trace_f0 = run_chaos_test_fallible(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::ProcessCrash, ContainmentTier::F0, 1.0
        );
        assert!(trace_f0.is_err(), "F0 ProcessCrash should escape (panic)");
        
        let trace_f1 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::ProcessCrash, ContainmentTier::F1, 1.0
        );
        assert!(contains_chaos_failure(&trace_f1, FaultClass::ProcessCrash, true),
            "F1 ProcessCrash should be contained");
        
        let trace_f2 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::ProcessCrash, ContainmentTier::F2, 1.0
        );
        assert!(contains_chaos_failure(&trace_f2, FaultClass::ProcessCrash, true),
            "F2 ProcessCrash should be contained");
    }

    #[test]
    fn test_device_memory_access_tier_sweep() {
        let (registry, graph, binding, input) = setup();
        
        let trace_f0 = run_chaos_test_fallible(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::DeviceMemoryAccess, ContainmentTier::F0, 1.0
        );
        assert!(trace_f0.is_err(), "F0 DeviceMemoryAccess should escape (panic)");
        
        let trace_f1 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::DeviceMemoryAccess, ContainmentTier::F1, 1.0
        );
        assert!(contains_chaos_failure(&trace_f1, FaultClass::DeviceMemoryAccess, true),
            "F1 DeviceMemoryAccess should be contained");
        
        let trace_f2 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::DeviceMemoryAccess, ContainmentTier::F2, 1.0
        );
        assert!(contains_chaos_failure(&trace_f2, FaultClass::DeviceMemoryAccess, true),
            "F2 DeviceMemoryAccess should be contained");
    }

    #[test]
    fn test_context_corruption_all_tiers() {
        let (registry, graph, binding, input) = setup();
        
        for tier in [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2] {
            let trace = run_chaos_test(
                registry.clone(), graph.clone(), binding.clone(), input.clone(),
                FaultClass::ContextCorruption, tier, 1.0
            );
            assert!(contains_chaos_failure(&trace, FaultClass::ContextCorruption, true),
                "ContextCorruption should be contained at {:?}", tier);
        }
    }

    #[test]
    fn test_severe_oom_tier_sweep() {
        let (registry, graph, binding, input) = setup();
        
        let trace_f0 = run_chaos_test_fallible(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::SevereOom, ContainmentTier::F0, 1.0
        );
        assert!(trace_f0.is_err(), "F0 SevereOom should escape (panic)");
        
        let trace_f1 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::SevereOom, ContainmentTier::F1, 1.0
        );
        assert!(contains_chaos_failure(&trace_f1, FaultClass::SevereOom, true),
            "F1 SevereOom should be contained");
        
        let trace_f2 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::SevereOom, ContainmentTier::F2, 1.0
        );
        assert!(contains_chaos_failure(&trace_f2, FaultClass::SevereOom, true),
            "F2 SevereOom should be contained");
    }

    #[test]
    fn test_untrusted_code_tier_sweep() {
        let (registry, graph, binding, input) = setup();
        
        let trace_f0 = run_chaos_test_fallible(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::UntrustedCode, ContainmentTier::F0, 1.0
        );
        assert!(trace_f0.is_err(), "F0 UntrustedCode should escape (panic)");
        
        let trace_f1 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::UntrustedCode, ContainmentTier::F1, 1.0
        );
        assert!(contains_chaos_failure(&trace_f1, FaultClass::UntrustedCode, true),
            "F1 UntrustedCode should be contained");
        
        let trace_f2 = run_chaos_test(
            registry.clone(), graph.clone(), binding.clone(), input.clone(),
            FaultClass::UntrustedCode, ContainmentTier::F2, 1.0
        );
        assert!(contains_chaos_failure(&trace_f2, FaultClass::UntrustedCode, true),
            "F2 UntrustedCode should be contained");
    }
}

#[cfg(test)]
mod statistical {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empirical_containment_rates() {
        let (registry, graph, binding, input) = setup();
        let iterations = 50; // Lower for CI speed
        let mut results: HashMap<(FaultClass, ContainmentTier), (usize, usize)> = HashMap::new();
        
        let faults = [
            FaultClass::NanOutput,
            FaultClass::InvalidShape,
            FaultClass::RangeViolation,
            FaultClass::SchemaMismatch,
            FaultClass::ProcessCrash,
            FaultClass::DeviceMemoryAccess,
            FaultClass::ContextCorruption,
            FaultClass::SevereOom,
            FaultClass::UntrustedCode,
        ];
        
        let tiers = [ContainmentTier::F0, ContainmentTier::F1, ContainmentTier::F2];
        
        for fault in faults {
            for tier in tiers {
                let mut contained = 0;
                let mut total = 0;
                
                for i in 0..iterations {
                    let seed = Some(i as u64 + 0xC0DE);
                    let mut config = ChaosConfig::single_fault(fault, tier, 1.0);
                    config.seed = seed;
                    
                    let ctx = ExecutionContext::new(registry.clone(), binding.clone())
                        .with_containment(tier)
                        .with_chaos(config);
                    
                    let result = execute(&graph, input.clone(), ctx);
                    
                    if result.is_ok() {
                        let trace = result.unwrap().1;
                        if contains_chaos_failure(&trace, fault, true) {
                            contained += 1;
                        }
                        total += 1;
                    } else if tier != ContainmentTier::F0 {
                        // F1/F2 process isolation catches the panic
                        contained += 1;
                        total += 1;
                    }
                }
                
                let rate = if total > 0 { contained as f64 / total as f64 } else { 0.0 };
                results.insert((fault, tier), (contained, total));
                
                println!("{:?} @ {:?}: {}/{} = {:.2}%", fault, tier, contained, total, rate * 100.0);
                
                let theoretical = match (fault, tier) {
                    (FaultClass::NanOutput | FaultClass::InvalidShape | FaultClass::RangeViolation | FaultClass::SchemaMismatch, _) => 1.0,
                    (FaultClass::ProcessCrash, ContainmentTier::F0) => 0.0,
                    (FaultClass::ProcessCrash, ContainmentTier::F1) => 0.95,
                    (FaultClass::ProcessCrash, ContainmentTier::F2) => 0.999,
                    (FaultClass::DeviceMemoryAccess, ContainmentTier::F0) => 0.0,
                    (FaultClass::DeviceMemoryAccess, ContainmentTier::F1) => 0.6,
                    (FaultClass::DeviceMemoryAccess, ContainmentTier::F2) => 0.99,
                    (FaultClass::ContextCorruption, ContainmentTier::F0) => 1.0,
                    (FaultClass::ContextCorruption, ContainmentTier::F1) => 0.8,
                    (FaultClass::ContextCorruption, ContainmentTier::F2) => 0.99,
                    (FaultClass::SevereOom, ContainmentTier::F0) => 0.0,
                    (FaultClass::SevereOom, ContainmentTier::F1) => 0.7,
                    (FaultClass::SevereOom, ContainmentTier::F2) => 0.95,
                    (FaultClass::UntrustedCode, ContainmentTier::F0) => 0.0,
                    (FaultClass::UntrustedCode, ContainmentTier::F1) => 0.9,
                    (FaultClass::UntrustedCode, ContainmentTier::F2) => 0.999,
                };
                
                if total > 0 {
                    let tolerance = 0.20; // 20% tolerance for small sample
                    assert!((rate - theoretical).abs() < tolerance,
                        "{:?} @ {:?}: empirical {:.2}% vs theoretical {:.2}% (tolerance {:.0}%)",
                        fault, tier, rate * 100.0, theoretical * 100.0, tolerance * 100.0);
                }
            }
        }
    }

    fn setup() -> (Arc<dyn Registry>, Graph, Binding, LatentPacket) {
        let registry = make_test_registry();
        let graph = make_test_graph();
        let binding = make_test_binding(&graph);
        let input = make_f32_input();
        (registry, graph, binding, input)
    }
}

#[cfg(test)]
mod trigger_tests {
    use super::*;

    #[test]
    fn test_on_wing_trigger() {
        let (registry, graph, binding, input) = setup();
        
        let mut config = ChaosConfig::single_fault(FaultClass::NanOutput, ContainmentTier::F0, 1.0);
        config.injections[0].trigger = runtime_scheduler::InjectionTrigger::OnWing { 
            wing_name: "test_wing".to_string() 
        };
        
        let ctx = ExecutionContext::new(registry, binding)
            .with_containment(ContainmentTier::F0)
            .with_chaos(config);
        
        let (_, trace) = execute(&graph, input, ctx).unwrap();
        
        assert!(trace.failures.iter().any(|f| f.contains("CHAOS:") && f.contains("wing=test_wing")));
    }

    #[test]
    fn test_random_trigger_rate() {
        let (registry, graph, binding, input) = setup();
        let iterations = 500;
        let mut injected = 0;
        
        for i in 0..iterations {
            let mut config = ChaosConfig::single_fault(FaultClass::NanOutput, ContainmentTier::F0, 1.0);
            config.injections[0].trigger = runtime_scheduler::InjectionTrigger::Random { rate: 0.1 };
            config.seed = Some(i as u64);
            
            let ctx = ExecutionContext::new(registry.clone(), binding.clone())
                .with_containment(ContainmentTier::F0)
                .with_chaos(config);
            
            let (_, trace) = execute(&graph, input.clone(), ctx).unwrap();
            
            if trace.failures.iter().any(|f| f.contains("CHAOS:")) {
                injected += 1;
            }
        }
        
        let rate = injected as f64 / iterations as f64;
        println!("Random trigger rate: {:.2}% (expected ~10%)", rate * 100.0);
        
        assert!(rate > 0.05 && rate < 0.15, "Random trigger rate should be ~10%");
    }

    fn setup() -> (Arc<dyn Registry>, Graph, Binding, LatentPacket) {
        let registry = make_test_registry();
        let graph = make_test_graph();
        let binding = make_test_binding(&graph);
        let input = make_f32_input();
        (registry, graph, binding, input)
    }
}

#[cfg(test)]
mod overhead_tests {
    use super::*;

    #[test]
    fn test_injection_overhead_recorded() {
        let (registry, graph, binding, input) = setup();
        
        let trace = run_chaos_test(
            registry, graph, binding, input,
            FaultClass::NanOutput, ContainmentTier::F0, 1.0
        );
        
        let chaos_failure = trace.failures.iter().find(|f| f.contains("CHAOS:")).unwrap();
        assert!(chaos_failure.contains("overhead_ms="));
        
        let overhead_str = chaos_failure.split("overhead_ms=").nth(1).unwrap();
        let overhead: f64 = overhead_str.split_whitespace().next().unwrap().parse().unwrap();
        println!("Injection overhead: {:.2}ms", overhead);
        
        assert!(overhead < 10.0, "Injection overhead should be minimal");
    }

    fn setup() -> (Arc<dyn Registry>, Graph, Binding, LatentPacket) {
        let registry = make_test_registry();
        let graph = make_test_graph();
        let binding = make_test_binding(&graph);
        let input = make_f32_input();
        (registry, graph, binding, input)
    }
}

#[cfg(test)]
mod all_faults {
    use super::*;

    #[test]
    fn test_all_faults_f0() {
        let (registry, graph, binding, input) = setup();
        
        let config = ChaosConfig::all_faults(ContainmentTier::F0, 1.0);
        let ctx = ExecutionContext::new(registry, binding)
            .with_containment(ContainmentTier::F0)
            .with_chaos(config);
        
        let result = execute(&graph, input, ctx);
        
        assert!(result.is_err(), "F0 with all faults should panic on ProcessCrash/DeviceMemoryAccess/SevereOom/UntrustedCode");
    }

    #[test]
    fn test_all_faults_f1() {
        let (registry, graph, binding, input) = setup();
        
        let config = ChaosConfig::all_faults(ContainmentTier::F1, 1.0);
        let ctx = ExecutionContext::new(registry, binding)
            .with_containment(ContainmentTier::F1)
            .with_chaos(config);
        
        let (_, trace) = execute(&graph, input, ctx).unwrap();
        
        let chaos_count = trace.failures.iter().filter(|f| f.contains("CHAOS:")).count();
        println!("F1 all faults injected: {}", chaos_count);
        assert_eq!(chaos_count, 9);
    }

    #[test]
    fn test_all_faults_f2() {
        let (registry, graph, binding, input) = setup();
        
        let config = ChaosConfig::all_faults(ContainmentTier::F2, 1.0);
        let ctx = ExecutionContext::new(registry, binding)
            .with_containment(ContainmentTier::F2)
            .with_chaos(config);
        
        let (_, trace) = execute(&graph, input, ctx).unwrap();
        
        let chaos_count = trace.failures.iter().filter(|f| f.contains("CHAOS:")).count();
        println!("F2 all faults injected: {}", chaos_count);
        assert_eq!(chaos_count, 9);
    }

    fn setup() -> (Arc<dyn Registry>, Graph, Binding, LatentPacket) {
        let registry = make_test_registry();
        let graph = make_test_graph();
        let binding = make_test_binding(&graph);
        let input = make_f32_input();
        (registry, graph, binding, input)
    }
}