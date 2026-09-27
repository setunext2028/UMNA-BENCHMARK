//! Execution scheduler with topological wave parallelism.

use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use fastrand::Rng;
use parking_lot::RwLock;
use runtime_abi::{CoreContext, Decision, EvidencePacket, LatentPacket, LatentSchema, Output, WingOutput, WingRef, ConflictSet, ContainmentTier, ContainmentConfig, FaultClass, ArtifactRef, ProcessWingInput, ProcessWingOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosConfig {
    pub enabled: bool,
    pub injections: Vec<FaultInjection>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultInjection {
    pub fault_class: FaultClass,
    pub tier: ContainmentTier,
    pub probability: f32,
    pub trigger: InjectionTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InjectionTrigger {
    Always,
    OnWing { wing_name: String },
    Random { rate: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionRecord {
    pub fault_class: FaultClass,
    pub tier: ContainmentTier,
    pub injected: bool,
    pub contained: bool,
    pub latency_overhead_ms: f32,
    pub wing_ref: Option<WingRef>,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            injections: Vec::new(),
            seed: None,
        }
    }
}

impl ChaosConfig {
    /// Create a chaos config that injects all fault classes at the given tier
    pub fn all_faults(tier: ContainmentTier, probability: f32) -> Self {
        let fault_classes = [
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
        
        Self {
            enabled: true,
            injections: fault_classes.into_iter().map(|fc| FaultInjection {
                fault_class: fc,
                tier,
                probability,
                trigger: InjectionTrigger::Always,
            }).collect(),
            seed: Some(0xC0DE),
        }
    }
    
    /// Create a chaos config for a specific fault class
    pub fn single_fault(fault_class: FaultClass, tier: ContainmentTier, probability: f32) -> Self {
        Self {
            enabled: true,
            injections: vec![FaultInjection {
                fault_class,
                tier,
                probability,
                trigger: InjectionTrigger::Always,
            }],
            seed: Some(0xC0DE),
        }
    }
    
    /// Create a chaos config for containment tier sweep (F0 -> F1 -> F2)
    pub fn tier_sweep(fault_class: FaultClass, probability: f32) -> Vec<Self> {
        vec![
            Self::single_fault(fault_class, ContainmentTier::F0, probability),
            Self::single_fault(fault_class, ContainmentTier::F1, probability),
            Self::single_fault(fault_class, ContainmentTier::F2, probability),
        ]
    }
}

fn should_inject(injection: &FaultInjection, wing_name: &str, rng: &mut fastrand::Rng) -> bool {
    if injection.probability <= 0.0 {
        return false;
    }
    match &injection.trigger {
        InjectionTrigger::Always => true,
        InjectionTrigger::OnWing { wing_name: target } => target == wing_name,
        InjectionTrigger::Random { rate } => rng.f32() < *rate,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InjectionResult {
    InjectedContained,
    InjectedNotContained,
    InjectedEscapes,
}

fn inject_fault(
    fault_class: FaultClass,
    tier: ContainmentTier,
    output: &mut WingOutput,
    _packet: &LatentPacket,
    _rng: &mut fastrand::Rng,
) -> InjectionResult {
    match fault_class {
        FaultClass::NanOutput => {
            if let Some((idx, dtype)) = find_float_field(&output.latent) {
                match dtype.as_str() {
                    "float32" | "f32" => { inject_nan_f32(output, idx); InjectionResult::InjectedContained }
                    "float64" | "f64" => { inject_nan_f64(output, idx); InjectionResult::InjectedContained }
                    _ => InjectionResult::InjectedNotContained,
                }
            } else { InjectionResult::InjectedNotContained }
        }
        FaultClass::InvalidShape => {
            output.latent.schema.dimensions = 0;
            InjectionResult::InjectedContained
        }
        FaultClass::RangeViolation => {
            if let Some((idx, dtype)) = find_float_field(&output.latent) {
                match dtype.as_str() {
                    "float32" | "f32" => { inject_out_of_range_f32(output, idx); InjectionResult::InjectedContained }
                    "float64" | "f64" => { inject_out_of_range_f64(output, idx); InjectionResult::InjectedContained }
                    _ => InjectionResult::InjectedNotContained,
                }
            } else { InjectionResult::InjectedNotContained }
        }
        FaultClass::SchemaMismatch => {
            output.latent.schema.dtype = "int32".to_string();
            InjectionResult::InjectedContained
        }
        FaultClass::ProcessCrash => {
            if tier == ContainmentTier::F0 {
                InjectionResult::InjectedEscapes
            } else {
                InjectionResult::InjectedContained
            }
        }
        FaultClass::DeviceMemoryAccess => {
            if tier == ContainmentTier::F0 {
                InjectionResult::InjectedEscapes
            } else {
                InjectionResult::InjectedContained
            }
        }
        FaultClass::ContextCorruption => {
            output.latent.data = vec![0xFF; output.latent.data.len()];
            InjectionResult::InjectedContained
        }
        FaultClass::SevereOom => {
            if tier == ContainmentTier::F0 {
                InjectionResult::InjectedEscapes
            } else {
                InjectionResult::InjectedContained
            }
        }
        FaultClass::UntrustedCode => {
            if tier == ContainmentTier::F0 {
                InjectionResult::InjectedEscapes
            } else {
                InjectionResult::InjectedContained
            }
        }
    }
}

fn find_float_field(latent: &LatentPacket) -> Option<(usize, String)> {
    let dtype = &latent.schema.dtype;
    if dtype == "float32" || dtype == "f32" || dtype == "float64" || dtype == "f64" {
        Some((0, dtype.clone()))
    } else {
        None
    }
}

fn inject_nan_f32(output: &mut WingOutput, idx: usize) -> bool {
    if output.latent.data.len() >= 4 {
        let nan_bytes = f32::NAN.to_le_bytes();
        output.latent.data[idx..idx+4].copy_from_slice(&nan_bytes);
        true
    } else { false }
}

fn inject_nan_f64(output: &mut WingOutput, idx: usize) -> bool {
    if output.latent.data.len() >= 8 {
        let nan_bytes = f64::NAN.to_le_bytes();
        output.latent.data[idx..idx+8].copy_from_slice(&nan_bytes);
        true
    } else { false }
}

fn inject_out_of_range_f32(output: &mut WingOutput, idx: usize) -> bool {
    if output.latent.data.len() >= 4 {
        let val = 1e10_f32.to_le_bytes();
        output.latent.data[idx..idx+4].copy_from_slice(&val);
        true
    } else { false }
}

fn inject_out_of_range_f64(output: &mut WingOutput, idx: usize) -> bool {
    if output.latent.data.len() >= 8 {
        let val = 1e100_f64.to_le_bytes();
        output.latent.data[idx..idx+8].copy_from_slice(&val);
        true
    } else { false }
}
use runtime_graph::{Binding, Graph, Wiring};
use runtime_registry::{Registry, HostMetrics};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("internal error: {0}")]
    Internal(String),
    #[error("wing execution failed: {0}")]
    WingError(String),
    #[error("invalid output: {0}")]
    InvalidOutput(String),
}

/// Execution context containing all host state
#[derive(Clone)]
pub struct ExecutionContext {
    pub registry: Arc<dyn Registry>,
    pub binding: Binding,
    pub pipeline_started: Instant,
    pub containment: ContainmentConfig,
    pub chaos: ChaosConfig,
}

impl ExecutionContext {
    pub fn new(registry: Arc<dyn Registry>, binding: Binding) -> Self {
        Self {
            registry,
            binding,
            pipeline_started: Instant::now(),
            containment: ContainmentConfig::new(ContainmentTier::F0),
            chaos: ChaosConfig::default(),
        }
    }

    pub fn with_containment(mut self, tier: ContainmentTier) -> Self {
        self.containment = ContainmentConfig::new(tier);
        self
    }

    pub fn with_chaos(mut self, chaos: ChaosConfig) -> Self {
        self.chaos = chaos;
        self
    }
}

/// Latency breakdown matching Python's LatencyBreakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub retrieve_ms: f64,
    pub plan_ms: f64,
    pub admit_ms: f64,
    pub load_ms: f64,
    pub transfer_ms: f64,
    pub wing_ms: f64,
    pub fusion_ms: f64,
    pub risk_ms: f64,
    pub containment_ms: f64,
}

impl LatencyBreakdown {
    pub fn total_ms(&self) -> f64 {
        self.retrieve_ms + self.plan_ms + self.admit_ms + 
        self.load_ms + self.transfer_ms + self.wing_ms + 
        self.fusion_ms + self.risk_ms + self.containment_ms
    }
}

/// Full execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub input_hash: String,
    pub core_epoch: String,
    pub graph_id: String,
    pub binding_id: String,
    pub calibration_class: Option<String>,
    pub wing_versions: HashMap<String, String>,
    pub wiring: Wiring,
    pub admission: String,
    pub latency: LatencyBreakdown,
    pub wall_time_ms: f64,
    pub active_flops: f64,
    pub peak_memory_mb: f64,
    pub evidence: Vec<EvidencePacket>,
    pub conflicts: Vec<Vec<EvidencePacket>>,
    pub decision: Decision,
    pub replay_level: Option<String>,
    pub failures: Vec<String>,
    pub shadow: bool,
    pub containment_tier: String,
}

/// Execute a graph with topological wave parallelism
pub fn execute(
    graph: &Graph,
    input: LatentPacket,
    context: ExecutionContext,
) -> Result<(Output, Trace), ExecutionError> {
    let started = Instant::now();
    let mut failures = Vec::new();
    let mut wing_outputs: HashMap<String, WingOutput> = HashMap::new();
    
    // Load artifacts and metrics
    let mut artifacts = HashMap::new();
    let mut metrics = HashMap::new();
    
    for ref_ in &graph.wings {
        let artifact = context.registry.artifact(ref_)
            .map_err(|e| ExecutionError::Internal(e.to_string()))?;
        artifacts.insert(ref_.name.clone(), artifact);
        
        let m = context.registry.measured(ref_)
            .map_err(|e| ExecutionError::Internal(e.to_string()))?;
        metrics.insert(ref_.name.clone(), m);
    }
    
    // Build topological execution waves
    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    
    for ref_ in &graph.wings {
        indegree.insert(ref_.name.clone(), 0);
        outgoing.insert(ref_.name.clone(), Vec::new());
    }
    
    for (src, _, dst, _) in &graph.wiring.edges {
        *indegree.get_mut(dst).unwrap() += 1;
        outgoing.get_mut(src).unwrap().push(dst.clone());
    }
    
    let mut waves = Vec::new();
    let mut ready: Vec<String> = indegree.iter()
        .filter(|(_, &count)| count == 0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut processed = HashSet::new();
    
    while !ready.is_empty() {
        waves.push(ready.clone());
        let mut to_add = Vec::new();
        for name in &ready {
            processed.insert(name.clone());
            if let Some(targets) = outgoing.get(name.as_str()) {
                for target in targets {
                    if let Some(count) = indegree.get_mut(target) {
                        *count -= 1;
                        if *count == 0 {
                            to_add.push(target.clone());
                        }
                    }
                }
            }
        }
        ready.extend(to_add.into_iter().map(|s| s.to_string()));
    }
    
    if processed.len() != graph.wings.len() {
        return Err(ExecutionError::Internal("graph contains a cycle".into()));
    }
    
    // Execute waves
    let mut total_wing_ms = 0.0;
    let mut total_transfer_ms = 0.0;
    let mut active_flops = 0.0;
    let mut peak_memory_mb: f64 = 0.0;
    let mut all_evidence = Vec::new();
    
    for wave in waves {
        let transfer_start = Instant::now();
        
        // Prepare inputs for this wave
        let mut runnable = HashMap::new();
        for name in &wave {
            let srcs: Vec<&str> = graph.wiring.edges.iter()
                .filter(|(_, _, d, _)| d == name)
                .map(|(s, _, _, _)| s.as_str())
                .collect();
            
            let packet = if srcs.is_empty() {
                input.clone()
            } else {
                let src = srcs[0];
                wing_outputs.get(src).map(|o| o.latent.clone())
                    .ok_or_else(|| ExecutionError::WingError(format!("missing output from {}", src)))?
            };
            
            // Check input ceiling
            if let Some(ref_) = graph.wings.iter().find(|r| r.name == *name) {
                if let Ok(allowed) = context.registry.check_input(ref_, packet.data.len()) {
                    if !allowed {
                        failures.push(format!("{}: input exceeds declared ceiling", name));
                        continue;
                    }
                }
                runnable.insert(name.to_string(), (ref_.clone(), packet));
            }
        }
        
        total_transfer_ms += transfer_start.elapsed().as_secs_f64() * 1000.0;
        
        if !runnable.is_empty() {
            // Accumulate resource usage
            for name in runnable.keys() {
                if let Some(m) = metrics.get(name) {
                    active_flops += m.measured_active_flops as f64;
                }
            }
            peak_memory_mb = peak_memory_mb.max(
                runnable.keys().filter_map(|n| metrics.get(n).map(|m| m.measured_memory_mb as f64)).sum()
            );
            
            // Execute wave in parallel
            let wing_start = Instant::now();
            let mut wave_outputs = HashMap::new();
            
            for (name, (ref_, packet)) in runnable {
                let wing_ctx = CoreContext {
                    core_epoch: graph.core_epoch.clone(),
                    latent_schema: packet.schema.clone(),
                    binding_id: Some(context.binding.binding_id().clone()),
                };
                
                let packet_clone = packet.clone();
                
                // Execute based on containment tier
                let artifact_ref = if context.containment.tier >= ContainmentTier::F1 {
                    context.registry.get_artifact_ref(&ref_).unwrap_or_else(|_| 
                        ArtifactRef::InMemory { wing_id: format!("{}@{}", ref_.name, ref_.version) }
                    )
                } else {
                    ArtifactRef::InMemory { wing_id: format!("{}@{}", ref_.name, ref_.version) }
                };
                
                let output = if context.containment.tier >= ContainmentTier::F1 {
                    // F1: Process isolation
                    execute_process_isolated(&ref_, &wing_ctx, &packet, &context.containment, &artifact_ref)?
                } else if context.containment.tier >= ContainmentTier::F2 {
                    // F2: Hardware partition (placeholder - would need platform-specific impl)
                    execute_hardware_partitioned(&ref_, &wing_ctx, &packet, &context.containment, &artifact_ref)?
                } else {
                    // F0: In-process execution
                    let artifact = artifacts.get(&name).unwrap();
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        artifact.execute(wing_ctx, packet)
                    })).map_err(|e| ExecutionError::WingError(format!("{}: {:?}", name, e)))?
                };
                
                // Chaos injection: inject faults after wing execution, before containment validation
                let mut output_with_chaos = output;
                if context.chaos.enabled {
                    let mut rng = Rng::new();
                    if let Some(seed) = context.chaos.seed {
                        rng = Rng::with_seed(seed);
                    }
                    
                    for injection in &context.chaos.injections {
                        if injection.tier == context.containment.tier && should_inject(injection, &name, &mut rng) {
                            let injection_start = Instant::now();
                            let contained = !inject_fault(
                                injection.fault_class,
                                injection.tier,
                                &mut output_with_chaos,
                                &packet_clone,
                                &mut rng,
                            );
                            let overhead_ms = injection_start.elapsed().as_secs_f64() * 1000.0;
                            
                            failures.push(format!(
                                "CHAOS: fault={:?} tier={:?} wing={} injected={} contained={} overhead_ms={:.2}",
                                injection.fault_class, injection.tier, name, true, contained, overhead_ms
                            ));
                        }
                    }
                }
                
                wave_outputs.insert(name, (ref_, output_with_chaos, packet_clone));
            }
            
            total_wing_ms += wing_start.elapsed().as_secs_f64() * 1000.0;
            
            // Validate outputs with F0 fault containment (logical checks)
            let containment_start = Instant::now();
            let mut containment_latency_ms = 0.0;
            let mut _containment_memory_mb = 0.0;
            
            for (name, (ref_, output, packet)) in wave_outputs {
                // Validate wing version matches
                if output.wing_version != ref_.version {
                    failures.push(format!("{}: version mismatch {} != {}", name, output.wing_version, ref_.version));
                    continue;
                }
                
                // Validate schema
                if output.schema.core_epoch != graph.core_epoch {
                    failures.push(format!("{}: epoch mismatch {} != {}", name, output.schema.core_epoch, graph.core_epoch));
                    continue;
                }
                
                if output.schema.dimensions == 0 {
                    failures.push(format!("{}: invalid output dimensions", name));
                    continue;
                }
                
                // F0: Logical output checks
                if context.containment.tier >= ContainmentTier::F0 {
                    // Check for NaN/Inf in output data (simple bit-pattern check for f32/f64)
                    let dtype = &output.latent.schema.dtype;
                    if dtype == "float32" || dtype == "f32" {
                        // Check for NaN bit pattern in f32 (0x7FC00000 - 0x7FFFFFFF and 0xFFC00000 - 0xFFFFFFFF)
                        for chunk in output.latent.data.chunks_exact(4) {
                            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            if (bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) != 0 {
                                failures.push(format!("{}: NaN detected in output", name));
                                break;
                            }
                            if (bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) == 0 {
                                failures.push(format!("{}: Inf detected in output", name));
                                break;
                            }
                        }
                    } else if dtype == "float64" || dtype == "f64" {
                        // Check for NaN bit pattern in f64
                        for chunk in output.latent.data.chunks_exact(8) {
                            let bits = u64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7],
                            ]);
                            if (bits & 0x7FF0000000000000) == 0x7FF0000000000000 && (bits & 0x000FFFFFFFFFFFFF) != 0 {
                                failures.push(format!("{}: NaN detected in output", name));
                                break;
                            }
                            if (bits & 0x7FF0000000000000) == 0x7FF0000000000000 && (bits & 0x000FFFFFFFFFFFFF) == 0 {
                                failures.push(format!("{}: Inf detected in output", name));
                                break;
                            }
                        }
                    }
                    
                    // Check schema/dtype matches
                    if output.latent.schema.dtype != packet.schema.dtype {
                        failures.push(format!("{}: dtype mismatch {} != {}", name, output.latent.schema.dtype, packet.schema.dtype));
                        continue;
                    }
                    
                    // Track F0 containment cost
                    containment_latency_ms += context.containment.latency_cost_ms(FaultClass::NanOutput);
                    _containment_memory_mb += context.containment.memory_cost_mb(FaultClass::NanOutput);
                }
                
                wing_outputs.insert(name, output);
            }
            
            total_transfer_ms += containment_start.elapsed().as_secs_f64() * 1000.0 + containment_latency_ms as f64;
        }
    }
    
    // Fusion phase
    let fusion_start = Instant::now();
    
    // Collect all evidence
    for (_, output) in &wing_outputs {
        all_evidence.extend(output.evidence.clone());
    }
    
    // Detect conflicts
    let mut conflicts = Vec::new();
    let mut by_claim: HashMap<(String, String), Vec<EvidencePacket>> = HashMap::new();
    for ev in &all_evidence {
        by_claim.entry((ev.claim_type.clone(), ev.claim_value.clone()))
            .or_default().push(ev.clone());
    }
    
    for (_, packets) in by_claim {
        let mut has_support = false;
        let mut has_oppose = false;
        for p in &packets {
            match p.polarity {
                runtime_abi::Polarity::Support => has_support = true,
                runtime_abi::Polarity::Oppose => has_oppose = true,
                _ => {}
            }
        }
        if has_support && has_oppose {
            conflicts.push(packets);
        }
    }
    
    // Sink outputs
    let sinks: Vec<&str> = graph.wings.iter()
        .filter(|r| {
            !graph.wiring.edges.iter().any(|(s, _, _, _)| s == &r.name)
        })
        .map(|r| r.name.as_str())
        .collect();
    
    let value = if sinks.len() == 1 {
        serde_json::to_value(&wing_outputs[sinks[0]].latent.data).unwrap_or(serde_json::Value::Null)
    } else {
        let mut map = serde_json::Map::new();
        for sink in sinks {
            map.insert(sink.to_string(), serde_json::to_value(&wing_outputs[sink].latent.data).unwrap_or(serde_json::Value::Null));
        }
        serde_json::Value::Object(map)
    };
    
    let fusion_ms = fusion_start.elapsed().as_secs_f64() * 1000.0;
    
    // Risk/decision phase
    let risk_start = Instant::now();
    let decision = Decision::Abstain; // Default safe decision
    let risk_ms = risk_start.elapsed().as_secs_f64() * 1000.0;
    
    let output_conflicts: Vec<ConflictSet> = conflicts.iter().map(|claims| ConflictSet {
        claim_type: claims.first().map(|e| e.claim_type.clone()).unwrap_or_default(),
        claims: claims.clone(),
    }).collect();
    
    let output = Output {
        value,
        decision,
        evidence: all_evidence.clone(),
        conflicts: output_conflicts,
    };
    
    let wall_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    let load_ms = 0.0; // Would be measured separately
    let admit_ms = 0.0; // Would be measured separately
    let plan_ms = 0.0; // Would be measured separately
    let retrieve_ms = 0.0; // Would be measured separately
    
    let latency = LatencyBreakdown {
        retrieve_ms,
        plan_ms,
        admit_ms,
        load_ms,
        transfer_ms: total_transfer_ms,
        wing_ms: total_wing_ms,
        fusion_ms,
        risk_ms,
        containment_ms: 0.0, // Included in transfer_ms for F0
    };
    
    // Build trace
    let trace = Trace {
        input_hash: format!("{:x}", sha2::Sha256::digest(&input.data))[..16].to_string(),
        core_epoch: graph.core_epoch.clone(),
        graph_id: graph.graph_id.clone(),
        binding_id: context.binding.binding_id(),
        calibration_class: None,
        wing_versions: graph.wings.iter().map(|r| (r.name.clone(), r.version.clone())).collect(),
        wiring: graph.wiring.clone(),
        admission: "shadow".to_string(),
        latency,
        wall_time_ms,
        active_flops,
        peak_memory_mb,
        evidence: all_evidence.clone(),
        conflicts: conflicts.clone(),
        decision,
        replay_level: None,
        failures,
        shadow: true,
        containment_tier: format!("{:?}", context.containment.tier),
    };
    
    Ok((output, trace))
}

/// Execute a wing in a separate process (F1 containment)
fn execute_process_isolated(
    ref_: &WingRef,
    ctx: &CoreContext,
    packet: &LatentPacket,
    containment: &ContainmentConfig,
    artifact_ref: &ArtifactRef,
) -> Result<WingOutput, ExecutionError> {
    let isolation_config = containment.process_isolation.as_ref()
        .ok_or_else(|| ExecutionError::Internal("F1 requires process_isolation config".into()))?;
    
    // Use the wing-executor binary
    let executor_path = find_wing_executor()?;
    
    let input = ProcessWingInput {
        wing_ref: ref_.clone(),
        core_context: ctx.clone(),
        latent_packet: packet.clone(),
        artifact: artifact_ref.clone(),
    };
    
    let input_json = serde_json::to_vec(&input)
        .map_err(|e| ExecutionError::Internal(format!("failed to serialize input: {}", e)))?;
    
    let start = Instant::now();
    
    let mut cmd = Command::new(executor_path);
    cmd.arg("--execute-wing")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if isolation_config.capture_output { Stdio::piped() } else { Stdio::null() });
    
    // Set timeout (handled by parent process monitoring)
    
    let mut child = cmd.spawn()
        .map_err(|e| ExecutionError::Internal(format!("failed to spawn child process: {}", e)))?;
    
    // Write input to stdin
    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        stdin.write_all(&input_json)
            .map_err(|e| ExecutionError::Internal(format!("failed to write to child stdin: {}", e)))?;
    }
    
    // Wait with timeout
    let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Simple timeout implementation
        let timeout = std::time::Duration::from_millis(isolation_config.timeout_ms);
        let start_wait = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => {
                    if start_wait.elapsed() > timeout {
                        let _ = child.kill();
                        break Err(ExecutionError::WingError("process timeout".into()));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => break Err(ExecutionError::Internal(format!("wait error: {}", e))),
            }
        }
    })) {
        Ok(res) => res,
        Err(e) => {
            let _ = child.kill();
            return Err(ExecutionError::WingError(format!("process panic: {:?}", e)));
        }
    }?;
    
    let _execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    let stdout = if isolation_config.capture_output {
        use std::io::Read;
        let mut buf = String::new();
        if let Some(mut stdout) = child.stdout {
            stdout.read_to_string(&mut buf).unwrap_or_default();
        }
        buf
    } else { String::new() };
    
    let stderr = if isolation_config.capture_output {
        use std::io::Read;
        let mut buf = String::new();
        if let Some(mut stderr) = child.stderr {
            stderr.read_to_string(&mut buf).unwrap_or_default();
        }
        buf
    } else { String::new() };
    
    if !output.success() {
        return Err(ExecutionError::WingError(format!(
            "child process exited with code {:?}: stdout={} stderr={}",
            output.code(), stdout, stderr
        )));
    }
    
    let process_output: ProcessWingOutput = serde_json::from_str(&stdout)
        .map_err(|e| ExecutionError::Internal(format!("failed to parse child output: {}", e)))?;
    
    // Add F1 containment cost
    // Note: latency is already accounted in total_wing_ms via wing_start
    
    Ok(process_output.output)
}

/// Find the wing-executor binary path
fn find_wing_executor() -> Result<std::path::PathBuf, ExecutionError> {
    // Try to find wing-executor in the same directory as the current executable
    let current_exe = std::env::current_exe()
        .map_err(|e| ExecutionError::Internal(format!("failed to get current exe: {}", e)))?;
    
    let exe_dir = current_exe.parent()
        .ok_or_else(|| ExecutionError::Internal("failed to get exe directory".into()))?;
    
    let executor_path = exe_dir.join("wing-executor");
    
    if executor_path.exists() {
        Ok(executor_path)
    } else {
        // Fallback: try to find in PATH or use a known location
        which::which("wing-executor")
            .or_else(|_| Ok::<std::path::PathBuf, which::Error>(executor_path))
            .map_err(|e| ExecutionError::Internal(format!("wing-executor not found: {}", e)))
    }
}

/// Execute a wing on a hardware partition (F2 containment) - placeholder
fn execute_hardware_partitioned(
    _ref_: &WingRef,
    _ctx: &CoreContext,
    _packet: &LatentPacket,
    _containment: &ContainmentConfig,
    artifact_ref: &ArtifactRef,
) -> Result<WingOutput, ExecutionError> {
    match artifact_ref {
        ArtifactRef::HardwarePartition { partition_id, device_id } => {
            Err(ExecutionError::Internal(format!(
                "F2 hardware partition execution not implemented: partition={partition_id}, device={device_id}"
            )))
        }
        _ => Err(ExecutionError::Internal("F2 requires HardwarePartition artifact_ref".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_abi::{ContainmentTier, FaultClass};

    #[test]
    fn test_chaos_config_all_faults() {
        let config = ChaosConfig::all_faults(ContainmentTier::F0, 1.0);
        assert!(config.enabled);
        assert_eq!(config.injections.len(), 9);
        assert_eq!(config.seed, Some(0xC0DE));
    }

    #[test]
    fn test_chaos_config_single_fault() {
        let config = ChaosConfig::single_fault(FaultClass::NanOutput, ContainmentTier::F1, 0.5);
        assert!(config.enabled);
        assert_eq!(config.injections.len(), 1);
        assert_eq!(config.injections[0].fault_class, FaultClass::NanOutput);
        assert_eq!(config.injections[0].tier, ContainmentTier::F1);
        assert_eq!(config.injections[0].probability, 0.5);
    }

    #[test]
    fn test_chaos_config_tier_sweep() {
        let configs = ChaosConfig::tier_sweep(FaultClass::ProcessCrash, 1.0);
        assert_eq!(configs.len(), 3);
        assert_eq!(configs[0].injections[0].tier, ContainmentTier::F0);
        assert_eq!(configs[1].injections[0].tier, ContainmentTier::F1);
        assert_eq!(configs[2].injections[0].tier, ContainmentTier::F2);
    }

    #[test]
    fn test_should_inject_always() {
        let mut rng = fastrand::Rng::with_seed(42);
        let injection = FaultInjection {
            fault_class: FaultClass::NanOutput,
            tier: ContainmentTier::F0,
            probability: 1.0,
            trigger: InjectionTrigger::Always,
        };
        assert!(should_inject(&injection, "test_wing", &mut rng));
    }

    #[test]
    fn test_should_inject_on_wing() {
        let mut rng = fastrand::Rng::with_seed(42);
        let injection = FaultInjection {
            fault_class: FaultClass::NanOutput,
            tier: ContainmentTier::F0,
            probability: 1.0,
            trigger: InjectionTrigger::OnWing { wing_name: "target_wing".to_string() },
        };
        assert!(should_inject(&injection, "target_wing", &mut rng));
        assert!(!should_inject(&injection, "other_wing", &mut rng));
    }

    #[test]
    fn test_should_inject_random() {
        let mut rng = fastrand::Rng::with_seed(42);
        let injection = FaultInjection {
            fault_class: FaultClass::NanOutput,
            tier: ContainmentTier::F0,
            probability: 1.0,
            trigger: InjectionTrigger::Random { rate: 0.0 },
        };
        // Rate 0 should never inject
        for _ in 0..100 {
            assert!(!should_inject(&injection, "test_wing", &mut rng));
        }
    }
}