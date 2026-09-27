/// Runtime ABI: the typed capability contract between Wings, planner, and runtime.
/// 
/// This crate mirrors the Python runtime.abi module with exact type correspondence.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Type expression: a declared neural type, e.g., TypeExpr("Image", ("XRay", "Chest"))
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeExpr {
    pub family: String,
    pub args: Vec<String>,
}

impl TypeExpr {
    pub fn new(family: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            family: family.into(),
            args,
        }
    }

    pub fn bare(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            args: Vec::new(),
        }
    }
}

/// Can a value of type `given` be accepted where `required` is declared?
/// MVP strictness: families must match and argument sets must be equal.
pub fn type_unifies(given: &TypeExpr, required: &TypeExpr) -> bool {
    given.family == required.family && given.args == required.args
}

impl fmt::Display for TypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            write!(f, "{}", self.family)
        } else {
            write!(f, "{}<{}>", self.family, self.args.join(","))
        }
    }
}

/// Ordered input or output contract of a Wing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeSet {
    pub entries: Vec<TypeExpr>,
}

impl TypeSet {
    pub fn new(entries: Vec<TypeExpr>) -> Self {
        Self { entries }
    }
    
    pub fn empty() -> Self {
        Self { entries: Vec::new() }
    }
    
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The latent/type contract valid for one Core epoch
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentSchema {
    pub core_epoch: String,
    pub dimensions: usize,
    pub dtype: String,
}

/// The typed value exchanged between Wings and the Core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentPacket {
    pub schema: LatentSchema,
    /// Serialized ndarray data (using numpy's .npy format)
    pub data: Vec<u8>,
    pub provenance: String,
}

impl LatentPacket {
    pub fn new(schema: LatentSchema, data: Vec<u8>, provenance: String) -> Self {
        Self { schema, data, provenance }
    }
}

/// Fixed context handed to a Wing at execute time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreContext {
    pub core_epoch: String,
    pub latent_schema: LatentSchema,
    pub binding_id: Option<String>,
}

/// Evidence polarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    Support,
    Oppose,
    Unknown,
}

/// One scored claim from one Wing version on one route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePacket {
    pub claim_type: String,
    pub claim_value: String,
    pub polarity: Polarity,
    pub source_wing: String,
    pub source_version: String,
    pub route_id: String,
    pub provenance: String,
    pub local_score: f32,
    pub calibration_binding: String,
    pub validity_domain: String,
    pub timestamp: String,
}

/// Contradictory claims kept apart; never averaged by the runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictSet {
    pub claim_type: String,
    pub claims: Vec<EvidencePacket>,
}

/// What a Wing may return. It carries no claim about its own reliability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WingOutput {
    pub latent: LatentPacket,
    pub schema: LatentSchema,
    pub evidence: Vec<EvidencePacket>,
    pub local_score: f32,
    pub wing_version: String,
    pub provenance: String,
}

/// The capability module contract
pub trait Wing: Send + Sync {
    fn manifest(&self) -> Manifest;
    fn input_types(&self) -> TypeSet;
    fn output_types(&self) -> TypeSet;
    fn execute(&self, ctx: CoreContext, x: LatentPacket) -> WingOutput;
}

/// Wing dependency declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version_range: String,
}

/// Resource ceilings the host enforces
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceCeilings {
    pub max_memory_mb: u32,
    pub max_input_size: u32,
}

/// Manifest integrity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestIntegrity {
    pub artifact_hash: String,
}

/// Manifest provenance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestProvenance {
    pub training_dataset_ids: Vec<String>,
    pub training_code_commit: String,
}

/// The complete allow-list a Wing may declare
/// There is deliberately no reliability, accuracy, latency, evidence-gain,
/// relevance or confidence field here: those are host-owned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub core_epoch: String,
    pub author: String,
    pub input_types: TypeSet,
    pub output_types: TypeSet,
    pub capabilities: Vec<String>,
    pub dependencies: Vec<Dependency>,
    pub resource_ceilings: Option<ResourceCeilings>,
    pub integrity: Option<ManifestIntegrity>,
    pub provenance: Option<ManifestProvenance>,
}

/// Final decision of one execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    Accept,
    Abstain,
    Expand,
}

/// Trace replay fidelity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayLevel {
    L1Bitwise,
    L2NumericallyClose,
    L3RouteEquivalent,
    L4DecisionEquivalent,
}

/// The final served (or abstained) result of one execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    pub value: serde_json::Value,
    pub decision: Decision,
    pub evidence: Vec<EvidencePacket>,
    pub conflicts: Vec<ConflictSet>,
}

/// An exact, pinned reference to an installed Wing version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WingRef {
    pub name: String,
    pub version: String,
}

impl WingRef {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl std::fmt::Display for WingRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Host-measured routing attributes. Written by the harness, read by the planner.
/// Never declared by a Wing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostMetrics {
    pub ref_: WingRef,
    pub measured_latency_ms: f32,
    pub measured_memory_mb: f32,
    pub measured_reliability: f32,
    pub measured_utility_features: Vec<f32>,
    pub measured_active_flops: f32,
    pub failure_history: Vec<String>,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("host-owned field {0} is not allowed in a manifest")]
    HostOwnedField(String),
    #[error("unknown manifest field: {0}")]
    UnknownField(String),
    #[error("invalid type expression: {0}")]
    InvalidTypeExpr(String),
    #[error("invalid version: {0}")]
    InvalidVersion(String),
    #[error("invalid version range: {0}")]
    InvalidVersionRange(String),
    #[error("malformed manifest: {0}")]
    Malformed(String),
}

/// Parse "Latent<Text>" or "Latent" into a TypeExpr
pub fn parse_type_expr(text: &str) -> Result<TypeExpr, ManifestError> {
    let text = text.trim();
    if let Some(start) = text.find('<') {
        if text.ends_with('>') {
            let family = text[..start].trim().to_string();
            let args_str = text[start+1..text.len()-1].trim();
            let args: Vec<String> = args_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            return Ok(TypeExpr::new(family, args));
        }
    }
    Ok(TypeExpr::bare(text))
}

/// HOST_OWNED_FIELDS that must never appear in a manifest
pub const HOST_OWNED_FIELDS: &[&str] = &[
    "reliability",
    "accuracy", 
    "expected_latency_ms",
    "expected_evidence_gain",
    "routing_priority",
    "quality",
    "confidence",
];

/// ALLOWED_MANIFEST_KEYS - the strict allow-list
pub const ALLOWED_MANIFEST_KEYS: &[&str] = &[
    "wing",
    "name",
    "version",
    "core_epoch",
    "author",
    "types",
    "inputs",
    "outputs",
    "capabilities",
    "dependencies",
    "required",
    "version_range",
    "resource_ceilings",
    "max_memory_mb",
    "max_input_size",
    "integrity",
    "artifact_hash",
    "provenance",
    "training_dataset_ids",
    "training_code_commit",
];

/// Fault containment tier (F0, F1, F2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContainmentTier {
    /// F0: Logical output checks. Same process, same device context.
    /// Detects NaNs, invalid shapes, range/schema violations.
    /// Cannot contain process crashes, illegal device memory access, context corruption, severe OOM.
    F0,
    /// F1: Process / runtime-context isolation.
    /// Contains process-level crashes, some device-context failures, untrusted code.
    /// Costs: IPC, synchronization, memory duplication, host/device transfer, lost fusion opportunities.
    F1,
    /// F2: Hardware partition or dedicated accelerator.
    /// Strongest containment, highest resource cost.
    F2,
}

/// Fault class categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultClass {
    /// NaN or Inf in output
    NanOutput,
    /// Invalid output shape
    InvalidShape,
    /// Output value out of declared range
    RangeViolation,
    /// Schema/type mismatch
    SchemaMismatch,
    /// Process crash
    ProcessCrash,
    /// Illegal device memory access (context poisoning)
    DeviceMemoryAccess,
    /// Context corruption
    ContextCorruption,
    /// Severe out-of-memory
    SevereOom,
    /// Untrusted code execution
    UntrustedCode,
}

/// Containment profile for a fault class at a given tier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContainmentProfile {
    pub tier: ContainmentTier,
    pub fault_class: FaultClass,
    /// Probability of containing this fault class at this tier (0.0 - 1.0)
    pub isolation_probability: f32,
    /// Additional latency cost in milliseconds
    pub latency_cost_ms: f32,
    /// Additional memory cost in MB
    pub memory_cost_mb: f32,
}

impl ContainmentProfile {
    /// Default profiles matching the paper's characterization
    pub fn defaults() -> Vec<Self> {
        vec![
            // F0 profiles
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::NanOutput, isolation_probability: 1.0, latency_cost_ms: 0.1, memory_cost_mb: 1.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::InvalidShape, isolation_probability: 1.0, latency_cost_ms: 0.1, memory_cost_mb: 1.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::RangeViolation, isolation_probability: 1.0, latency_cost_ms: 0.1, memory_cost_mb: 1.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::SchemaMismatch, isolation_probability: 1.0, latency_cost_ms: 0.1, memory_cost_mb: 1.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::ProcessCrash, isolation_probability: 0.0, latency_cost_ms: 0.0, memory_cost_mb: 0.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::DeviceMemoryAccess, isolation_probability: 0.0, latency_cost_ms: 0.0, memory_cost_mb: 0.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::ContextCorruption, isolation_probability: 0.0, latency_cost_ms: 0.0, memory_cost_mb: 0.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::SevereOom, isolation_probability: 0.0, latency_cost_ms: 0.0, memory_cost_mb: 0.0 },
            Self { tier: ContainmentTier::F0, fault_class: FaultClass::UntrustedCode, isolation_probability: 0.0, latency_cost_ms: 0.0, memory_cost_mb: 0.0 },
            // F1 profiles
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::NanOutput, isolation_probability: 1.0, latency_cost_ms: 2.0, memory_cost_mb: 16.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::InvalidShape, isolation_probability: 1.0, latency_cost_ms: 2.0, memory_cost_mb: 16.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::RangeViolation, isolation_probability: 1.0, latency_cost_ms: 2.0, memory_cost_mb: 16.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::SchemaMismatch, isolation_probability: 1.0, latency_cost_ms: 2.0, memory_cost_mb: 16.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::ProcessCrash, isolation_probability: 0.95, latency_cost_ms: 5.0, memory_cost_mb: 32.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::DeviceMemoryAccess, isolation_probability: 0.6, latency_cost_ms: 10.0, memory_cost_mb: 64.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::ContextCorruption, isolation_probability: 0.8, latency_cost_ms: 5.0, memory_cost_mb: 32.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::SevereOom, isolation_probability: 0.7, latency_cost_ms: 5.0, memory_cost_mb: 64.0 },
            Self { tier: ContainmentTier::F1, fault_class: FaultClass::UntrustedCode, isolation_probability: 0.9, latency_cost_ms: 10.0, memory_cost_mb: 64.0 },
            // F2 profiles
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::NanOutput, isolation_probability: 1.0, latency_cost_ms: 5.0, memory_cost_mb: 128.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::InvalidShape, isolation_probability: 1.0, latency_cost_ms: 5.0, memory_cost_mb: 128.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::RangeViolation, isolation_probability: 1.0, latency_cost_ms: 5.0, memory_cost_mb: 128.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::SchemaMismatch, isolation_probability: 1.0, latency_cost_ms: 5.0, memory_cost_mb: 128.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::ProcessCrash, isolation_probability: 0.999, latency_cost_ms: 10.0, memory_cost_mb: 256.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::DeviceMemoryAccess, isolation_probability: 0.99, latency_cost_ms: 15.0, memory_cost_mb: 256.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::ContextCorruption, isolation_probability: 0.99, latency_cost_ms: 15.0, memory_cost_mb: 256.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::SevereOom, isolation_probability: 0.95, latency_cost_ms: 10.0, memory_cost_mb: 256.0 },
            Self { tier: ContainmentTier::F2, fault_class: FaultClass::UntrustedCode, isolation_probability: 0.999, latency_cost_ms: 15.0, memory_cost_mb: 256.0 },
        ]
    }
}

/// Configuration for fault containment during execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContainmentConfig {
    pub tier: ContainmentTier,
    pub profiles: Vec<ContainmentProfile>,
    pub process_isolation: Option<ProcessIsolationConfig>,
}

impl ContainmentConfig {
    pub fn new(tier: ContainmentTier) -> Self {
        Self {
            tier,
            profiles: ContainmentProfile::defaults(),
            process_isolation: if tier >= ContainmentTier::F1 { Some(ProcessIsolationConfig::default()) } else { None },
        }
    }

    pub fn profile(&self, fault_class: FaultClass) -> Option<&ContainmentProfile> {
        self.profiles.iter().find(|p| p.tier == self.tier && p.fault_class == fault_class)
    }

    pub fn isolation_probability(&self, fault_class: FaultClass) -> f32 {
        self.profile(fault_class).map(|p| p.isolation_probability).unwrap_or(0.0)
    }

    pub fn latency_cost_ms(&self, fault_class: FaultClass) -> f32 {
        self.profile(fault_class).map(|p| p.latency_cost_ms).unwrap_or(0.0)
    }

    pub fn memory_cost_mb(&self, fault_class: FaultClass) -> f32 {
        self.profile(fault_class).map(|p| p.memory_cost_mb).unwrap_or(0.0)
    }
}

/// Reference to a wing artifact for execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactRef {
    /// In-memory wing (F0 only)
    InMemory { wing_id: String },
    /// Executable path for process isolation (F1)
    Executable { path: String, args: Vec<String> },
    /// Hardware partition identifier (F2)
    HardwarePartition { partition_id: String, device_id: String },
}

/// Configuration for F1 process isolation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessIsolationConfig {
    /// Maximum time to wait for child process (ms)
    pub timeout_ms: u64,
    /// Maximum memory for child process (MB)
    pub max_memory_mb: u64,
    /// Whether to capture stdout/stderr
    pub capture_output: bool,
    /// Environment variables to pass to child
    pub env: Vec<(String, String)>,
}

impl Default for ProcessIsolationConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30000,
            max_memory_mb: 1024,
            capture_output: true,
            env: Vec::new(),
        }
    }
}

/// Input for process-isolated wing execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessWingInput {
    pub wing_ref: WingRef,
    pub core_context: CoreContext,
    pub latent_packet: LatentPacket,
    pub artifact: ArtifactRef,
}

/// Output from process-isolated wing execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessWingOutput {
    pub output: WingOutput,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time_ms: f64,
}

/// Core epoch identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoreEpoch {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl CoreEpoch {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
    
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err("CoreEpoch must be in format 'major.minor.patch'".into());
        }
        Ok(Self {
            major: parts[0].parse().map_err(|_| "invalid major")?,
            minor: parts[1].parse().map_err(|_| "invalid minor")?,
            patch: parts[2].parse().map_err(|_| "invalid patch")?,
        })
    }
}

impl std::fmt::Display for CoreEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Core migration compatibility result for a component
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CompatibilityStatus {
    /// Component is compatible without changes
    Compatible,
    /// Component needs retraining (Wing) or revalidation (Bridge/Head)
    RequiresRetraining { reason: String },
    /// Component needs verification (can be verified without full retraining)
    RequiresVerification { reason: String },
    /// Component is incompatible and cannot be migrated
    Incompatible { reason: String },
}

/// Compatibility analysis result for a Core migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCompatibilityReport {
    pub from_epoch: CoreEpoch,
    pub to_epoch: CoreEpoch,
    pub wings: Vec<ComponentCompatibility>,
    pub bridges: Vec<ComponentCompatibility>,
    pub heads: Vec<ComponentCompatibility>,
    pub routes: Vec<RouteCompatibility>,
    pub calibrators: Vec<CalibratorCompatibility>,
    pub calibration_classes: Vec<ClassCompatibility>,
    pub benchmarks: Vec<BenchmarkCompatibility>,
}

/// Individual component compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCompatibility {
    pub name: String,
    pub version: String,
    pub status: CompatibilityStatus,
    pub latent_schema_changes: Vec<SchemaChange>,
}

/// Schema change between epochs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaChange {
    pub field: String,
    pub from_value: String,
    pub to_value: String,
}

/// Route compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteCompatibility {
    pub route_id: String,
    pub graph_id: String,
    pub status: CompatibilityStatus,
    pub affected_wings: Vec<WingRef>,
}

/// Calibrator compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratorCompatibility {
    pub calibrator_id: String,
    pub calibration_class: String,
    pub status: CompatibilityStatus,
    pub coverage_delta: Option<f32>,
}

/// Calibration class compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassCompatibility {
    pub class_id: String,
    pub status: CompatibilityStatus,
    pub affected_routes: Vec<String>,
}

/// Benchmark compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCompatibility {
    pub benchmark_id: String,
    pub status: CompatibilityStatus,
    pub score_delta: Option<f32>,
}

/// Migration cost breakdown (Experiment J)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCost {
    pub n_retrain: usize,
    pub n_revalidate: usize,
    pub n_recalibrate: usize,
    pub t_engineering_hours: f32,
    pub w1: f32,
    pub w2: f32,
    pub w3: f32,
    pub w4: f32,
    pub total_cost: f32,
}

impl MigrationCost {
    pub fn compute(n_retrain: usize, n_revalidate: usize, n_recalibrate: usize, t_engineering_hours: f32, weights: (f32, f32, f32, f32)) -> Self {
        let (w1, w2, w3, w4) = weights;
        let total_cost = w1 * n_retrain as f32 + w2 * n_revalidate as f32 + w3 * n_recalibrate as f32 + w4 * t_engineering_hours;
        Self {
            n_retrain,
            n_revalidate,
            n_recalibrate,
            t_engineering_hours,
            w1, w2, w3, w4,
            total_cost,
        }
    }
}

/// Migration execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub from_epoch: CoreEpoch,
    pub to_epoch: CoreEpoch,
    pub compatibility_report: MigrationCompatibilityReport,
    pub estimated_cost: MigrationCost,
    pub wings_to_retrain: Vec<WingRef>,
    pub bridges_to_revalidate: Vec<BridgeRef>,
    pub heads_to_revalidate: Vec<HeadRef>,
    pub calibrators_to_renew: Vec<CalibratorRef>,
    pub benchmarks_to_regenerate: Vec<String>,
    pub estimated_duration_days: f32,
}

/// Reference to a Bridge version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BridgeRef {
    pub name: String,
    pub version: String,
}

impl BridgeRef {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl std::fmt::Display for BridgeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Reference to a Head version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HeadRef {
    pub name: String,
    pub version: String,
}

impl HeadRef {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl std::fmt::Display for HeadRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Reference to a Calibrator version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CalibratorRef {
    pub name: String,
    pub version: String,
}

impl CalibratorRef {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl std::fmt::Display for CalibratorRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Wing retraining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub validation_split: f32,
    pub early_stopping_patience: usize,
    pub checkpoint_path: String,
}

/// Bridge revalidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRevalidationConfig {
    pub test_dataset: String,
    pub metrics: Vec<String>,
    pub tolerance: f32,
}

/// Calibration renewal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRenewalConfig {
    pub alpha: f32,
    pub holdout_fraction: f32,
    pub min_coverage: f32,
}

/// Benchmark regeneration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRegenerationConfig {
    pub benchmark_name: String,
    pub dataset: String,
    pub metrics: Vec<String>,
    pub num_runs: usize,
}

/// Migration automation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationAutomationResult {
    pub wings_retrained: Vec<RetrainingResult>,
    pub bridges_revalidated: Vec<RevalidationResult>,
    pub heads_revalidated: Vec<RevalidationResult>,
    pub calibrators_renewed: Vec<RenewalResult>,
    pub benchmarks_regenerated: Vec<RegenerationResult>,
    pub total_time_seconds: f64,
    pub success: bool,
}

/// Wing retraining result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingResult {
    pub wing_ref: WingRef,
    pub success: bool,
    pub final_loss: Option<f32>,
    pub final_accuracy: Option<f32>,
    pub epochs_completed: usize,
    pub model_path: Option<String>,
    pub error: Option<String>,
}

/// Bridge/Head revalidation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevalidationResult {
    pub component_ref: String,
    pub component_type: String, // "bridge" or "head"
    pub success: bool,
    pub metrics: HashMap<String, f32>,
    pub within_tolerance: bool,
    pub error: Option<String>,
}

/// Calibration renewal result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenewalResult {
    pub calibrator_ref: CalibratorRef,
    pub calibration_class: String,
    pub success: bool,
    pub new_coverage: Option<f32>,
    pub new_threshold: Option<f32>,
    pub conformal_set_size: Option<f32>,
    pub error: Option<String>,
}

/// Benchmark regeneration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegenerationResult {
    pub benchmark_name: String,
    pub success: bool,
    pub metrics: HashMap<String, f32>,
    pub num_runs: usize,
    pub error: Option<String>,
}

/// Training framework backend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingFramework {
    PyTorch,
    TensorFlow,
    JAX,
    Custom(String),
}

/// Artifact versioning info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactVersion {
    pub version: String,
    pub artifact_hash: String,
    pub framework: TrainingFramework,
    pub training_config: serde_json::Value,
    pub parent_version: Option<String>,
    pub created_at: String,
    pub core_epoch: String,
}

/// Calibration binding for a specific graph/class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBinding {
    pub binding_id: String,
    pub calibration_class: String,
    pub calibrator_ref: CalibratorRef,
    pub threshold: f32,
    pub coverage: f32,
    pub conformal_set_size: f32,
    pub valid_for_epochs: Vec<String>,
    pub created_at: String,
}

/// Wing artifact with versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WingArtifact {
    pub wing_ref: WingRef,
    pub manifest: Manifest,
    pub version_info: ArtifactVersion,
    pub model_bytes: Vec<u8>, // Serialized model weights
    pub calibration_bindings: Vec<CalibrationBinding>,
}

/// Bridge artifact with versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeArtifact {
    pub bridge_ref: BridgeRef,
    pub version_info: ArtifactVersion,
    pub model_bytes: Vec<u8>,
}

/// Artifact registry for versioned storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRegistry {
    pub wings: HashMap<WingRef, WingArtifact>,
    pub bridges: HashMap<BridgeRef, BridgeArtifact>,
    pub heads: HashMap<HeadRef, serde_json::Value>,
    pub calibrators: HashMap<CalibratorRef, serde_json::Value>,
}

/// Training job specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub job_id: String,
    pub wing_ref: WingRef,
    pub framework: TrainingFramework,
    pub config: RetrainingConfig,
    pub dataset_ids: Vec<String>,
    pub target_epoch: String,
    pub priority: u32,
}

/// Training job status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TrainingJobStatus {
    Pending,
    Running { progress: f32, current_epoch: usize },
    Completed { result: RetrainingResult },
    Failed { error: String },
}

/// Calibration binding update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingUpdate {
    pub binding_id: String,
    pub old_threshold: f32,
    pub new_threshold: f32,
    pub old_coverage: f32,
    pub new_coverage: f32,
    pub reason: String,
}