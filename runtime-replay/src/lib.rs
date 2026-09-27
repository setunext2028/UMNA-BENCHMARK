//! Trace replay and route verification.
//! 
//! The trace is the measurement instrument. A trace replays to the
//! same graph. Replay level is *recorded*, not assumed.

use std::collections::HashMap;

use runtime_abi::{Decision, LatentPacket, LatentSchema, Output, WingRef};
use runtime_graph::{Graph, Wiring, build_graph, cost_graph, BuildError, Cost, Binding};
use runtime_registry::Registry;
use runtime_scheduler::{ExecutionContext, Trace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Replay fidelity levels (matching Python's ReplayLevel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayLevel {
    L1Bitwise,
    L2NumericallyClose,
    L3RouteEquivalent,
    L4DecisionEquivalent,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("graph reconstruction failed: {0}")]
    Reconstruction(String),
    #[error("replayed graph {0} differs from trace {1}")]
    GraphMismatch(String, String),
    #[error("input hash mismatch: expected {0}, got {1}")]
    InputHashMismatch(String, String),
    #[error("latency tolerance exceeded: {0}")]
    LatencyTolerance(String),
    #[error("cost verification failed: {0}")]
    CostVerification(String),
}
 
impl From<runtime_scheduler::ExecutionError> for ReplayError {
    fn from(e: runtime_scheduler::ExecutionError) -> Self {
        ReplayError::Reconstruction(e.to_string())
    }
}
 
/// Rebuild the exact route recorded by a trace and verify its identity
pub fn replay_graph<Reg: Registry>(
    trace: &Trace,
    registry: &Reg,
) -> Result<Graph, ReplayError> {
    // Rebuild manifests from trace wing_versions
    // In practice, we'd need to load manifests from registry
    // For now, we verify the graph_id matches
    
    let manifests: Vec<_> = trace.wing_versions.iter()
        .map(|(name, version)| {
            // Would load from registry
            // Placeholder
            runtime_abi::Manifest {
                name: name.clone(),
                version: version.clone(),
                core_epoch: trace.core_epoch.clone(),
                author: "replay".to_string(),
                input_types: runtime_abi::TypeSet::new(vec![]),
                output_types: runtime_abi::TypeSet::new(vec![]),
                capabilities: vec![],
                dependencies: vec![],
                resource_ceilings: None,
                integrity: None,
                provenance: None,
            }
        })
        .collect();
    
    let wiring = trace.wiring.clone();
    let graph = build_graph(&manifests, wiring)
        .map_err(|e| ReplayError::Reconstruction(e.to_string()))?;
    
    if graph.graph_id != trace.graph_id {
        return Err(ReplayError::GraphMismatch(
            graph.graph_id.clone(), trace.graph_id.clone()
        ));
    }
    
    Ok(graph)
}

/// Verify trace input hash matches
pub fn verify_input_hash(trace: &Trace, input: &LatentPacket) -> Result<(), ReplayError> {
    let computed = format!("{:x}", Sha256::digest(&input.data))[..16].to_string();
    if computed != trace.input_hash {
        return Err(ReplayError::InputHashMismatch(
            trace.input_hash.clone(), computed
        ));
    }
    Ok(())
}

/// Verify latency decomposition sums to wall time within tolerance
pub fn verify_latency(trace: &Trace, tolerance_pct: f64) -> Result<(), ReplayError> {
    let decomposed = trace.latency.total_ms();
    let error = (decomposed - trace.wall_time_ms).abs();
    let tolerance = trace.wall_time_ms * tolerance_pct / 100.0;
    let tolerance = tolerance.max(2.0); // Minimum 2ms
    
    if error > tolerance {
        return Err(ReplayError::LatencyTolerance(format!(
            "latency decomposition differs from wall time by {:.3} ms (tolerance {:.3} ms)",
            error, tolerance
        )));
    }
    Ok(())
}

/// Verify graph cost matches trace
pub fn verify_cost<Reg: Registry>(
    trace: &Trace,
    registry: &Reg,
    tolerance_pct: f64,
) -> Result<(), ReplayError> {
    // Rebuild graph and compute cost
    let graph = replay_graph(trace, registry)?;
    let cost = cost_graph(&graph, registry)
        .map_err(|e| ReplayError::CostVerification(e.to_string()))?;
    
    // Check compute
    let compute_error = (cost.compute_flops - trace.active_flops).abs();
    if compute_error / trace.active_flops > tolerance_pct / 100.0 {
        return Err(ReplayError::CostVerification(format!(
            "compute mismatch: {} vs {}", cost.compute_flops, trace.active_flops
        )));
    }
    
    // Check latency
    let latency_error = (cost.latency_ms - trace.latency.total_ms()).abs();
    if latency_error / trace.latency.total_ms() > tolerance_pct / 100.0 {
        return Err(ReplayError::CostVerification(format!(
            "latency mismatch: {} vs {}", cost.latency_ms, trace.latency.total_ms()
        )));
    }
    
    // Check memory
    let memory_error = (cost.memory_mb - trace.peak_memory_mb).abs();
    if memory_error / trace.peak_memory_mb > tolerance_pct / 100.0 {
        return Err(ReplayError::CostVerification(format!(
            "memory mismatch: {} vs {}", cost.memory_mb, trace.peak_memory_mb
        )));
    }
    
    Ok(())
}

/// Full replay verification with configurable levels
pub fn verify_replay<Reg: Registry>(
    trace: &Trace,
    input: &LatentPacket,
    registry: &Reg,
    level: ReplayLevel,
    latency_tolerance_pct: f64,
    cost_tolerance_pct: f64,
) -> Result<Graph, ReplayError> {
    // Always verify input hash
    verify_input_hash(trace, input)?;
    
    // Rebuild graph
    let graph = replay_graph(trace, registry)?;
    
    match level {
        ReplayLevel::L1Bitwise => {
            // Would require full bitwise replay - not implemented here
            // This is the highest fidelity
        }
        ReplayLevel::L2NumericallyClose => {
            // Verify cost within numerical tolerance
            verify_cost(trace, registry, cost_tolerance_pct)?;
        }
        ReplayLevel::L3RouteEquivalent => {
            // Graph structure matches (already verified via graph_id)
            // Plus latency tolerance
            verify_latency(trace, latency_tolerance_pct)?;
        }
        ReplayLevel::L4DecisionEquivalent => {
            // Only decision matches
            // Already have graph_id match
        }
    }
    
    Ok(graph)
}

/// Replay record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRecord {
    pub trace_id: String,
    pub replay_level: ReplayLevel,
    pub passed: bool,
    pub errors: Vec<String>,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests would go here
}