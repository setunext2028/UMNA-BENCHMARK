//! Typed graph construction and validation.
//! 
//! A graph is a set of Wings plus a wiring between their typed outputs and
//! inputs. Type validity is decided at build time: an invalid composition
//! raises TypeError and must never reach execution.

use std::collections::HashMap;

use runtime_abi::{Manifest, TypeExpr, TypeSet, WingRef, type_unifies};
use runtime_registry::version_in_range;
use runtime_registry::Registry;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Directed edges between Wing outputs and Wing inputs
/// Each edge is (from_wing_name, from_output_index, to_wing_name, to_input_index)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wiring {
    pub edges: Vec<(String, usize, String, usize)>,
}

impl Wiring {
    pub fn new(edges: Vec<(String, usize, String, usize)>) -> Self {
        Self { edges }
    }
    
    pub fn empty() -> Self {
        Self { edges: Vec::new() }
    }
    
    pub fn len(&self) -> usize {
        self.edges.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Estimates of what this graph costs to run. Estimates, not promises.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cost {
    pub compute_flops: f64,
    pub latency_ms: f64,
    pub memory_mb: f64,
}

/// A typed execution graph over pinned Wing versions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Graph {
    pub graph_id: String,
    pub wings: Vec<WingRef>,
    pub wiring: Wiring,
    pub core_epoch: String,
}

/// Where calibration attaches: (Core epoch, graph, head, distribution, hardware profile)
/// A change to any load-bearing component is a new binding and needs a new calibrator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub core_epoch: String,
    pub graph: Graph,
    pub head: String,
    pub distribution: String,
    pub hardware_profile: String,
}

impl Binding {
    pub fn new(
        core_epoch: String,
        graph: Graph,
        head: String,
        distribution: String,
        hardware_profile: String,
    ) -> Self {
        Self {
            core_epoch,
            graph,
            head,
            distribution,
            hardware_profile,
        }
    }
    
    pub fn binding_id(&self) -> String {
        format!("{}|{}|{}|{}|{}", 
            self.core_epoch, 
            self.graph.graph_id, 
            self.head, 
            self.distribution, 
            self.hardware_profile
        )
    }
}

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("type error: {0}")]
    TypeError(String),
    #[error("cost error: {0}")]
    CostError(#[from] CostError),
    #[error("manifest error: {0}")]
    ManifestError(#[from] runtime_registry::RegistryError),
}

#[derive(Debug, Error)]
pub enum CostError {
    #[error("missing metrics for wing: {0}")]
    MissingMetrics(String),
    #[error("graph wiring contains a cycle or unresolved edge")]
    CycleOrUnresolved,
    #[error("graph wiring references a Wing outside the graph")]
    OutsideWing,
}

/// Compute canonical graph ID from wings and wiring
fn graph_id(wings: &[WingRef], wiring: &Wiring) -> String {
    let mut hasher = Sha256::new();
    
    // Sort wings for deterministic ID
    let mut sorted_wings = wings.to_vec();
    sorted_wings.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
    
    for ref_ in &sorted_wings {
        hasher.update(ref_.name.as_bytes());
        hasher.update(b"@");
        hasher.update(ref_.version.as_bytes());
        hasher.update(b"|");
    }
    
    // Sort edges for deterministic ID
    let mut sorted_edges = wiring.edges.clone();
    sorted_edges.sort();
    
    for (src, so, dst, di) in &sorted_edges {
        hasher.update(src.as_bytes());
        hasher.update(b":");
        hasher.update(so.to_string().as_bytes());
        hasher.update(b"->");
        hasher.update(dst.as_bytes());
        hasher.update(b":");
        hasher.update(di.to_string().as_bytes());
        hasher.update(b";");
    }
    
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Build a graph from concrete Wing manifests, checking types
pub fn build_graph(
    manifests: &[Manifest],
    wiring: Wiring,
) -> Result<Graph, BuildError> {
    if manifests.is_empty() {
        return Err(BuildError::TypeError("a graph needs at least one wing".into()));
    }
    
    // Check for duplicate wing names
    let mut seen = HashMap::new();
    for m in manifests {
        if seen.insert(m.name.clone(), m.version.clone()).is_some() {
            return Err(BuildError::TypeError(
                "a graph cannot contain two versions of one wing".into()
            ));
        }
    }
    
    // Check all wings share one Core epoch
    let epochs: std::collections::HashSet<_> = manifests.iter().map(|m| m.core_epoch.clone()).collect();
    if epochs.len() > 1 {
        return Err(BuildError::TypeError(format!(
            "mixed Core epochs in one graph: {:?}; one deployment, one epoch", epochs
        )));
    }
    let epoch = epochs.into_iter().next().unwrap();
    
    // Build name -> manifest map
    let by_name: HashMap<String, &Manifest> = manifests.iter()
        .map(|m| (m.name.clone(), m))
        .collect();
    
    // Validate wiring
    let mut incoming: HashMap<String, usize> = HashMap::new();
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    
    for m in manifests {
        incoming.insert(m.name.clone(), 0);
        outgoing.insert(m.name.clone(), Vec::new());
    }
    
    for (src, so, dst, di) in &wiring.edges {
        let src_manifest = by_name.get(src)
            .ok_or_else(|| BuildError::TypeError(format!("wiring references unknown wing {}", src)))?;
        let dst_manifest = by_name.get(dst)
            .ok_or_else(|| BuildError::TypeError(format!("wiring references unknown wing {}", dst)))?;
        
        let outputs = &src_manifest.output_types.entries;
        let inputs = &dst_manifest.input_types.entries;
        
        if *so >= outputs.len() {
            return Err(BuildError::TypeError(format!(
                "{} has no output at index {}", src, so
            )));
        }
        if *di >= inputs.len() {
            return Err(BuildError::TypeError(format!(
                "{} has no input at index {}", dst, di
            )));
        }
        
        if !type_unifies(&outputs[*so], &inputs[*di]) {
            return Err(BuildError::TypeError(format!(
                "type mismatch on edge {}:{} -> {}:{}: {} does not unify with {}",
                src, so, dst, di, outputs[*so], inputs[*di]
            )));
        }
        
        *incoming.get_mut(dst).unwrap() += 1;
        outgoing.get_mut(src).unwrap().push(dst.clone());
        
        if incoming[dst] > 1 {
            return Err(BuildError::TypeError(format!(
                "{} has more than one incoming data edge; the S2 Wing ABI accepts one LatentPacket",
                dst
            )));
        }
    }
    
    // Check for cycles (topological sort)
    let mut remaining = incoming.clone();
    let mut ready: Vec<String> = incoming.iter()
        .filter(|(_, &count)| count == 0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut visited = 0;
    
    while let Some(name) = ready.pop() {
        visited += 1;
        if let Some(targets) = outgoing.get(&name) {
            for target in targets {
                *remaining.get_mut(target).unwrap() -= 1;
                if remaining[target] == 0 {
                    ready.push(target.clone());
                }
            }
        }
    }
    
    if visited != by_name.len() {
        return Err(BuildError::TypeError("graph wiring contains a cycle".into()));
    }
    
    // Check dependencies are satisfied
    for manifest in manifests {
        for dep in &manifest.dependencies {
            let satisfied = manifests.iter().any(|m| {
                m.name == dep.name && m.name != manifest.name && version_in_range(&m.version, &dep.version_range).unwrap_or(false)
            });
            if !satisfied {
                return Err(BuildError::TypeError(format!(
                    "{} requires {} {}, no wing in the graph satisfies it",
                    manifest.name, dep.name, dep.version_range
                )));
            }
        }
    }
    
    // Build canonical graph
    let refs: Vec<WingRef> = manifests.iter()
        .map(|m| WingRef::new(&m.name, &m.version))
        .collect();
    
    // Canonicalize wiring
    let mut canonical_edges = wiring.edges.clone();
    canonical_edges.sort();
    let canonical_wiring = Wiring::new(canonical_edges);
    
    let graph = Graph {
        graph_id: graph_id(&refs, &canonical_wiring),
        wings: refs,
        wiring: canonical_wiring,
        core_epoch: epoch,
    };
    
    Ok(graph)
}

/// Estimate scheduled graph cost from host-owned Wing measurements
pub fn cost_graph<Reg: Registry>(graph: &Graph, registry: &Reg) -> Result<Cost, CostError> {
    let mut metrics = HashMap::new();
    for ref_ in &graph.wings {
        let m = registry.measured(ref_)
            .map_err(|e| CostError::MissingMetrics(e.to_string()))?;
        metrics.insert(ref_.name.clone(), m);
    }
    
    let names: HashMap<String, &WingRef> = graph.wings.iter()
        .map(|r| (r.name.clone(), r))
        .collect();
    
    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    
    for ref_ in &graph.wings {
        indegree.insert(ref_.name.clone(), 0);
        outgoing.insert(ref_.name.clone(), Vec::new());
    }
    
    for (src, _, dst, _) in &graph.wiring.edges {
        if !names.contains_key(src) || !names.contains_key(dst) {
            return Err(CostError::OutsideWing);
        }
        *indegree.get_mut(dst).unwrap() += 1;
        outgoing.get_mut(src).unwrap().push(dst.clone());
    }
    
    let compute_flops: f64 = metrics.values().map(|m| m.measured_active_flops as f64).sum();
    let mut latency_ms: f64 = 0.0;
    let mut memory_mb: f64 = 0.0;
    
    let mut processed = std::collections::HashSet::new();
    let mut ready: Vec<String> = indegree.iter()
        .filter(|(_, &count)| count == 0)
        .map(|(name, _)| name.clone())
        .collect();
    
    while !ready.is_empty() {
        let wave_latency = ready.iter()
            .map(|name| metrics.get(name).map(|m| m.measured_latency_ms as f64).unwrap_or(0.0))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        latency_ms += wave_latency;
        memory_mb = memory_mb.max(
            ready.iter().map(|name| metrics.get(name).map(|m| m.measured_memory_mb as f64).unwrap_or(0.0)).sum()
        );
        
        let mut to_add = Vec::new();
        for name in &ready {
            processed.insert(name.clone());
            if let Some(targets) = outgoing.get(name) {
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
        ready.extend(to_add);
        ready = indegree.iter()
            .filter(|(name, &count)| count == 0 && !processed.contains(name.as_str()))
            .map(|(name, _)| name.clone())
            .collect();
    }
    
    if processed.len() != names.len() {
        return Err(CostError::CycleOrUnresolved);
    }
    
    Ok(Cost {
        compute_flops,
        latency_ms,
        memory_mb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_abi::{Manifest, TypeExpr, TypeSet, ResourceCeilings, ManifestIntegrity, ManifestProvenance};
    
    fn make_manifest(name: &str) -> Manifest {
        Manifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            core_epoch: "1.x".to_string(),
            author: "test".to_string(),
            input_types: TypeSet::new(vec![TypeExpr::new("Latent", vec!["Token".to_string()])]),
            output_types: TypeSet::new(vec![TypeExpr::new("Latent", vec!["Token".to_string()])]),
            capabilities: vec![],
            dependencies: vec![],
            resource_ceilings: Some(ResourceCeilings { max_memory_mb: 128, max_input_size: 100000 }),
            integrity: Some(ManifestIntegrity { artifact_hash: "a".repeat(64) }),
            provenance: Some(ManifestProvenance { 
                training_dataset_ids: vec!["test".to_string()], 
                training_code_commit: "test".to_string() 
            }),
        }
    }
    
    #[test]
    fn test_graph_id_deterministic() {
        let m1 = make_manifest("a");
        let m2 = make_manifest("b");
        let w1 = Wiring::new(vec![("a".into(), 0, "b".into(), 0)]);
        let w2 = Wiring::new(vec![("a".into(), 0, "b".into(), 0)]);
        
        let g1 = build_graph(&[m1.clone(), m2.clone()], w1).unwrap();
        let g2 = build_graph(&[m2, m1], w2).unwrap(); // Different order
        
        assert_eq!(g1.graph_id, g2.graph_id);
    }
    
    #[test]
    fn test_type_mismatch_fails() {
        let mut m1 = make_manifest("a");
        m1.output_types = TypeSet::new(vec![TypeExpr::new("Latent", vec!["A".to_string()])]);
        let mut m2 = make_manifest("b");
        m2.input_types = TypeSet::new(vec![TypeExpr::new("Latent", vec!["B".to_string()])]);
        
        let w = Wiring::new(vec![("a".into(), 0, "b".into(), 0)]);
        let result = build_graph(&[m1, m2], w);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_cycle_fails() {
        let m1 = make_manifest("a");
        let m2 = make_manifest("b");
        let w = Wiring::new(vec![("a".into(), 0, "b".into(), 0), ("b".into(), 0, "a".into(), 0)]);
        let result = build_graph(&[m1, m2], w);
        assert!(result.is_err());
    }
}