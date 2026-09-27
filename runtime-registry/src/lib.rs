//! Registry seams: load, validate, resolve, pin.
//! 
//! The registry is the only place a manifest becomes an installed, measurable
//! Wing. HostMetrics is the single source of routing-relevant numbers; nothing
//! routing-relevant is ever read from a manifest.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use runtime_abi::{
    Dependency, Manifest, ManifestError, ManifestIntegrity, ManifestProvenance,
    ResourceCeilings, TypeExpr, TypeSet, WingRef, Wing, HOST_OWNED_FIELDS,
};
pub use runtime_abi::HostMetrics;
use regex::Regex;
use thiserror::Error;

/// Semantic version regex
static SEMVER_REGEX: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
    Regex::new(r"^\d+\.\d+\.\d+$").unwrap()
});

fn parse_semver(version: &str) -> Result<(u32, u32, u32), ManifestError> {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() != 3 {
        return Err(ManifestError::InvalidVersion(version.to_string()));
    }
    Ok((
        parts[0].parse().map_err(|_| ManifestError::InvalidVersion(version.to_string()))?,
        parts[1].parse().map_err(|_| ManifestError::InvalidVersion(version.to_string()))?,
        parts[2].parse().map_err(|_| ManifestError::InvalidVersion(version.to_string()))?,
    ))
}

/// Check if a version satisfies a range spec like ">=2,<3" or ">=1.2.0,<2.0.0"
pub fn version_in_range(version: &str, range_spec: &str) -> Result<bool, ManifestError> {
    if range_spec == "*" {
        return Ok(true);
    }
    
    // Exact match
    if SEMVER_REGEX.is_match(range_spec) {
        return Ok(version == range_spec);
    }
    
    let (v_major, v_minor, v_patch) = parse_semver(version)?;
    
    for clause in range_spec.split(',') {
        let clause = clause.trim();
        let (op, bound_str) = if clause.starts_with(">=") {
            (">=", &clause[2..])
        } else if clause.starts_with("<=") {
            ("<=", &clause[2..])
        } else if clause.starts_with('>') {
            (">", &clause[1..])
        } else if clause.starts_with('<') {
            ("<", &clause[1..])
        } else {
            ("=", clause)
        };
        
        let (b_major, b_minor, b_patch) = parse_semver(bound_str)?;
        
        let v = (v_major, v_minor, v_patch);
        let b = (b_major, b_minor, b_patch);
        
        let ok = match op {
            ">" => v > b,
            ">=" => v >= b,
            "<" => v < b,
            "<=" => v <= b,
            "=" => v == b,
            _ => return Err(ManifestError::InvalidVersionRange(range_spec.to_string())),
        };
        
        if !ok {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Registry errors
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),
    #[error("version error: {0}")]
    Version(String),
    #[error("not installed: {0}")]
    NotInstalled(String),
    #[error("epoch mismatch: {0}")]
    EpochMismatch(String),
    #[error("version diamond: {0}")]
    VersionDiamond(String),
    #[error("dependency unsatisfied: {0}")]
    DependencyUnsatisfied(String),
    #[error("metrics error: {0}")]
    Metrics(String),
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Registry trait - the installed Wing set, epoch-pinned and hash-checked
pub trait Registry: Send + Sync {
    fn load(&self, manifest: Manifest) -> Result<WingRef, RegistryError>;
    fn validate(&self, manifest: &Manifest) -> Result<(), RegistryError>;
    fn resolve(&self, refs: &[WingRef], core_epoch: Option<&str>) -> Result<Vec<WingRef>, RegistryError>;
    fn pin(&self, core_epoch: &str) -> Result<(), RegistryError>;
    fn measured(&self, ref_: &WingRef) -> Result<HostMetrics, RegistryError>;
    fn retrieve(&self, x: &TypeExpr, m: usize) -> Result<Vec<WingRef>, RegistryError>;
    fn installed(&self) -> Vec<WingRef>;
    fn check_input(&self, ref_: &WingRef, size: usize) -> Result<bool, RegistryError>;
    fn artifact(&self, ref_: &WingRef) -> Result<Arc<dyn runtime_abi::Wing>, RegistryError>;
    fn get_manifest(&self, ref_: &WingRef) -> Result<Manifest, RegistryError>;
    fn core_epoch(&self) -> String;
    fn get_artifact_ref(&self, ref_: &WingRef) -> Result<runtime_abi::ArtifactRef, RegistryError>;
    fn set_artifact_ref(&self, ref_: &WingRef, artifact: runtime_abi::ArtifactRef) -> Result<(), RegistryError>;
}

/// In-memory, file-backed registry of the MVP
pub struct InMemoryRegistry {
    epoch: RwLock<Option<String>>,
    installed: DashMap<String, DashMap<String, Manifest>>, // name -> version -> manifest
    artifacts: DashMap<WingRef, Arc<dyn runtime_abi::Wing>>,
    artifact_refs: DashMap<WingRef, runtime_abi::ArtifactRef>,
    metrics: DashMap<WingRef, HostMetrics>,
    order: RwLock<Vec<WingRef>>,
}

impl InMemoryRegistry {
    pub fn new() -> Self {
        Self {
            epoch: RwLock::new(None),
            installed: DashMap::new(),
            artifacts: DashMap::new(),
            artifact_refs: DashMap::new(),
            metrics: DashMap::new(),
            order: RwLock::new(Vec::new()),
        }
    }
    
    pub fn with_core_epoch(epoch: String) -> Self {
        let reg = Self::new();
        reg.pin(&epoch).unwrap();
        reg
    }
    
    /// Register a wing artifact for testing
    pub fn set_artifact(&self, ref_: &WingRef, artifact: Arc<dyn runtime_abi::Wing>) -> Result<(), RegistryError> {
        if !self.order.read().contains(ref_) {
            return Err(RegistryError::NotInstalled(ref_.to_string()));
        }
        self.artifacts.insert(ref_.clone(), artifact);
        Ok(())
    }
    
    /// Seed directory loads every wings/<name>/manifest.yaml under it
    pub fn seed_from_dir(&self, dir: &std::path::Path) -> Result<(), RegistryError> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let manifest_path = entry.path().join("manifest.yaml");
            if manifest_path.exists() {
                let content = std::fs::read_to_string(&manifest_path)?;
                let manifest: Manifest = serde_yaml::from_str(&content)?;
                self.load(manifest)?;
            }
        }
        Ok(())
    }
}

impl Registry for InMemoryRegistry {
    fn load(&self, manifest: Manifest) -> Result<WingRef, RegistryError> {
        self.validate(&manifest)?;
        
        // Check epoch pin
        let epoch_guard = self.epoch.read();
        if let Some(pinned) = epoch_guard.as_ref() {
            if manifest.core_epoch != *pinned {
                return Err(RegistryError::EpochMismatch(format!(
                    "wing {} declares core_epoch {}, pinned epoch is {}",
                    manifest.name, manifest.core_epoch, pinned
                )));
            }
        }
        drop(epoch_guard);
        
        let ref_ = WingRef::new(&manifest.name, &manifest.version);
        
        // Check for existing
        if let Some(versions) = self.installed.get(&manifest.name) {
            if let Some(existing_ref) = versions.get(&manifest.version) {
                let existing = existing_ref.value();
                if *existing != manifest {
                    return Err(RegistryError::Version(format!(
                        "{} is already installed with a different manifest", ref_
                    )));
                }
                return Ok(ref_);
            }
        }
        
        // Install
        self.installed
            .entry(manifest.name.clone())
            .or_default()
            .insert(manifest.version.clone(), manifest.clone());
        self.order.write().push(ref_.clone());
        
        Ok(ref_)
    }
    
    fn validate(&self, manifest: &Manifest) -> Result<(), RegistryError> {
        // Check host-owned fields
        // (In practice, this is done at YAML parse time)
        
        if manifest.name.is_empty() {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "wing name must be non-empty".into()
            )));
        }
        
        parse_semver(&manifest.version)?;
        
        if manifest.core_epoch.is_empty() {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "core_epoch must not be empty".into()
            )));
        }
        
        if manifest.author.is_empty() {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "author must be non-empty".into()
            )));
        }
        
        if manifest.input_types.entries.is_empty() || manifest.output_types.entries.is_empty() {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "a wing needs at least one input and one output type".into()
            )));
        }
        
        for dep in &manifest.dependencies {
            if dep.name.is_empty() || dep.version_range.is_empty() {
                return Err(RegistryError::Manifest(ManifestError::Malformed(
                    "dependencies need non-empty name and version_range".into()
                )));
            }
            // Self-dependency check
            version_in_range(&manifest.version, &dep.version_range)?;
        }
        
        if let Some(ceilings) = &manifest.resource_ceilings {
            if ceilings.max_memory_mb == 0 || ceilings.max_input_size == 0 {
                return Err(RegistryError::Manifest(ManifestError::Malformed(
                    "resource ceilings must be positive".into()
                )));
            }
        }
        
        if let Some(integrity) = &manifest.integrity {
            if integrity.artifact_hash.is_empty() {
                return Err(RegistryError::Manifest(ManifestError::Malformed(
                    "artifact_hash must be non-empty".into()
                )));
            }
        }
        
        if let Some(prov) = &manifest.provenance {
            for ds in &prov.training_dataset_ids {
                if ds.is_empty() {
                    return Err(RegistryError::Manifest(ManifestError::Malformed(
                        "training_dataset_ids must be non-empty strings".into()
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    fn resolve(&self, refs: &[WingRef], core_epoch: Option<&str>) -> Result<Vec<WingRef>, RegistryError> {
        let epoch = core_epoch
            .map(|s| s.to_string())
            .or_else(|| self.epoch.read().clone())
            .ok_or_else(|| RegistryError::EpochMismatch("no Core epoch pinned".into()))?;
        
        let mut resolved: HashMap<String, String> = HashMap::new();
        let mut pending = refs.to_vec();
        
        while let Some(ref_) = pending.pop() {
            let manifest = self.installed
                .get(&ref_.name)
                .and_then(|v| v.get(&ref_.version).map(|m| m.clone()))
                .ok_or_else(|| RegistryError::NotInstalled(ref_.to_string()))?;
            
            if manifest.core_epoch != epoch {
                return Err(RegistryError::EpochMismatch(format!(
                    "{} belongs to Core epoch {}, resolution requires {}",
                    ref_, manifest.core_epoch, epoch
                )));
            }
            
            if let Some(existing) = resolved.get(&manifest.name) {
                if existing != &manifest.version {
                    return Err(RegistryError::VersionDiamond(format!(
                        "version diamond for {}: {} and {} cannot coexist in one Core epoch",
                        manifest.name, existing, manifest.version
                    )));
                }
                continue;
            }
            
            resolved.insert(manifest.name.clone(), manifest.version.clone());
            
            for dep in &manifest.dependencies {
                // Find matching version
                let versions = self.installed
                    .get(&dep.name)
                    .ok_or_else(|| RegistryError::DependencyUnsatisfied(format!(
                        "{} requires {} {}, no installed version satisfies it",
                        ref_, dep.name, dep.version_range
                    )))?;
                
                let mut candidates: Vec<String> = versions
                    .iter()
                    .map(|entry| (entry.key().clone(), entry.value().clone()))
                    .filter(|(v, m)| m.core_epoch == epoch && version_in_range(v, &dep.version_range).unwrap_or(false))
                    .map(|(v, _)| v)
                    .collect();
                
                if candidates.is_empty() {
                    return Err(RegistryError::DependencyUnsatisfied(format!(
                        "{} requires {} {}, no installed version satisfies it",
                        ref_, dep.name, dep.version_range
                    )));
                }
                
                candidates.sort_by_key(|v| parse_semver(v).unwrap_or((0,0,0)));
                if let Some(last) = candidates.last() {
                    pending.push(WingRef::new(&dep.name, last.as_str()));
                }
            }
        }
        
        // Return in deterministic order
        let mut ordered = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        for ref_ in refs {
            let version = resolved.get(&ref_.name).unwrap();
            let picked = WingRef::new(&ref_.name, version);
            if seen.insert(picked.clone()) {
                ordered.push(picked);
            }
        }
        
        for (name, version) in resolved {
            let picked = WingRef::new(&name, &version);
            if seen.insert(picked.clone()) {
                ordered.push(picked);
            }
        }
        
        Ok(ordered)
    }
    
    fn pin(&self, core_epoch: &str) -> Result<(), RegistryError> {
        if core_epoch.is_empty() {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "Core epoch pin must be non-empty".into()
            )));
        }
        
        let mut epoch_guard = self.epoch.write();
        if let Some(existing) = epoch_guard.as_ref() {
            if existing != core_epoch {
                return Err(RegistryError::EpochMismatch(format!(
                    "Core epoch is already pinned to {}; cannot repin to {}",
                    existing, core_epoch
                )));
            }
        } else {
            // Check for incompatible installed wings
            let incompatible: Vec<WingRef> = self.order.read()
                .iter()
                .filter_map(|ref_| {
                    self.installed.get(&ref_.name)
                        .and_then(|v| v.get(&ref_.version).map(|m| m.clone()))
                        .filter(|m| m.core_epoch != core_epoch)
                        .map(|_| ref_.clone())
                })
                .collect();
            
            if !incompatible.is_empty() {
                return Err(RegistryError::EpochMismatch(format!(
                    "cannot pin Core epoch {}; incompatible installed wings: {:?}",
                    core_epoch, incompatible
                )));
            }
            
            *epoch_guard = Some(core_epoch.to_string());
        }
        Ok(())
    }
    
    fn measured(&self, ref_: &WingRef) -> Result<HostMetrics, RegistryError> {
        self.metrics.get(ref_)
            .map(|m| m.clone())
            .ok_or_else(|| RegistryError::Metrics(format!(
                "no host measurements for {}; the harness must measure a wing before it is eligible for routing",
                ref_
            )))
    }
    
    fn retrieve(&self, x: &TypeExpr, m: usize) -> Result<Vec<WingRef>, RegistryError> {
        if m == 0 {
            return Ok(Vec::new());
        }
        
        let epoch = self.epoch.read().clone();
        let mut hits = Vec::new();
        
        for ref_ in self.order.read().iter() {
            let manifest = self.installed
                .get(&ref_.name)
                .and_then(|v| v.get(&ref_.version).map(|m| m.clone()));
            
            if let Some(manifest) = manifest {
                if let Some(pinned) = &epoch {
                    if manifest.core_epoch != *pinned {
                        continue;
                    }
                }
                
                if manifest.input_types.entries.iter().any(|inp| runtime_abi::type_unifies(x, inp)) {
                    hits.push(ref_.clone());
                    if hits.len() >= m {
                        break;
                    }
                }
            }
        }
        
        Ok(hits)
    }
    
    fn installed(&self) -> Vec<WingRef> {
        self.order.read().clone()
    }
    
    fn check_input(&self, ref_: &WingRef, size: usize) -> Result<bool, RegistryError> {
        let manifest = self.installed
            .get(&ref_.name)
            .and_then(|v| v.get(&ref_.version).map(|m| m.clone()))
            .ok_or_else(|| RegistryError::NotInstalled(ref_.to_string()))?;
        
        if size < 0 {
            return Err(RegistryError::Manifest(ManifestError::Malformed(
                "input size cannot be negative".into()
            )));
        }
        
        if let Some(ceilings) = &manifest.resource_ceilings {
            Ok(size <= ceilings.max_input_size as usize)
        } else {
            Ok(true)
        }
    }
    fn artifact(&self, ref_: &WingRef) -> Result<Arc<dyn runtime_abi::Wing>, RegistryError> {
        self.artifacts.get(ref_)
            .map(|a| a.clone())
            .ok_or_else(|| RegistryError::NotInstalled(ref_.to_string()))
    }
    
    fn get_manifest(&self, ref_: &WingRef) -> Result<Manifest, RegistryError> {
        self.installed.get(&ref_.name)
            .and_then(|versions| versions.get(&ref_.version).map(|v| v.value().clone()))
            .ok_or_else(|| RegistryError::NotInstalled(ref_.to_string()))
    }
    
    fn core_epoch(&self) -> String {
        self.epoch.read().clone().unwrap_or_default()
    }
    
    fn get_artifact_ref(&self, ref_: &WingRef) -> Result<runtime_abi::ArtifactRef, RegistryError> {
        Ok(self.artifact_refs.get(ref_)
            .map(|a| a.clone())
            .unwrap_or_else(|| {
                // Default to InMemory if not explicitly set
                runtime_abi::ArtifactRef::InMemory { 
                    wing_id: format!("{}@{}", ref_.name, ref_.version) 
                }
            }))
    }
    
    fn set_artifact_ref(&self, ref_: &WingRef, artifact: runtime_abi::ArtifactRef) -> Result<(), RegistryError> {
        if !self.order.read().contains(ref_) {
            return Err(RegistryError::NotInstalled(ref_.to_string()));
        }
        self.artifact_refs.insert(ref_.clone(), artifact);
        Ok(())
    }
}

impl InMemoryRegistry {
    /// Set host-measured metrics for a wing
    pub fn set_measured(&self, metrics: HostMetrics) -> Result<(), RegistryError> {
        if !self.order.read().contains(&metrics.ref_) {
            return Err(RegistryError::Metrics(format!(
                "cannot attach host measurements to uninstalled {}",
                metrics.ref_
            )));
        }
        
        if !metrics.measured_latency_ms.is_finite() 
            || !metrics.measured_memory_mb.is_finite()
            || !metrics.measured_active_flops.is_finite()
            || metrics.measured_latency_ms < 0.0
            || metrics.measured_memory_mb < 0.0
            || metrics.measured_active_flops < 0.0 {
            return Err(RegistryError::Metrics(
                "host-measured latency, memory and active FLOPs must be finite and non-negative".into()
            ));
        }
        
        if !metrics.measured_reliability.is_finite() 
            || metrics.measured_reliability < 0.0 
            || metrics.measured_reliability > 1.0 {
            return Err(RegistryError::Metrics(
                "host-measured reliability must be in [0, 1]".into()
            ));
        }
        
        for val in &metrics.measured_utility_features {
            if !val.is_finite() {
                return Err(RegistryError::Metrics(
                    "host-measured utility features must be finite".into()
                ));
            }
        }
        
        // Check against declared ceiling
        if let Some(manifest) = self.installed.get(&metrics.ref_.name)
            .and_then(|v| v.get(&metrics.ref_.version).map(|m| m.clone())) {
            if let Some(ceilings) = &manifest.resource_ceilings {
                if metrics.measured_memory_mb > ceilings.max_memory_mb as f32 {
                    return Err(RegistryError::Metrics(format!(
                        "{} measured memory {} MB exceeds declared ceiling {} MB",
                        metrics.ref_, metrics.measured_memory_mb, ceilings.max_memory_mb
                    )));
                }
            }
        }
        
        self.metrics.insert(metrics.ref_.clone(), metrics);
        Ok(())
    }
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
    fn test_load_and_validate() {
        let reg = InMemoryRegistry::with_core_epoch("1.x".to_string());
        let manifest = make_manifest("test_wing");
        let ref_ = reg.load(manifest).unwrap();
        assert_eq!(ref_.name, "test_wing");
        assert_eq!(ref_.version, "1.0.0");
    }
    
    #[test]
    fn test_epoch_mismatch_rejected() {
        let reg = InMemoryRegistry::with_core_epoch("1.x".to_string());
        let mut manifest = make_manifest("bad_wing");
        manifest.core_epoch = "2.x".to_string();
        let result = reg.load(manifest);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_version_in_range() {
        assert!(version_in_range("1.2.3", "1.2.3").unwrap());
        assert!(version_in_range("1.2.3", "*").unwrap());
        assert!(version_in_range("1.2.3", ">=1.0.0,<2.0.0").unwrap());
        assert!(!version_in_range("2.0.0", ">=1.0.0,<2.0.0").unwrap());
        assert!(!version_in_range("1.2.3", ">1.2.3").unwrap());
        assert!(version_in_range("1.2.4", ">1.2.3").unwrap());
    }
}