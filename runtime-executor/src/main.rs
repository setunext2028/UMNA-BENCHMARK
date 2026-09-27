//! Wing Executor: standalone binary for F1/F2 process-isolated wing execution.
//!
//! This binary is spawned by the scheduler when running at F1/F2 containment tier.
//! It communicates via stdin/stdout JSON serialization.

use std::io::{self, Read, Write};
use runtime_abi::{ProcessWingInput, ProcessWingOutput, WingOutput};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Wing execution error: {0}")]
    Wing(String),
    #[error("Artifact loading error: {0}")]
    Artifact(String),
    #[error("GPU error: {0}")]
    Gpu(String),
}

fn main() -> Result<(), ExecutorError> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 || args[1] != "--execute-wing" {
        eprintln!("Usage: wing-executor --execute-wing");
        std::process::exit(1);
    }
    
    // Read input from stdin
    let mut input_json = String::new();
    io::stdin().read_to_string(&mut input_json)?;
    
    // Parse input
    let input: ProcessWingInput = serde_json::from_str(&input_json)?;
    
    // Execute wing based on artifact type
    let output = match &input.artifact {
        runtime_abi::ArtifactRef::InMemory { wing_id } => {
            execute_in_memory(&input, wing_id)?
        }
        runtime_abi::ArtifactRef::Executable { path, args } => {
            execute_executable(&input, path, args)?
        }
        runtime_abi::ArtifactRef::HardwarePartition { partition_id, device_id } => {
            execute_hardware_partition(&input, partition_id, device_id)?
        }
    };
    
    // Write output to stdout
    let output_json = serde_json::to_string(&output)?;
    io::stdout().write_all(output_json.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}

fn execute_in_memory(input: &ProcessWingInput, wing_id: &str) -> Result<ProcessWingOutput, ExecutorError> {
    // In a real implementation, this would:
    // 1. Load the wing from a shared library or in-process registry
    // 2. Execute it with the provided input
    // 3. Return the output
    
    // For now, return a mock output
    let mock_output = WingOutput {
        latent: runtime_abi::LatentPacket {
            schema: input.core_context.latent_schema.clone(),
            data: vec![0u8; 32],
            provenance: format!("mock:{wing_id}"),
        },
        schema: input.core_context.latent_schema.clone(),
        evidence: vec![],
        local_score: 0.9,
        wing_version: "1.0.0".to_string(),
        provenance: format!("mock:{wing_id}"),
    };
    
    Ok(ProcessWingOutput {
        output: mock_output,
        exit_code: 0,
        stdout: format!("Executed wing {wing_id} in-memory"),
        stderr: String::new(),
        execution_time_ms: 1.0,
    })
}

fn execute_executable(input: &ProcessWingInput, path: &str, args: &[String]) -> Result<ProcessWingOutput, ExecutorError> {
    // In a real implementation, this would:
    // 1. Spawn the executable with the given args
    // 2. Pass input via stdin
    // 3. Capture output from stdout
    
    use std::process::{Command, Stdio};
    
    let mut cmd = Command::new(path);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    let mut child = cmd.spawn()
        .map_err(|e| ExecutorError::Artifact(format!("failed to spawn {path}: {e}")))?;
    
    // Write input to stdin
    let input_json = serde_json::to_vec(&input)?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(&input_json)?;
    }
    
    // Wait for completion
    let output = child.wait_with_output()?;
    
    if !output.status.success() {
        return Ok(ProcessWingOutput {
            output: WingOutput {
                latent: runtime_abi::LatentPacket {
                    schema: input.core_context.latent_schema.clone(),
                    data: vec![],
                    provenance: "error".to_string(),
                },
                schema: input.core_context.latent_schema.clone(),
                evidence: vec![],
                local_score: 0.0,
                wing_version: "0.0.0".to_string(),
                provenance: "error".to_string(),
            },
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            execution_time_ms: 0.0,
        });
    }
    
    // Parse output
    let process_output: ProcessWingOutput = serde_json::from_slice(&output.stdout)?;
    Ok(process_output)
}

/// CUDA MPS execution context
#[cfg(feature = "cuda")]
mod cuda_mps {
    use super::*;
    use std::process::{Command, Stdio};
    
    pub fn execute_cuda_mps(
        input: &ProcessWingInput,
        partition_id: &str,
        device_id: &str,
    ) -> Result<ProcessWingOutput, ExecutorError> {
        // CUDA MPS allows multiple processes to share a GPU
        // Each partition gets a dedicated MPS client context
        
        let mps_daemon = std::env::var("CUDA_MPS_PIPE_DIRECTORY")
            .unwrap_or_else(|_| "/tmp/nvidia-mps".to_string());
        
        // Set up MPS environment for this partition
        let mut cmd = Command::new("nvidia-cuda-mps-control");
        cmd.env("CUDA_MPS_PIPE_DIRECTORY", &mps_daemon)
            .env("CUDA_VISIBLE_DEVICES", device_id)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        
        let mut child = cmd.spawn()
            .map_err(|e| ExecutorError::Gpu(format!("failed to start MPS control: {e}")))?;
        
        // Create MPS client for this partition
        let create_client_cmd = format!("create_client -partition {}", partition_id);
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(create_client_cmd.as_bytes())?;
        }
        
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(ExecutorError::Gpu(format!(
                "MPS client creation failed: {}", 
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        
        // Now execute the wing within the MPS partition
        // This would typically load a CUDA module and execute kernels
        // For now, simulate with a mock
        let mock_output = WingOutput {
            latent: runtime_abi::LatentPacket {
                schema: input.core_context.latent_schema.clone(),
                data: vec![0u8; 32],
                provenance: format!("cuda-mps:{partition_id}:{device_id}"),
            },
            schema: input.core_context.latent_schema.clone(),
            evidence: vec![],
            local_score: 0.95,
            wing_version: "1.0.0".to_string(),
            provenance: format!("cuda-mps:{partition_id}:{device_id}"),
        };
        
        Ok(ProcessWingOutput {
            output: mock_output,
            exit_code: 0,
            stdout: format!("Executed wing in CUDA MPS partition {partition_id} on device {device_id}"),
            stderr: String::new(),
            execution_time_ms: 5.0,
        })
    }
}

/// Vulkan compute execution context
#[cfg(feature = "ash")]
mod vulkan_compute {
    use super::*;
    use ash::{vk, Entry};
    
    pub fn execute_vulkan(
        input: &ProcessWingInput,
        partition_id: &str,
        device_id: &str,
    ) -> Result<ProcessWingOutput, ExecutorError> {
        // Vulkan compute for GPU acceleration
        // Each partition gets a dedicated Vulkan device/queue
        
        let entry = Entry::linked()
            .map_err(|e| ExecutorError::Gpu(format!("failed to load Vulkan: {e}")))?;
        
        // In a real implementation:
        // 1. Create Vulkan instance with required extensions
        // 2. Select physical device matching device_id
        // 3. Create logical device with compute queue
        // 4. Allocate buffers for input/output tensors
        // 5. Load and bind compute shader (compiled SPIR-V)
        // 6. Dispatch compute commands
        // 7. Read back results
        
        // For now, return mock output
        let mock_output = WingOutput {
            latent: runtime_abi::LatentPacket {
                schema: input.core_context.latent_schema.clone(),
                data: vec![0u8; 32],
                provenance: format!("vulkan:{partition_id}:{device_id}"),
            },
            schema: input.core_context.latent_schema.clone(),
            evidence: vec![],
            local_score: 0.93,
            wing_version: "1.0.0".to_string(),
            provenance: format!("vulkan:{partition_id}:{device_id}"),
        };
        
        Ok(ProcessWingOutput {
            output: mock_output,
            exit_code: 0,
            stdout: format!("Executed wing in Vulkan partition {partition_id} on device {device_id}"),
            stderr: String::new(),
            execution_time_ms: 3.0,
        })
    }
}

/// Dedicated accelerator execution (TPU, NPU, etc.)
mod dedicated_accelerator {
    use super::*;
    
    pub fn execute_dedicated(
        input: &ProcessWingInput,
        partition_id: &str,
        device_id: &str,
    ) -> Result<ProcessWingOutput, ExecutorError> {
        // Placeholder for dedicated accelerators (TPU, NPU, etc.)
        // Would interface with accelerator-specific runtime (e.g., libtpu, RKNN, etc.)
        
        let mock_output = WingOutput {
            latent: runtime_abi::LatentPacket {
                schema: input.core_context.latent_schema.clone(),
                data: vec![0u8; 32],
                provenance: format!("dedicated:{partition_id}:{device_id}"),
            },
            schema: input.core_context.latent_schema.clone(),
            evidence: vec![],
            local_score: 0.97,
            wing_version: "1.0.0".to_string(),
            provenance: format!("dedicated:{partition_id}:{device_id}"),
        };
        
        Ok(ProcessWingOutput {
            output: mock_output,
            exit_code: 0,
            stdout: format!("Executed wing on dedicated accelerator partition {partition_id} device {device_id}"),
            stderr: String::new(),
            execution_time_ms: 2.0,
        })
    }
}

fn execute_hardware_partition(
    input: &ProcessWingInput,
    partition_id: &str,
    device_id: &str,
) -> Result<ProcessWingOutput, ExecutorError> {
    // Parse partition type from environment or config
    let partition_type = std::env::var("HARDWARE_PARTITION_TYPE")
        .unwrap_or_else(|_| "dedicated".to_string());
    
    match partition_type.as_str() {
        "cuda-mps" => {
            #[cfg(feature = "cuda")]
            return cuda_mps::execute_cuda_mps(input, partition_id, device_id);
            #[cfg(not(feature = "cuda"))]
            return Err(ExecutorError::Gpu("CUDA feature not enabled. Compile with --features cuda".into()));
        }
        "vulkan" => {
            #[cfg(feature = "ash")]
            return vulkan_compute::execute_vulkan(input, partition_id, device_id);
            #[cfg(not(feature = "ash"))]
            return Err(ExecutorError::Gpu("Vulkan feature not enabled. Compile with --features ash".into()));
        }
        "dedicated" => {
            dedicated_accelerator::execute_dedicated(input, partition_id, device_id)
        }
        _ => Err(ExecutorError::Gpu(format!(
            "Unknown hardware partition type: {partition_type}. Supported: cuda-mps, vulkan, dedicated"
        ))),
    }
}