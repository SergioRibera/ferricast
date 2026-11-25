//! The configuration's [`GpuPreference`] as the selection engine and the
//! hardware backends use it.

use ferricast_core::{
    BackendId, CodecError, CodecResult, DecoderConfig, EncoderConfig, Gpu, GpuAffinity, GpuError,
    GpuId, GpuPlacement, GpuPreference,
};

/// A video codec configuration that names a GPU preference.
pub trait GpuInput {
    /// The user's preference.
    fn gpu_preference(&self) -> &GpuPreference;
    /// The GPU the pictures come from, when known.
    fn source_gpu(&self) -> Option<&GpuId>;

    /// The machine's GPUs in the order the preference ranks them.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when the `FERRICAST_GPU`
    /// override does not parse.
    fn gpu_placement(&self) -> CodecResult<GpuPlacement> {
        GpuPlacement::resolve(self.gpu_preference(), self.source_gpu()).map_err(invalid_env)
    }

    /// The GPU `backend` (running on `affinity`'s GPUs) opens on; `None`
    /// to use the backend's own default device (GPUs unknown, or none
    /// suits it without a pin).
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] naming the preference when the
    /// pinned GPU is missing or the backend cannot run on it;
    /// [`CodecError::InvalidSelection`] for a bad override.
    fn gpu_for(&self, backend: BackendId, affinity: GpuAffinity) -> CodecResult<Option<Gpu>> {
        let placement = self.gpu_placement()?;
        placement
            .target(affinity)
            .map(Option::<&Gpu>::cloned)
            .map_err(|error| CodecError::BackendUnavailable {
                backend,
                reason: error.to_string(),
            })
    }
}

impl GpuInput for EncoderConfig {
    fn gpu_preference(&self) -> &GpuPreference {
        &self.gpu
    }

    fn source_gpu(&self) -> Option<&GpuId> {
        self.source_gpu.as_ref()
    }
}

impl GpuInput for DecoderConfig {
    fn gpu_preference(&self) -> &GpuPreference {
        &self.gpu
    }

    fn source_gpu(&self) -> Option<&GpuId> {
        None
    }
}

fn invalid_env(error: GpuError) -> CodecError {
    match error {
        GpuError::InvalidEnv {
            variable,
            value,
            reason,
        } => CodecError::InvalidSelection {
            origin: format!("the environment variable {variable}"),
            value,
            reason,
        },
        other @ (GpuError::NotFound { .. } | GpuError::Unusable { .. }) => {
            CodecError::InvalidSelection {
                origin: "the GPU preference".to_owned(),
                value: String::new(),
                reason: other.to_string(),
            }
        }
    }
}

/// A bare preference, for configuration-free probes: no frame source.
impl GpuInput for GpuPreference {
    fn gpu_preference(&self) -> &GpuPreference {
        self
    }

    fn source_gpu(&self) -> Option<&GpuId> {
        None
    }
}
