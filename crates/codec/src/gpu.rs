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

/// One GPU of this machine and the compiled hardware backends that run on
/// it, for a GPU picker (`--list-gpus`, a settings page, the istmo plugin).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "istmo", istmo_macros::message(crate = "::istmo_core"))]
pub struct GpuReport {
    /// The GPU, as sysfs describes it.
    pub gpu: Gpu,
    /// Video encode backends that can run on it.
    pub encoders: Vec<BackendId>,
    /// Video decode backends that can run on it.
    pub decoders: Vec<BackendId>,
}

impl GpuReport {
    /// The GPUs [`ferricast_core::Gpus::scan`] finds, each with the
    /// backends of `encoders` and `decoders` that run on it (by their
    /// [`crate::Candidate::on_gpu`] affinity; whether the driver works is
    /// only known when one opens).
    #[must_use]
    pub fn scan(
        encoders: &[crate::VideoEncoderRegistry],
        decoders: &[crate::VideoDecoderRegistry],
    ) -> Vec<Self> {
        Self::of(ferricast_core::Gpus::scan(), encoders, decoders)
    }

    /// [`Self::scan`] over `gpus`.
    #[must_use]
    pub fn of(
        gpus: ferricast_core::Gpus,
        encoders: &[crate::VideoEncoderRegistry],
        decoders: &[crate::VideoDecoderRegistry],
    ) -> Vec<Self> {
        gpus.into_iter()
            .map(|gpu| Self {
                encoders: on_gpu(encoders, &gpu),
                decoders: on_gpu(decoders, &gpu),
                gpu,
            })
            .collect()
    }
}

/// The distinct backends of `registries` that run on `gpu`.
fn on_gpu<C, S: ?Sized>(registries: &[crate::Registry<C, S>], gpu: &Gpu) -> Vec<BackendId> {
    let mut backends = Vec::new();
    for candidate in registries.iter().flat_map(crate::Registry::candidates) {
        if candidate
            .gpu_affinity()
            .is_some_and(|affinity| affinity.accepts(gpu))
            && !backends.contains(&candidate.backend())
        {
            backends.push(candidate.backend());
        }
    }
    backends
}

#[cfg(test)]
mod tests {
    use ferricast_core::{
        Acceleration, Codec, CodecTarget, DevNumber, GpuKind, GpuVendor, Gpus, Operation,
    };

    use super::*;
    use crate::{Candidate, VideoDecoderRegistry, VideoEncoderRegistry};

    fn gpu(minor: u32, vendor: GpuVendor) -> Gpu {
        Gpu {
            render_node: format!("/dev/dri/renderD{minor}"),
            render_dev: DevNumber { major: 226, minor },
            primary_dev: None,
            pci: None,
            vendor,
            device_id: 0,
            driver: None,
            boot_vga: false,
            kind: GpuKind::Unknown,
        }
    }

    #[test]
    fn reports_list_the_backends_each_gpu_runs() {
        let encoders = [VideoEncoderRegistry::new(CodecTarget::encode(Codec::H264))
            .candidate(
                Candidate::new(
                    BackendId::Nvenc,
                    Acceleration::Hardware,
                    |_: &EncoderConfig| {
                        Err(CodecError::backend(
                            BackendId::Nvenc,
                            Operation::Open,
                            "test",
                        ))
                    },
                )
                .on_gpu(GpuAffinity::Nvidia),
            )
            .candidate(
                Candidate::new(
                    BackendId::Vaapi,
                    Acceleration::Hardware,
                    |_: &EncoderConfig| {
                        Err(CodecError::backend(
                            BackendId::Vaapi,
                            Operation::Open,
                            "test",
                        ))
                    },
                )
                .on_gpu(GpuAffinity::NotNvidia),
            )];
        let decoders = [
            VideoDecoderRegistry::new(CodecTarget::decode(Codec::H264)).candidate(
                Candidate::new(
                    BackendId::Vaapi,
                    Acceleration::Hardware,
                    |_: &DecoderConfig| {
                        Err(CodecError::backend(
                            BackendId::Vaapi,
                            Operation::Open,
                            "test",
                        ))
                    },
                )
                .on_gpu(GpuAffinity::Any),
            ),
        ];
        let gpus = Gpus::from(vec![
            gpu(128, GpuVendor::INTEL),
            gpu(129, GpuVendor::NVIDIA),
        ]);
        let reports = GpuReport::of(gpus, &encoders, &decoders);
        assert_eq!(reports[0].encoders, [BackendId::Vaapi]);
        assert_eq!(reports[1].encoders, [BackendId::Nvenc]);
        assert_eq!(reports[0].decoders, [BackendId::Vaapi]);
        assert_eq!(reports[1].decoders, [BackendId::Vaapi]);
    }
}
