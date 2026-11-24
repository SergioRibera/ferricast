//! Macros that declare a codec facade: a named type implementing the
//! codec trait over a `Selecting*` body and a registry function.
//!
//! ```ignore
//! ferricast_codec::video_decoder_facade! {
//!     /// VP9 decoder over whichever backend its configuration selects.
//!     pub struct Vp9Decoder: Codec::Vp9, registry = vp9_registry, syntax = None;
//! }
//! ```
//!
//! Each facade gets `new()`, `registry()` (the backends of this build
//! for its job), `backends()` (their capability report) and `backend()`
//! (the one running now).

#[doc(hidden)]
pub use ferricast_core as __core;

/// Declares a [`ferricast_core::VideoEncoder`] facade.
#[macro_export]
macro_rules! video_encoder_facade {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident: $codec:expr, registry = $registry:path;
    ) => {
        $(#[$meta])*
        $vis struct $name($crate::SelectingVideoEncoder);

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self($crate::SelectingVideoEncoder::new($registry()))
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// An encoder that selects its backend on `configure`.
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            /// The backends of this build for this job, in automatic order.
            #[must_use]
            pub fn registry() -> $crate::VideoEncoderRegistry {
                $registry()
            }

            /// Every backend for this job with whether it can run here.
            #[must_use]
            pub fn backends() -> ::std::vec::Vec<$crate::BackendReport> {
                $registry().report()
            }

            /// The backend encoding now, once configured.
            #[must_use]
            pub fn backend(
                &self,
            ) -> ::core::option::Option<(
                $crate::facade::__core::BackendId,
                $crate::facade::__core::Acceleration,
            )> {
                self.0.backend()
            }
        }

        impl $crate::facade::__core::VideoEncoder for $name {
            const CODEC: $crate::facade::__core::Codec = $codec;

            fn configure(
                &mut self,
                config: &$crate::facade::__core::EncoderConfig,
            ) -> $crate::facade::__core::CodecResult<()> {
                self.0.configure(config)
            }

            fn encode(
                &mut self,
                frame: $crate::facade::__core::CapturedFrame,
            ) -> $crate::facade::__core::CodecResult<$crate::facade::__core::EncodedFrame> {
                self.0.encode(frame)
            }

            fn flush(
                self,
            ) -> $crate::facade::__core::CodecResult<
                ::std::vec::Vec<$crate::facade::__core::EncodedFrame>,
            > {
                self.0.flush()
            }

            fn get_headers(
                &mut self,
            ) -> $crate::facade::__core::CodecResult<::std::vec::Vec<u8>> {
                self.0.headers()
            }

            fn request_keyframe(&mut self) {
                self.0.request_keyframe();
            }

            fn set_bitrate_kbps(&mut self, kbps: u32) -> $crate::facade::__core::CodecResult<()> {
                self.0.set_bitrate_kbps(kbps)
            }
        }
    };
}

/// Declares a [`ferricast_core::VideoDecoder`] facade. `syntax` names
/// the parameter sets replayed on a mid-stream hand-over
/// ([`crate::bitstream::NalSyntax`]), `None` for self-contained packets.
#[macro_export]
macro_rules! video_decoder_facade {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident: $codec:expr, registry = $registry:path, syntax = $syntax:expr;
    ) => {
        $(#[$meta])*
        $vis struct $name($crate::SelectingVideoDecoder);

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self($crate::SelectingVideoDecoder::new($registry(), $syntax))
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// A decoder that selects its backend on `configure`.
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            /// The backends of this build for this job, in automatic order.
            #[must_use]
            pub fn registry() -> $crate::VideoDecoderRegistry {
                $registry()
            }

            /// Every backend for this job with whether it can run here.
            #[must_use]
            pub fn backends() -> ::std::vec::Vec<$crate::BackendReport> {
                $registry().report()
            }

            /// The backend decoding now, once configured.
            #[must_use]
            pub fn backend(
                &self,
            ) -> ::core::option::Option<(
                $crate::facade::__core::BackendId,
                $crate::facade::__core::Acceleration,
            )> {
                self.0.backend()
            }
        }

        impl $crate::facade::__core::VideoDecoder for $name {
            const CODEC: $crate::facade::__core::Codec = $codec;

            fn configure(
                &mut self,
                config: &$crate::facade::__core::DecoderConfig,
            ) -> $crate::facade::__core::CodecResult<()> {
                self.0.configure(config)
            }

            fn decode(
                &mut self,
                frame: $crate::facade::__core::EncodedFrame,
            ) -> $crate::facade::__core::CodecResult<
                ::core::option::Option<$crate::facade::__core::CapturedFrame>,
            > {
                self.0.decode(frame)
            }

            fn flush(
                &mut self,
            ) -> $crate::facade::__core::CodecResult<
                ::std::vec::Vec<$crate::facade::__core::CapturedFrame>,
            > {
                self.0.flush()
            }
        }
    };
}

/// Declares a [`ferricast_core::AudioEncoder`] facade.
#[macro_export]
macro_rules! audio_encoder_facade {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident, registry = $registry:path;
    ) => {
        $(#[$meta])*
        $vis struct $name($crate::SelectingAudioEncoder);

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self($crate::SelectingAudioEncoder::new($registry()))
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// An encoder that selects its backend on `configure`.
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            /// The backends of this build for this job, in automatic order.
            #[must_use]
            pub fn registry() -> $crate::AudioEncoderRegistry {
                $registry()
            }

            /// Every backend for this job with whether it can run here.
            #[must_use]
            pub fn backends() -> ::std::vec::Vec<$crate::BackendReport> {
                $registry().report()
            }

            /// The backend encoding now, once configured.
            #[must_use]
            pub fn backend(
                &self,
            ) -> ::core::option::Option<(
                $crate::facade::__core::BackendId,
                $crate::facade::__core::Acceleration,
            )> {
                self.0.backend()
            }
        }

        impl $crate::facade::__core::AudioEncoder for $name {
            fn configure(
                &mut self,
                config: &$crate::facade::__core::AudioEncoderConfig,
            ) -> $crate::facade::__core::CodecResult<()> {
                self.0.configure(config)
            }

            fn encode(
                &mut self,
                frame: &$crate::facade::__core::AudioFrame,
            ) -> $crate::facade::__core::CodecResult<()> {
                self.0.encode(frame)
            }

            fn take_output(&mut self) -> ::std::vec::Vec<$crate::facade::__core::AudioFrame> {
                self.0.take_output()
            }

            fn codec_config(&self) -> ::std::vec::Vec<u8> {
                self.0.codec_config()
            }

            fn flush(
                self,
            ) -> $crate::facade::__core::CodecResult<
                ::std::vec::Vec<$crate::facade::__core::AudioFrame>,
            > {
                self.0.flush()
            }
        }
    };
}

/// Declares a [`ferricast_core::AudioDecoder`] facade.
#[macro_export]
macro_rules! audio_decoder_facade {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident: $codec:expr, registry = $registry:path;
    ) => {
        $(#[$meta])*
        $vis struct $name($crate::SelectingAudioDecoder);

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self($crate::SelectingAudioDecoder::new($registry()))
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// A decoder that selects its backend on `configure`.
            #[must_use]
            pub fn new() -> Self {
                Self::default()
            }

            /// The backends of this build for this job, in automatic order.
            #[must_use]
            pub fn registry() -> $crate::AudioDecoderRegistry {
                $registry()
            }

            /// Every backend for this job with whether it can run here.
            #[must_use]
            pub fn backends() -> ::std::vec::Vec<$crate::BackendReport> {
                $registry().report()
            }

            /// The backend decoding now, once configured.
            #[must_use]
            pub fn backend(
                &self,
            ) -> ::core::option::Option<(
                $crate::facade::__core::BackendId,
                $crate::facade::__core::Acceleration,
            )> {
                self.0.backend()
            }
        }

        impl $crate::facade::__core::AudioDecoder for $name {
            const CODEC: $crate::facade::__core::AudioCodec = $codec;

            fn configure(
                &mut self,
                config: &$crate::facade::__core::AudioDecoderConfig,
            ) -> $crate::facade::__core::CodecResult<()> {
                self.0.configure(config)
            }

            fn decode(
                &mut self,
                frame: $crate::facade::__core::AudioFrame,
            ) -> $crate::facade::__core::CodecResult<
                ::core::option::Option<$crate::facade::__core::DecodedAudio>,
            > {
                self.0.decode(frame)
            }

            fn flush(
                &mut self,
            ) -> $crate::facade::__core::CodecResult<
                ::std::vec::Vec<$crate::facade::__core::DecodedAudio>,
            > {
                self.0.flush()
            }
        }
    };
}
