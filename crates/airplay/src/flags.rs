use std::fmt::Display;

bitflags::bitflags! {
     #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
     pub struct Features: u64 {
        const VIDEO_SUPPORTED = 0x1;
        const PHOTO_SUPPORTED = 0x2;
        const VIDEO_FAIRPLAY = 0x4;
        const VOLUME_CONTROL = 0x8;
        const VIDEO_HTTP_LIVE_STREAM = 0x10;
        const SLIDESHOW_SUPPORTED = 0x20;
        const MIRRORING_SUPPORTED = 0x80;
        const SCREEN_ROTATE = 0x100;
        const AUDIO_SUPPORTED = 0x200;
        const AUDIO_REDUNDANCY_SUPPORTED = 0x800;
        const FAIRPLAY_SECURE_AUTH_SUPPORTED = 0x1000;
        const PHOTO_CACHING = 0x2000;
        const AUTH4 = 0x4000;

        const SUPPORT_LEGACY_PAIRING = (1 << 27);

        const HAS_UNIFIED_ADVERTISER_INFO = (1 << 26);

        const RAOP = (1 << 30);
        const SUPPORTS_VOLUME = (1 << 32);
        const AIRPLAY_VIDEO_PLAY_QUEUE = (1 << 33);
        const AIRPLAY_FROM_CLOUD = (1 << 34);

        const CORE_UTILS_PAIRING_AND_ENCRYPTION = (1 << 38);
        const BUFFERED_AUDIO = (1 << 40);
        const PTP = (1 << 41);
        const SCREEN_MULTI_CODEC = (1 << 42);
        const SYSTEM_PAIRING  = (1 << 43);

        const HK_PAIRING_AND_ACCESS_CONTROL = (1 << 46);

        const TRANSIENT_PAIRING = (1 << 48);

        const UNIFIED_PAIR_SETUP_MFI = (1 << 51);

        // There is other flags but.... who cares?
     }
}

impl Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
