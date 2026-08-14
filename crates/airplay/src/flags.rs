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

        // There is other flags but.... who cares?
     }
}
