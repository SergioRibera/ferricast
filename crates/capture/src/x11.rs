use bytes::Bytes;
use ferricast_core::{FerricastError, ScreenCapture};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicPtr, Ordering},
    },
    time::Instant,
};
use tracing::info;
use xcb::{
    shm::Seg,
    x::{Format, ScreenBuf},
};

pub struct X11Capture {
    seg_id: i32,
    segment: Option<Seg>,
    conn: Option<Arc<xcb::Connection>>,
    screen: Option<ScreenBuf>,
    pixmap: Option<Format>,
    is_running: AtomicBool,
    size: (usize, usize),
    /// fps the caller asked for in `start()`. X11 has no real
    /// negotiation — we just poll on demand — so we surface the
    /// configured value so downstream paces correctly.
    fps: u32,
    buffer_ptr: AtomicPtr<u8>,
    time: Instant,
}

impl X11Capture {
    pub fn new() -> Self {
        Self {
            seg_id: 0,
            screen: None,
            segment: None,
            conn: None,
            pixmap: None,
            is_running: AtomicBool::new(false),
            size: (0, 0),
            fps: 0,
            buffer_ptr: AtomicPtr::new(core::ptr::null_mut()),
        }
    }
}

impl ScreenCapture for X11Capture {
    async fn start(
        &mut self,
        _source: ferricast_core::CaptureSource,
        config: ferricast_core::CaptureConfig,
    ) -> ferricast_core::Result<()> {
        info!("Connecting to Xserver");
        let (conn, screen_num) = xcb::Connection::connect(None)
            .map_err(|_| FerricastError::Capture("Cannot connect to server".to_string()))?;

        let screen = conn.get_setup().roots().nth(screen_num as usize).unwrap();

        let pixmap = conn
            .get_setup()
            .pixmap_formats()
            .iter()
            .find(|f| f.depth() == f.bits_per_pixel())
            .unwrap();

        let pixmap = pixmap.to_owned();

        let screen = screen.to_owned();

        let w = config.width.unwrap_or(screen.width_in_pixels() as u32) as usize;
        let h = config.height.unwrap_or(screen.height_in_pixels() as u32) as usize;

        let root = screen.root();

        let segment = conn.generate_id();

        info!("Creating shared memory");
        let seg_id = unsafe { libc::shmget(libc::IPC_PRIVATE, w * h * 4, libc::IPC_CREAT | 0o600) };

        if seg_id == -1 {
            // TODO: should try without it in case that is imposible to create one(?
            return Err(FerricastError::Capture(
                "Cannot create shared memory".to_string(),
            ));
        }

        let buffer = unsafe { libc::shmat(seg_id, core::ptr::null(), 0) } as *mut u8;

        if buffer as i32 == -1 {
            return Err(FerricastError::Capture(
                "Cannot map shared memory".to_string(),
            ));
        }

        conn.send_request(&xcb::shm::Attach {
            shmseg: segment,
            shmid: seg_id as u32,
            read_only: false,
        });

        conn.flush()
            .map_err(|_| FerricastError::Capture("Cannot flush x11 server".to_string()))?;

        info!("Connected");

        self.buffer_ptr = AtomicPtr::new(buffer);
        self.seg_id = seg_id;
        self.segment = Some(segment);
        self.conn = Some(Arc::new(conn));
        self.screen = Some(screen);
        self.is_running = AtomicBool::new(true);
        self.size = (w, h);
        self.fps = config.fps;
        self.pixmap = Some(pixmap);
        self.time = Instant::now();
        Ok(())
    }
    async fn stop(&mut self) -> ferricast_core::Result<()> {
        info!("Closing connection");
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FerricastError::Capture(
                "Trying to close recorder without starting it".to_string(),
            ));
        }

        let conn = self.conn.as_ref().unwrap();

        unsafe {
            if libc::shmctl(self.seg_id, libc::IPC_RMID, core::ptr::null_mut()) == -1 {
                return Err(FerricastError::Capture("Cannot clean segment".to_string()));
            }
        }

        conn.send_request(&xcb::shm::Detach {
            shmseg: self.segment.unwrap(),
        });

        conn.flush()
            .map_err(|_| FerricastError::Capture("Cannot flush x11 server".to_string()))?;

        self.is_running.store(false, Ordering::SeqCst);

        Ok(())
    }
    fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }
    async fn next_frame(&mut self) -> ferricast_core::Result<ferricast_core::CapturedFrame> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(FerricastError::Capture(
                "Trying to close recorder without starting it".to_string(),
            ));
        }

        let buffer = unsafe {
            std::slice::from_raw_parts(
                self.buffer_ptr.load(Ordering::Relaxed),
                self.size.0 * self.size.1 * 4,
            )
        };

        let conn = self.conn.as_ref().unwrap();
        let screen = self.screen.as_ref().unwrap();
        let format = self.pixmap.as_ref().unwrap();

        let cookie = conn.send_request(&xcb::shm::GetImage {
            drawable: xcb::x::Drawable::Window(screen.root()),
            x: 0,
            y: 0,
            width: self.size.0 as u16,
            height: self.size.1 as u16,
            plane_mask: !0,
            format: 2,
            shmseg: self.segment.unwrap(),
            offset: 0,
        });

        let _reply = conn
            .wait_for_reply(cookie)
            .map_err(|_| FerricastError::Capture("Cannot get frame from xserver".to_string()));

        Ok(ferricast_core::CapturedFrame::Cpu(
            ferricast_core::RawFrame {
                width: self.size.0 as u32,
                height: self.size.1 as u32,
                stride: format.bits_per_pixel() as u32,
                format: ferricast_core::PixelFormat::Bgra,
                data: Bytes::from(buffer.to_vec()),
                timestamp_us: self.time.elapsed().as_micros() as u64,
            },
        ))
    }
    fn get_pixel_format(&self) -> ferricast_core::PixelFormat {
        ferricast_core::PixelFormat::Bgra
    }
    fn get_screen_size(&self) -> (usize, usize) {
        self.size
    }
    fn get_framerate(&self) -> u32 {
        self.fps
    }
}
