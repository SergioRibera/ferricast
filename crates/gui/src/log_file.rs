//! The trace log file, capped in size: once it grows past the limit it
//! is moved to `<name>.1` (replacing the previous one) and a new file
//! is started, so a long session or a noisy network cannot fill the
//! disk.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Size at which the log file is rotated.
pub const MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;

pub struct RotatingFile {
    path: PathBuf,
    file: File,
    written: u64,
    max_bytes: u64,
}

impl RotatingFile {
    /// Opens `path` for appending; a file already over `max_bytes` is
    /// rotated first.
    ///
    /// # Errors
    /// When the file cannot be opened or rotated.
    pub fn open(path: &Path, max_bytes: u64) -> io::Result<Self> {
        let existing = std::fs::metadata(path).map_or(0, |m| m.len());
        if existing >= max_bytes {
            std::fs::rename(path, backup_path(path))?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let written = file.metadata()?.len();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            written,
            max_bytes,
        })
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        std::fs::rename(&self.path, backup_path(&self.path))?;
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.written = 0;
        Ok(())
    }
}

fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".1");
    PathBuf::from(name)
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        if self.written > 0 && self.written.saturating_add(len) > self.max_bytes {
            // On failure keep appending to the current file: losing the
            // cap beats losing the log.
            let _ = self.rotate();
        }
        let n = self.file.write(buf)?;
        self.written = self
            .written
            .saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_past_the_cap_and_keeps_one_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ferricast.log");
        let mut log = RotatingFile::open(&path, 100).unwrap();
        for _ in 0..50 {
            log.write_all(b"0123456789\n").unwrap();
        }
        log.flush().unwrap();
        let current = std::fs::metadata(&path).unwrap().len();
        let backup = std::fs::metadata(backup_path(&path)).unwrap().len();
        assert!(current <= 100, "{current}");
        assert!(backup <= 100, "{backup}");
        let files = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(files, 2);
    }

    #[test]
    fn oversized_file_is_rotated_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ferricast.log");
        std::fs::write(&path, vec![b'x'; 500]).unwrap();
        let log = RotatingFile::open(&path, 100).unwrap();
        assert_eq!(log.written, 0);
        assert_eq!(std::fs::metadata(backup_path(&path)).unwrap().len(), 500);
    }
}
