//! Tests against the installed NVIDIA driver. Each one prints why and
//! returns early when there is no driver or no device.

use std::process::Command;
use std::sync::Arc;

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{
    Context, Device, DeviceBuffer, Event, HostImage, HostImageMut, Point, Region, Stream,
};
use ferricast_nvcodec::encode::Session;
use ferricast_nvcodec::loader::{CudaLibrary, CuvidLibrary, NvencLibrary};
use ferricast_nvcodec::version::{
    API_OVERRIDE_VAR, AnyApi, Api, ApiVersion, ApiVisitor, NvencApi, NvencVersion,
};

/// The primary context of device 0, or `None` (with the reason printed)
/// without a usable NVIDIA driver.
fn context() -> Option<Arc<Context>> {
    let opened = CudaLibrary::get()
        .and_then(|_| Device::get(0))
        .and_then(Context::primary);
    match opened {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping: no usable NVIDIA CUDA driver: {error}");
            None
        }
    }
}

/// The driver's newest NVENC API, or `None` (with the reason printed).
fn nvenc_driver_max() -> Option<NvencVersion> {
    match NvencLibrary::get().and_then(NvencLibrary::max_supported_version) {
        Ok(version) => Some(version),
        Err(error) => {
            eprintln!("skipping: no usable NVENC driver: {error}");
            None
        }
    }
}

#[test]
fn devices_describe_themselves() {
    let Some(context) = context() else { return };
    let cuda = CudaLibrary::get().unwrap();
    eprintln!("CUDA driver {}", cuda.driver_version().unwrap());
    let devices = Device::all().unwrap();
    assert!(!devices.is_empty());
    for device in devices {
        let bus_id = device.pci_bus_id().unwrap();
        eprintln!(
            "{device:?}: {} at {bus_id}, {}, SM {}, {} MiB",
            device.name().unwrap(),
            device.uuid().unwrap(),
            device.compute_capability().unwrap(),
            device.total_memory().unwrap() >> 20
        );
        assert_eq!(Device::by_pci_bus_id(bus_id).unwrap(), device);
        assert!(device.compute_capability().unwrap().major >= 3);
    }
    let memory = context.memory_info().unwrap();
    assert!(memory.free <= memory.total && memory.total > 0);
    assert!(matches!(
        Device::get(u32::MAX),
        Err(Error::DeviceOutOfRange { .. })
    ));
}

#[test]
fn nvdec_library_loads() {
    if context().is_none() {
        return;
    }
    let cuvid = CuvidLibrary::get().unwrap();
    eprintln!(
        "{}: reconfigure {}, decode status {}",
        CuvidLibrary::file_name(),
        cuvid.reconfigure_decoder.is_some(),
        cuvid.get_decode_status.is_some()
    );
}

/// A test pattern no two rows or columns of which are alike.
fn pattern(width: usize, rows: usize, pitch: usize) -> Vec<u8> {
    let mut image = vec![0xee; pitch * rows];
    for (row, line) in image.chunks_mut(pitch).enumerate() {
        for (x, byte) in line.iter_mut().take(width).enumerate() {
            *byte = u8::try_from((x * 7 + row * 13) % 251).unwrap();
        }
    }
    image
}

#[test]
fn pitched_round_trip_host_device_host() {
    let Some(context) = context() else { return };
    // An odd width, so the driver's pitch differs from it.
    let (width, rows) = (1000, 37);
    let host_pitch = 1024;
    let source = pattern(width, rows, host_pitch);

    let mut device = DeviceBuffer::pitched(&context, width, rows).unwrap();
    assert!(device.pitch() >= width, "pitch {}", device.pitch());
    device.fill(0x55).unwrap();
    device
        .upload(
            device.region(),
            HostImage {
                data: &source,
                pitch: host_pitch,
            },
        )
        .unwrap();

    // Tightly packed on the way back.
    let mut back = vec![0; width * rows];
    device
        .download(
            device.region(),
            HostImageMut {
                data: &mut back,
                pitch: width,
            },
        )
        .unwrap();
    for (row, (got, want)) in back
        .chunks(width)
        .zip(source.chunks(host_pitch))
        .enumerate()
    {
        assert_eq!(got, &want[..width], "row {row}");
    }
}

#[test]
fn async_copies_and_device_to_device() {
    let Some(context) = context() else { return };
    let (width, rows) = (640, 48);
    let source = pattern(width, rows, width);
    let stream = Stream::new(&context).unwrap();
    let done = Event::new(&context).unwrap();

    let mut first = DeviceBuffer::pitched(&context, width, rows).unwrap();
    let mut second = DeviceBuffer::pitched(&context, width, rows).unwrap();
    second.fill_async(0, &stream).unwrap();
    // SAFETY: `source` outlives the stream synchronisation below.
    unsafe {
        first
            .upload_async(
                first.region(),
                HostImage {
                    data: &source,
                    pitch: width,
                },
                &stream,
            )
            .unwrap();
    }
    // Bottom half of `first` to the top of `second`.
    let half = Region::new(width, rows / 2).at(Point {
        x_bytes: 0,
        row: rows / 2,
    });
    second
        .copy_from_async(Point::default(), &first, half, &stream)
        .unwrap();
    let mut back = vec![0xaa; width * rows];
    // SAFETY: `back` is not touched until the event below completes.
    unsafe {
        second
            .download_async(
                second.region(),
                HostImageMut {
                    data: &mut back,
                    pitch: width,
                },
                &stream,
            )
            .unwrap();
    }
    done.record(&stream).unwrap();
    done.synchronize().unwrap();
    assert!(done.is_complete().unwrap());
    assert!(stream.is_idle().unwrap());

    let (top, bottom) = back.split_at(width * rows / 2);
    assert_eq!(top, &source[width * rows / 2..]);
    assert!(bottom.iter().all(|&byte| byte == 0));

    // Synchronous device-to-device with an x offset.
    let mut third = DeviceBuffer::pitched(&context, width, rows).unwrap();
    third.fill(0).unwrap();
    third
        .copy_from(
            Point {
                x_bytes: 10,
                row: 1,
            },
            &first,
            Region::new(100, 2),
        )
        .unwrap();
    let mut row = vec![0; width];
    third
        .download(
            Region::new(width, 1).at(Point { x_bytes: 0, row: 2 }),
            HostImageMut {
                data: &mut row,
                pitch: width,
            },
        )
        .unwrap();
    assert_eq!(&row[10..110], &source[width..width + 100]);
    assert!(row[..10].iter().chain(&row[110..]).all(|&byte| byte == 0));
}

#[test]
fn copies_are_bounds_checked() {
    let Some(context) = context() else { return };
    let mut buffer = DeviceBuffer::pitched(&context, 64, 8).unwrap();
    let small = vec![0; 64 * 7];
    assert!(matches!(
        buffer.upload(
            buffer.region(),
            HostImage {
                data: &small,
                pitch: 64,
            }
        ),
        Err(Error::CopyOutOfBounds { side: "source", .. })
    ));
    assert!(matches!(
        buffer.upload(
            Region::new(65, 1),
            HostImage {
                data: &small,
                pitch: 65,
            }
        ),
        Err(Error::CopyOutOfBounds {
            side: "destination",
            ..
        })
    ));
    // An empty copy touches nothing.
    buffer
        .upload(
            Region::new(0, 8),
            HostImage {
                data: &[],
                pitch: 0,
            },
        )
        .unwrap();
}

#[test]
fn nvenc_negotiates_the_newest_version_the_driver_accepts() {
    let Some(driver_max) = nvenc_driver_max() else {
        return;
    };
    let requested = ApiVersion::requested().unwrap();
    let negotiated = ApiVersion::negotiate();
    eprintln!(
        "driver NVENC API {driver_max}, {API_OVERRIDE_VAR}={requested:?}: negotiated {negotiated:?}"
    );
    if requested.is_none() {
        let expected = ApiVersion::ALL
            .into_iter()
            .rev()
            .find(|version| version.version() <= driver_max);
        assert_eq!(negotiated.ok(), expected);
    }
}

#[test]
fn every_version_the_driver_accepts_opens_a_session() {
    struct OpenSession<'a>(&'a Arc<Context>);
    impl ApiVisitor for OpenSession<'_> {
        type Output = Result<(), Error>;
        fn visit<A: NvencApi>(self) -> Self::Output {
            let api = Api::<A>::load()?;
            let session = Session::open(api, self.0)?;
            assert!(!session.as_ptr().is_null());
            assert_eq!(session.api().version(), A::VERSION);
            drop(session);
            Ok(())
        }
    }

    let Some(context) = context() else { return };
    let Some(driver_max) = nvenc_driver_max() else {
        return;
    };
    for version in ApiVersion::ALL {
        let opened = version.dispatch(OpenSession(&context));
        eprintln!("NVENC API {version}: {opened:?}");
        if version.version() <= driver_max {
            opened.unwrap();
            assert_eq!(AnyApi::load(version).unwrap().version(), version);
        } else {
            let error = opened.unwrap_err();
            eprintln!("  {error}");
            assert!(
                matches!(error, Error::ApiNotSupportedByDriver { requested, .. } if requested == version),
                "{error}"
            );
            assert!(
                error
                    .to_string()
                    .contains(version.minimum_driver().for_this_os())
            );
        }
    }
}

/// Runs in a child process with `FERRICAST_NVENC_API` set (see
/// `forced_versions_through_the_environment`); passes trivially otherwise.
#[test]
fn child_negotiates_with_the_override() {
    let Some(value) = std::env::var_os(API_OVERRIDE_VAR) else {
        return;
    };
    let value = value.to_string_lossy().into_owned();
    let Some(driver_max) = nvenc_driver_max() else {
        return;
    };
    let negotiated = AnyApi::negotiate();
    eprintln!("{API_OVERRIDE_VAR}={value}: {negotiated:?}");
    let forced = value
        .parse::<NvencVersion>()
        .ok()
        .and_then(ApiVersion::from_version);
    match (forced, negotiated) {
        (None, Err(Error::InvalidApiOverride { value: seen, .. })) => assert_eq!(seen, value),
        (Some(forced), Ok(api)) if forced.version() <= driver_max => {
            assert_eq!(api.version(), forced);
        }
        (Some(forced), Err(error @ Error::ApiNotSupportedByDriver { .. }))
            if forced.version() > driver_max =>
        {
            let message = error.to_string();
            assert!(
                message.contains(&format!("requested by {API_OVERRIDE_VAR}")),
                "{message}"
            );
            assert!(
                message.contains(forced.minimum_driver().for_this_os()),
                "{message}"
            );
        }
        (forced, outcome) => panic!("{forced:?} with driver {driver_max}: {outcome:?}"),
    }
}

#[test]
fn forced_versions_through_the_environment() {
    if nvenc_driver_max().is_none() {
        return;
    }
    let values = ApiVersion::ALL
        .iter()
        .map(ToString::to_string)
        .chain(["10.0".to_owned(), "garbage".to_owned()]);
    for value in values {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child_negotiates_with_the_override",
                "--nocapture",
            ])
            .env(API_OVERRIDE_VAR, &value)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("{}", stderr.trim());
        assert!(output.status.success(), "{API_OVERRIDE_VAR}={value}");
        assert!(
            stderr.contains(&format!("{API_OVERRIDE_VAR}={value}:")),
            "the child test did not run for {value}"
        );
    }
}

/// `DeviceBuffer::upload` and `DeviceBuffer::fill` run after the work
/// already queued on the crate's (non-blocking) streams and are complete,
/// for every stream, when they return.
#[test]
fn blocking_upload_and_fill_are_ordered_with_streams() {
    let Some(context) = context() else { return };
    let (width, rows) = (8192, 2048);
    let mut target = DeviceBuffer::pitched(&context, width, rows).unwrap();
    let mut source = DeviceBuffer::pitched(&context, width, rows).unwrap();
    source.fill(0x22).unwrap();
    let stream = Stream::new(&context).unwrap();
    let mut host = vec![0; width * rows];
    let mut expect = |buffer: &DeviceBuffer, value: u8, what: &str| {
        let region = buffer.region();
        buffer
            .download(
                region,
                HostImageMut {
                    data: &mut host,
                    pitch: width,
                },
            )
            .unwrap();
        assert!(host.iter().all(|&byte| byte == value), "{what}");
    };
    // Enough queued copies that a call racing them would finish first.
    let queue_copies = |target: &mut DeviceBuffer, source: &DeviceBuffer| {
        for _ in 0..32 {
            target
                .copy_from_async(Point { x_bytes: 0, row: 0 }, source, source.region(), &stream)
                .unwrap();
        }
    };
    let pattern = vec![0x33; width * rows];
    for _ in 0..3 {
        queue_copies(&mut target, &source);
        target.fill(0x11).unwrap();
        expect(&target, 0x11, "fill lands after the queued copies");

        queue_copies(&mut target, &source);
        target
            .upload(
                target.region(),
                HostImage {
                    data: &pattern,
                    pitch: width,
                },
            )
            .unwrap();
        expect(&target, 0x33, "upload lands after the queued copies");

        // Complete on return: a stream reading right after sees it.
        target.fill(0x44).unwrap();
        source
            .copy_from_async(Point { x_bytes: 0, row: 0 }, &target, target.region(), &stream)
            .unwrap();
        stream.synchronize().unwrap();
        expect(&source, 0x44, "a stream sees the finished fill");
        source.fill(0x22).unwrap();
    }
}
