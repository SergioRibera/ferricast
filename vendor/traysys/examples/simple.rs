use std::sync::Arc;

use traysys::{TrayIcon, TrayMenuBuilder};

fn main() {
    let icon =
        TrayIcon::from_png(include_bytes!("../../../bin/client/resources/icon.png")).unwrap();
    let tray_menu = TrayMenuBuilder::default()
        .name("Simple Example")
        .icon(icon.clone())
        .add_label("name", "App Menu", Some(icon))
        .add_item("coso", ("Say Hello", || println!("Hello, World!")))
        .add_separator()
        .add_submenu("file", true, "File", None, |menu| {
            menu.add_item(
                "submenu open",
                ("Open", false, || println!("File Open clicked")),
            )
            .add_item(
                "submenu save",
                ("Save", true, || println!("File Save clicked")),
            )
            .add_submenu("submenu submenu", true, "Things", None, |menu| {
                menu.add_label("submenu submenu label", "Test nested", None)
                    .add_separator()
                    .add_item(
                        "submenu submenu item",
                        ("Nested item", true, || println!("Nested item clicked")),
                    )
                    // add other nested
                    .add_submenu(
                        "submenu submenu submenu",
                        false,
                        "More Things",
                        None,
                        |menu| {
                            menu.add_label("submenu submenu submenu label", "Test nested", None)
                                .add_separator()
                                .add_item(
                                    "submenu submenu submenu item",
                                    ("Nested item", true, || println!("Nested item clicked")),
                                )
                        },
                    )
            })
        })
        .build();
    let tray_menu = Arc::new(tray_menu);

    {
        let tray_menu = tray_menu.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(15));
            tray_menu.update_label("name", "New App Name");
            tray_menu.update_label("coso", "Saying Hi!!!!");
            tray_menu.update_submenu("submenu submenu", |menu| {
                menu.new_name("New Label")
                    .add_item(
                        "submenu submenu item",
                        ("New Nested item", true, || {
                            println!("Modified Nested item clicked")
                        }),
                    )
                    .update_submenu("submenu submenu submenu", true, "New More Things", |menu| {
                        menu.add_label("submenu submenu submenu label", "Test nested", None)
                            .add_separator()
                            .add_item(
                                "submenu submenu submenu item",
                                ("Nested item", true, || println!("Nested item clicked")),
                            )
                    })
            });
        });
    }

    tray_menu.start();

    #[cfg(not(target_os = "macos"))]
    std::thread::park();
}
