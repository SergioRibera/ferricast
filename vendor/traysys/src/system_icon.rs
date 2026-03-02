use crate::TrayIcon;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemIcon {
    Lock,
    LockOpen,
    ChevronRight,
    ChevronLeft,
    ChevronUp,
    ChevronDown,
    Checkmark,
    XMark,
    Plus,
    Minus,
    Gear,
    ArrowRight,
    ArrowLeft,
    Circle,
    CircleFilled,
    Person,
    Globe,
    Star,
    StarFilled,
    Folder,
    Document,
    Trash,
    QuestionMark,
    ExclamationMark,
    Dashboard,
    Power,
    PowerOff,
    Restart,
    Shield,
    Network,
    Bell,
    BellSlash,
    Eye,
    House,
    Download,
    Upload,
    Paperclip,
    Calendar,
    Clock,
    Info,
    InfoCircle,
    WarningTriangle,
    ErrorCircle,
    List,
}

impl SystemIcon {
    #[cfg(target_os = "macos")]
    fn sf_symbol_name(&self) -> &'static str {
        match self {
            SystemIcon::Lock => "lock",
            SystemIcon::LockOpen => "lock.open",
            SystemIcon::ChevronRight => "chevron.right",
            SystemIcon::ChevronLeft => "chevron.left",
            SystemIcon::ChevronUp => "chevron.up",
            SystemIcon::ChevronDown => "chevron.down",
            SystemIcon::Checkmark => "checkmark",
            SystemIcon::XMark => "xmark",
            SystemIcon::Plus => "plus",
            SystemIcon::Minus => "minus",
            SystemIcon::Gear => "gearshape",
            SystemIcon::ArrowRight => "arrow.right",
            SystemIcon::ArrowLeft => "arrow.left",
            SystemIcon::Circle => "circle",
            SystemIcon::CircleFilled => "circle.fill",
            SystemIcon::Person => "person",
            SystemIcon::Globe => "globe",
            SystemIcon::Star => "star",
            SystemIcon::StarFilled => "star.fill",
            SystemIcon::Folder => "folder",
            SystemIcon::Document => "doc",
            SystemIcon::Trash => "trash",
            SystemIcon::QuestionMark => "questionmark",
            SystemIcon::ExclamationMark => "exclamationmark",
            SystemIcon::Dashboard => "square.grid.2x2",
            SystemIcon::Power => "power",
            SystemIcon::PowerOff => "power.circle",
            SystemIcon::Restart => "arrow.clockwise",
            SystemIcon::Shield => "shield",
            SystemIcon::Network => "network",
            SystemIcon::Bell => "bell",
            SystemIcon::BellSlash => "bell.slash",
            SystemIcon::Eye => "eye",
            SystemIcon::House => "house",
            SystemIcon::Download => "arrow.down.circle",
            SystemIcon::Upload => "arrow.up.circle",
            SystemIcon::Paperclip => "paperclip",
            SystemIcon::Calendar => "calendar",
            SystemIcon::Clock => "clock",
            SystemIcon::Info => "info",
            SystemIcon::InfoCircle => "info.circle",
            SystemIcon::WarningTriangle => "exclamationmark.triangle",
            SystemIcon::ErrorCircle => "exclamationmark.circle",
            SystemIcon::List => "list.bullet",
        }
    }

    pub fn unicode(&self) -> &'static str {
        match self {
            SystemIcon::Lock => "🔒",
            SystemIcon::LockOpen => "🔓",
            SystemIcon::ChevronRight => "›",
            SystemIcon::ChevronLeft => "‹",
            SystemIcon::ChevronUp => "^",
            SystemIcon::ChevronDown => "v",
            SystemIcon::Checkmark => "✓",
            SystemIcon::XMark => "✕",
            SystemIcon::Plus => "+",
            SystemIcon::Minus => "−",
            SystemIcon::Gear => "⚙",
            SystemIcon::ArrowRight => "→",
            SystemIcon::ArrowLeft => "←",
            SystemIcon::Circle => "○",
            SystemIcon::CircleFilled => "●",
            SystemIcon::Person => "👤",
            SystemIcon::Globe => "🌐",
            SystemIcon::Star => "☆",
            SystemIcon::StarFilled => "★",
            SystemIcon::Folder => "📁",
            SystemIcon::Document => "📄",
            SystemIcon::Trash => "🗑",
            SystemIcon::QuestionMark => "?",
            SystemIcon::ExclamationMark => "!",
            SystemIcon::Dashboard => "▦",
            SystemIcon::Power => "⏻",
            SystemIcon::PowerOff => "⏼",
            SystemIcon::Restart => "⟲",
            SystemIcon::Shield => "🛡",
            SystemIcon::Network => "🌐",
            SystemIcon::Bell => "🔔",
            SystemIcon::BellSlash => "🔕",
            SystemIcon::Eye => "👁",
            SystemIcon::House => "🏠",
            SystemIcon::Download => "⬇",
            SystemIcon::Upload => "⬆",
            SystemIcon::Paperclip => "📎",
            SystemIcon::Calendar => "📅",
            SystemIcon::Clock => "🕐",
            SystemIcon::Info => "ℹ",
            SystemIcon::InfoCircle => "ⓘ",
            SystemIcon::WarningTriangle => "⚠",
            SystemIcon::ErrorCircle => "⊗",
            SystemIcon::List => "☰",
        }
    }

    #[cfg(target_os = "macos")]
    pub fn to_tray_icon(&self) -> Option<TrayIcon> {
        use crate::macos::sf_symbols::create_sf_symbol_icon;
        create_sf_symbol_icon(self.sf_symbol_name(), 13.0)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn to_tray_icon(&self) -> Option<TrayIcon> {
        None
    }

    pub fn with_label(&self, label: &str) -> String {
        format!("{} {}", label, self.unicode())
    }

    pub fn with_label_padded(&self, label: &str, padding: usize) -> String {
        let pad = " ".repeat(padding);
        format!("{}{}{}", label, pad, self.unicode())
    }
}
