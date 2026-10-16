//! Preferências da interface do ClothEvo, separadas dos arquivos de projeto de vestuário.
//! Layouts e temas podem evoluir sem acoplar a UI ao modelo de moldes ou ao solver.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UiCustomization {
    pub version: u32,
    pub theme: ThemeSettings,
    pub toolbar_buttons: Vec<CustomToolbarButton>,
    pub split_vertical: bool,
    pub split_swapped: bool,
}

impl Default for UiCustomization {
    fn default() -> Self {
        Self {
            version: 1,
            theme: ThemeSettings::default(),
            toolbar_buttons: Vec::new(),
            split_vertical: false,
            split_swapped: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ThemeSettings {
    pub panel_bg: [u8; 3],
    pub window_bg: [u8; 3],
    pub deep_bg: [u8; 3],
    pub muted_bg: [u8; 3],
    pub text: [u8; 3],
    pub accent: [u8; 3],
    pub selection: [u8; 3],
    pub button_bg: [u8; 3],
    pub button_hover: [u8; 3],
    pub border: [u8; 3],
    /// Arredondamento dos componentes em pixels lógicos.
    pub corner_radius: u8,
    /// Espaçamento entre controles em pixels lógicos.
    pub spacing: f32,
    /// Escala global dos widgets.
    pub ui_scale: f32,
}

impl Default for ThemeSettings {
    fn default() -> Self {
        // Mantém o aspecto da interface anterior como preset inicial.
        Self {
            panel_bg: [15, 19, 35],
            window_bg: [23, 27, 45],
            deep_bg: [8, 12, 25],
            muted_bg: [28, 33, 54],
            text: [236, 232, 219],
            accent: [255, 180, 65],
            selection: [205, 57, 135],
            button_bg: [31, 39, 65],
            button_hover: [53, 61, 98],
            border: [54, 63, 91],
            corner_radius: 4,
            spacing: 8.0,
            ui_scale: 1.0,
        }
    }
}

impl ThemeSettings {
    pub fn graphite() -> Self {
        Self {
            panel_bg: [34, 35, 36],
            window_bg: [27, 28, 29],
            deep_bg: [20, 21, 22],
            muted_bg: [43, 44, 45],
            text: [224, 226, 228],
            accent: [42, 184, 173],
            selection: [24, 108, 103],
            button_bg: [48, 49, 50],
            button_hover: [64, 66, 67],
            border: [68, 70, 72],
            corner_radius: 5,
            spacing: 7.0,
            ui_scale: 1.0,
        }
    }

    pub fn amethyst() -> Self {
        Self {
            accent: [181, 126, 255],
            selection: [104, 66, 154],
            button_bg: [43, 36, 60],
            button_hover: [65, 51, 88],
            border: [79, 65, 103],
            ..Self::default()
        }
    }

    pub fn apply_named_preset(&mut self, name: &str) {
        let scale = self.ui_scale;
        *self = match name {
            "Graphite" => Self::graphite(),
            "Amethyst" => Self::amethyst(),
            _ => Self::default(),
        };
        self.ui_scale = scale;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiCommand {
    ToggleSimulation,
    SingleStep,
    ResetSimulation,
    NewProject,
    OpenProject,
    SaveProject,
    ImportAvatar,
    ToggleAvatar,
    ToggleAvatarWireframe,
    ToggleAvatarCollision,
    Show3dViewport,
    Show2dViewport,
    ShowSplitViewport,
    ToggleMetrics,
}

impl UiCommand {
    pub const ALL: &'static [(Self, &'static str)] = &[
        (Self::ToggleSimulation, "Iniciar / pausar simulação"),
        (Self::SingleStep, "Avançar um passo"),
        (Self::ResetSimulation, "Reiniciar simulação"),
        (Self::NewProject, "Novo projeto"),
        (Self::OpenProject, "Abrir projeto"),
        (Self::SaveProject, "Salvar projeto"),
        (Self::ImportAvatar, "Importar avatar"),
        (Self::ToggleAvatar, "Exibir / ocultar avatar"),
        (Self::ToggleAvatarWireframe, "Alternar wireframe do avatar"),
        (Self::ToggleAvatarCollision, "Alternar colisão do avatar"),
        (Self::Show3dViewport, "Abrir 3D Viewport"),
        (Self::Show2dViewport, "Abrir 2D Viewport"),
        (Self::ShowSplitViewport, "Exibir 3D + 2D"),
        (Self::ToggleMetrics, "Mostrar / ocultar métricas"),
    ];

    pub fn label(self) -> &'static str {
        Self::ALL.iter().find_map(|(command, label)| (*command == self).then_some(*label))
            .unwrap_or("Comando")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomToolbarButton {
    pub id: u64,
    pub label: String,
    pub command: UiCommand,
    pub visible: bool,
}

impl Default for CustomToolbarButton {
    fn default() -> Self {
        Self {
            id: 1,
            label: "Simular".into(),
            command: UiCommand::ToggleSimulation,
            visible: true,
        }
    }
}

pub fn configuration_path() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    let root = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| "A variável APPDATA não está definida.".to_owned())?;

    #[cfg(not(target_os = "windows"))]
    let root = if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(path)
    } else {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| "Não foi possível localizar a pasta de configuração do usuário.".to_owned())?;
        home.join(".config")
    };

    Ok(root.join("ClothEvo").join("ui-customization.json"))
}

impl UiCustomization {
    pub fn load() -> (Self, Option<String>) {
        let path = match configuration_path() {
            Ok(path) => path,
            Err(error) => return (Self::default(), Some(error)),
        };
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return (Self::default(), None);
            }
            Err(error) => {
                return (Self::default(), Some(format!("Não foi possível ler preferências: {error}")));
            }
        };

        match serde_json::from_str::<Self>(&contents) {
            Ok(mut settings) => {
                settings.version = 1;
                settings.theme.spacing = settings.theme.spacing.clamp(2.0, 20.0);
                settings.theme.ui_scale = settings.theme.ui_scale.clamp(0.75, 1.75);
                settings.theme.corner_radius = settings.theme.corner_radius.min(16);
                (settings, None)
            }
            Err(error) => (
                Self::default(),
                Some(format!("Preferências inválidas; valores padrão foram carregados: {error}")),
            ),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = configuration_path()?;
        let parent = path.parent().ok_or_else(|| "Caminho de configuração inválido.".to_owned())?;
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Não foi possível criar a pasta de preferências: {error}"))?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|error| format!("Não foi possível serializar as preferências: {error}"))?;
        std::fs::write(&path, json)
            .map_err(|error| format!("Não foi possível salvar as preferências: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn customization_round_trips_through_json() {
        let mut config = UiCustomization::default();
        config.theme.accent = [11, 102, 188];
        config.split_vertical = true;
        config.toolbar_buttons.push(CustomToolbarButton {
            id: 8,
            label: "Alternar vista".into(),
            command: UiCommand::ShowSplitViewport,
            visible: true,
        });

        let json = serde_json::to_string(&config).unwrap();
        let loaded: UiCustomization = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.theme.accent, [11, 102, 188]);
        assert!(loaded.split_vertical);
        assert_eq!(loaded.toolbar_buttons.len(), 1);
        assert_eq!(loaded.toolbar_buttons[0].command, UiCommand::ShowSplitViewport);
    }

    #[test]
    fn missing_fields_use_defaults_for_future_migrations() {
        let loaded: UiCustomization = serde_json::from_str("{}").unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.theme.ui_scale, 1.0);
        assert!(loaded.toolbar_buttons.is_empty());
    }
}
