#define MyAppName "EV Carbon Optimizer"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "EV Carbon Optimizer"
#define MyAppExeName "EV_Carbon_Optimizer.exe"

[Setup]
AppId={{B73520D1-16F4-4E0F-A45A-5D9CB932A9A7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\EV Carbon Optimizer
DefaultGroupName=EV Carbon Optimizer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=EV_Carbon_Optimizer_Phase2_Setup_0.2.0
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "desktopicon"; Description: "바탕 화면 바로가기 만들기"; GroupDescription: "추가 바로가기:"; Flags: unchecked

[Files]
Source: "build\windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\EV Carbon Optimizer"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\EV Carbon Optimizer"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "EV Carbon Optimizer 실행"; Flags: nowait postinstall skipifsilent
