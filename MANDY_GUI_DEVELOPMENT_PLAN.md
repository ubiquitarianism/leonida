# Mandy Desktop Control Center — GUI Development Plan

> **Purpose:** Build a polished desktop GUI for [ProjectLeonida/mandy](https://github.com/ProjectLeonida/mandy) that brings setup, validation, building, launching, mods, saves, and diagnostics into one place.
>
> **Recommended approach:** Evolve the repository's existing Python/Tkinter launcher instead of introducing a second launcher or a large UI framework. Preserve the current `Launcher.bat` entry point and reuse the repository's existing scripts and modules.

---

## 1. Project context

The repository already contains a desktop launcher:

- `Launcher.bat` starts `tools/leonida_launcher.py` on Windows.
- `tools/leonida_launcher.py` provides Play, Settings, Mods, Saves, Tools, and About pages.
- `make_game.py` performs the package check and build stages.
- `tools/leonida_mods.py` provides the existing mod manager.
- `tools/leonida_saves.py` provides the existing save-management functionality.
- `docs/launcher.md` documents the current launcher and its supported operations.

The new GUI should therefore be an **upgrade and expansion** of the current launcher, not a parallel application with duplicated logic.

### Important project limitations to show in the interface

The application must explain the current state of the project accurately and must not promise that every system or runtime is playable. The repository README currently distinguishes between two runtime paths:

1. The Direct3D 12 runtime under `d3d12/` has reached an open-world state on Windows, but performance and compatibility are still experimental.
2. The recompiled runtime under `runtime/` has progressed through startup and story loading, but the README does not report a verified first world frame for that runtime.

Show this as a dated project-status notice sourced from the repository README. Do not hard-code changing project status into the GUI; read it from a small maintained status file or link to the README so it can be updated without a code release.

The repository requires the user's own game package with its files already readable on disk. The GUI must not provide game downloads, keys, decryption, protection bypasses, or instructions for obtaining copyrighted game files.

---

## 2. Goals

### Primary goals

- Give users one clear place to configure the project, validate their game folder, build the required files, and launch the available runtime.
- Make long build operations understandable through stage-level progress, live logs, elapsed time, and safe cancellation.
- Keep the interface usable while Python, PowerShell, shell scripts, or build tools are running.
- Reuse the repository's existing scripts and tested Python modules wherever possible.
- Preserve existing launcher settings and existing command-line behavior.
- Make disk-space requirements and the difference between the output folder and work folder obvious.
- Display clear warnings when a requested feature is unsupported on the user's platform or is not ready in the project.

### Non-goals

- Reimplementing the game runtime or shader translator in the GUI.
- Replacing `make_game.py`, `scripts/play-d3d12.ps1`, `scripts/play-linux.sh`, or the existing mod/save tools.
- Downloading, decrypting, unpacking protected content, or bypassing platform protections.
- Claiming guaranteed FPS, compatibility, or playability.
- Editing packed or encrypted save data.
- Bundling game assets, game screenshots, proprietary artwork, or publisher logos.

---

## 3. Recommended technology

| Area | Recommendation | Reason |
|---|---|---|
| Language | Python 3.12+ | Matches the repository's current launcher and build requirements. |
| GUI | Tkinter + `ttk` | Already used by the project; available with standard Python installations and avoids a large dependency. |
| Styling | Custom `ttk.Style` theme and reusable widgets | Enables a consistent modern dark interface without adding a UI framework. |
| Background work | `subprocess.Popen`, `threading`, `queue` | Keeps builds and runtime processes away from the GUI thread. |
| Settings | JSON | Compatible with the existing `%APPDATA%/Leonida/launcher.json` configuration. |
| Tests | `unittest` | Fits the existing Python test layout and has no extra dependency. |

**Do not add PySide6, Electron, a browser runtime, or a web server for the first version.** Those options increase installation size and maintenance work without being necessary for the required desktop controls.

### Platform scope

- **Windows 11:** Primary design and test target.
- **Linux:** Keep existing support where the repository's scripts support it. Hide or disable Windows-only tools instead of presenting nonfunctional buttons.
- **macOS:** Do not promise full GUI/runtime support unless the relevant repository scripts and dependencies have been verified. The Python tools may work independently of runtime support.

---

## 4. Application structure and navigation

Use a fixed left navigation rail and a flexible content area. The window should remember its size and last selected page.

### Main navigation

1. **Dashboard**
2. **Game Setup**
3. **Build Center**
4. **Play**
5. **Graphics & Runtime**
6. **Mods**
7. **Saves**
8. **Tools**
9. **Diagnostics**
10. **Settings**
11. **About**

The current launcher already implements several of these sections. Reuse and reorganize those implementations rather than creating a second copy of their functionality.

### Global status strip

Display these items consistently:

- Selected game folder and validation state.
- Selected output and work folders.
- Current task: Idle, Checking, Building, Launching, Running, Stopping, Failed, or Completed.
- Active runtime process state, when applicable.
- A link/button to open the current log folder.

Only one build or runtime launch should run at a time unless the underlying scripts explicitly support safe concurrency. Disable conflicting actions while a task is active.

---

## 5. Screen specifications

### 5.1 Dashboard

The dashboard should answer: **Is my setup ready, what can I do next, and what is currently running?**

Suggested layout:

- Header: `Mandy Control Center` and repository version/commit, when available.
- Large primary action: **Continue Setup**, **Build**, or **Play**, chosen from the actual current setup state.
- Status cards:
  - Game package: Not selected / Found / Invalid.
  - Build output: Missing / Partial / Ready (only when verified).
  - Work folder: path and available disk space.
  - Runtime: Available / Missing / Platform unsupported.
- Recent activity: last build result, last launch result, and most recent log.
- Quick actions: Open Logs, Open Output Folder, Open Screenshots, Check Game Files.
- Project status notice: a concise, date-stamped note with a link to the repository README.

Do not infer that a build is complete just because a directory exists. Verify the required files for the selected operation.

### 5.2 Game Setup

Provide folder pickers and validation feedback for:

- **Game folder:** must contain `game_scarlett_master.exe` and the expected package layout.
- **Output folder (`--out`):** destination for the built port/runtime files.
- **Work folder (`--work`):** intermediate shader and asset data. This may be placed on a separate drive.
- **Data folder:** optional runtime data/save location, preserving the existing launcher's behavior.
- **Mods folder:** defaults to the repository's `mods/` directory.

Actions:

- Browse for a folder.
- Find game copies using `make_game.py --find`.
- Select a folder using the repository's existing `--pick` behavior where appropriate.
- Run a quick package check.
- Run a full SHA-256 check only after a clear warning that it takes longer.
- Open the selected folder in the operating system's file manager.

Validation must be read-only. Never modify the user's game package. Show exactly which check failed and how to fix the selected path.

### 5.3 Build Center

Make the build workflow the main addition to the current launcher.

#### Build stages

Offer the stages supported by `make_game.py`:

| Stage | GUI label | Description |
|---|---|---|
| `check` | Validate files | Compare the package's expected file names and sizes. |
| `shaders` | Prepare shaders | Run the repository's shader extraction/translation stage. |
| `assets` | Export assets | Export supported models, textures, and audio data. |
| `runtime` | Build runtime | Build the runtime using the repository's supported platform/toolchain path. |
| `pipelines` | Prepare GPU pipelines | Run the pipeline preparation/check stage when its prerequisites are available. |

The GUI must use the actual supported stage names and pass them to `make_game.py --stages`. Do not invent new stages.

#### Build options

- Stage selection with a clear explanation of dependencies.
- Worker count (`--jobs`) with a conservative default based on available CPU cores.
- Optional full SHA-256 package verification (`--full-check`).
- Optional WAV conversion (`--wav`) with a prominent disk-space warning because WAV output is substantially larger.
- Advanced options, initially collapsed:
  - `--runtime-build DIR`
  - `--wave64-emulate`
  - `--descriptor-hash`
  - `--loop-limit`
  - `--compact-spirv`
  - `--no-float-controls`
  - `--bary-fallback`
  - `--texel-offset-fallback`
  - `--f16-interp-fallback`
  - `--mesh-compute`
  - `--native-mesh`
  - `--pc-tail-ubo`

Only expose advanced options whose meaning and valid values can be read from the current `make_game.py` argument parser. Keep an **Advanced** label beside them and provide the original CLI option in each tooltip.

#### Storage warnings

The README's current estimates are approximately 30 GB for shader work and approximately 370 GB for exported assets. Treat these as estimates, not guarantees. Before starting:

- Display the selected drive's free space.
- Display the repository's current estimate, with a link to the README.
- Warn if the selected work drive is likely too small.
- Explain that the game package is read-only and the work/output folders receive generated files.
- Explain that the work folder can be located on a different drive from the output folder.

Never automatically delete old work or output files to make space.

#### Live build view

While a build runs, show:

- Overall progress only when it can be derived from the build program's actual progress output.
- One row per selected stage: Waiting, Running, Completed, Failed, or Skipped.
- Current stage output and a scrollable live log.
- Elapsed time and the command's exit code after completion.
- Buttons: **Open Full Log**, **Copy Log**, and **Cancel Build**.

If the build does not emit reliable percentage information, use an indeterminate progress indicator. Never fabricate a percentage or ETA.

Cancellation must terminate the child process safely and must not delete partial build files. A later build should be able to resume according to the behavior already implemented by `make_game.py`.

### 5.4 Play

Retain the existing launcher behavior and make the primary action visually prominent.

- Large **PLAY** button.
- **STOP** replaces PLAY while the launched process is active.
- Show the current runtime executable/path and selected game folder.
- Show a readable status line and live process output.
- Prevent a second launch while the first is active.
- Confirm before stopping the runtime and its child processes.
- Preserve the existing command construction for Windows and Linux. Do not rebuild launch commands independently in several UI widgets.

### 5.5 Graphics & Runtime

Organize the settings already supported by `tools/leonida_launcher.py` and the corresponding launch script:

- DLSS on/off.
- Ray tracing.
- Smooth Motion.
- Shader precompilation.
- Series X mode.
- Quality mode.
- Resolution selection using only values supported by the current launch script (the current launcher lists default, 2160, 1440, 1080, 720, 540, and 480).
- Capture mode.
- Runtime logging.

For every option:

- Show its exact effect and the command-line switch it controls.
- Show platform availability.
- Keep defaults aligned with the current repository launcher.
- Do not claim that enabling a setting improves performance on every GPU.

Add a **Restore Defaults** action and a preview of the effective launch configuration before Play.

### 5.6 Mods

Reuse `tools/leonida_mods.py` and the existing `mod.json` format.

- List detected mods with name, folder, enabled/disabled state, and supported swap count/details when available.
- Toggle a mod using the repository's existing enable/disable function.
- Refresh the list and show invalid/malformed mod metadata as an error rather than silently ignoring it.
- Provide an **Open Mod Manager** action to launch the existing manager.
- Show the active mods directory and allow the user to change it in Game Setup or Settings.

Do not invent a new manifest format or overwrite unknown mod metadata.

### 5.7 Saves

Reuse `tools/leonida_saves.py` and the behavior documented in `docs/launcher.md`.

Supported UI actions should include the repository's existing safe operations:

- List save containers and their readable metadata.
- Back up one container or all containers.
- Restore from a backup, first backing up the current data and requiring confirmation.
- Duplicate a container under a new name.
- Move a deleted container into the existing backup/deleted area instead of permanently deleting it.
- Import/export a supported individual blob where the existing tool permits it.
- View supported plain data as read-only text/hex/strings.

Rules:

- Disable save operations while the game is running.
- Clearly distinguish readable data from packed/compressed/encrypted data.
- Do not attempt to decrypt, decompress, guess formats, repair checksums, or edit opaque story-save blobs.
- Warn when importing a blob of a different size if the existing tool reports that the header may no longer match.
- Require explicit confirmation for restore, duplicate, import, and delete/move actions.

### 5.8 Tools

Expose existing repository tools only:

- Open build logs.
- Open runtime logs.
- Open screenshots/captures.
- Open the output and work folders.
- Open the mod manager.
- Open the project's progress dashboard, if generated by the existing tool.
- Provide the existing NVIDIA DLSS download/rebuild actions only on the platforms where their scripts are supported, with source and size information before download.

Do not create a generic command prompt or arbitrary shell-command execution feature.

### 5.9 Diagnostics

Provide non-invasive checks and actionable reports:

- OS and architecture.
- Python version.
- Repository path and commit hash, when available.
- CPU model/core count, when detectable.
- GPU name and driver version, when detectable using available local tools.
- Vulkan loader/device availability where relevant.
- Required executables and expected output files.
- Free disk space on game, work, and output drives.
- Latest build/runtime log locations.

Diagnostics should be best-effort. Missing optional detection tools must not crash the app. Clearly label detected values versus unknown values. Do not upload system information or game paths anywhere.

### 5.10 Settings

- Persist all settings to the existing configuration location: `%APPDATA%/Leonida/launcher.json` on Windows and the existing XDG config location on Linux.
- Maintain backward compatibility with the current keys: `game_dir`, `data_dir`, `mods_dir`, `dlss`, `ray_tracing`, `smooth_motion`, `precompile`, `series_x`, `quality`, `resolution`, `capture`, and `log`.
- Add new keys for output directory, work directory, build stages, job count, and other new GUI preferences.
- Preserve unknown JSON keys when saving, so a future version does not erase settings it does not understand.
- Validate all paths and enum values when loading settings.
- Provide **Save**, **Apply**, and **Restore Defaults** actions.

### 5.11 About

Include:

- Short description of Mandy/Leonida.
- Explicit unofficial-project disclaimer.
- Link to the repository, README, documentation, issue tracker, and license.
- Project credits sourced from the repository's own credit files.
- Current commit/build information when available.

Do not use publisher logos or imply official endorsement.

---

## 6. Visual design

### Direction

Use a clean, restrained, dark desktop interface inspired by modern PC game launchers, while keeping the repository's own identity. Avoid copying another launcher or using copyrighted GTA artwork.

### Suggested palette

| Role | Suggested value |
|---|---|
| Main background | `#0E1116` |
| Navigation background | `#0A0D12` |
| Card/panel | `#151A22` |
| Borders/dividers | `#232B36` |
| Main text | `#E6E9EE` |
| Secondary text | `#8A94A3` |
| Accent | `#22C3B0` |
| Error | `#D9534F` |
| Warning | `#D9A441` |

These are starting values, not mandatory constants. Centralize them in one theme module.

### Layout guidelines

- Default window size: approximately 1240 × 800.
- Minimum usable size: approximately 960 × 640.
- Resizable window; content must not overlap when resized.
- Navigation rail: approximately 210–230 px on desktop.
- Use consistent spacing, card padding, headings, and status indicators.
- Use Segoe UI on Windows with a sensible system-font fallback elsewhere.
- Use icons only when available locally or from a dependency already included in the project.
- Use the newest user-generated runtime screenshot as an optional dashboard background; otherwise use a gradient or plain panel.
- Do not bundle game screenshots, models, textures, logos, or other copyrighted assets.

### Interaction rules

- Every long-running operation must show progress or an indeterminate running state.
- Never freeze the main window during a subprocess call, file scan, or log read.
- Tooltips should explain technical settings in plain language.
- Destructive or irreversible actions require confirmation.
- Disabled controls must include a visible reason, such as “Unavailable on Linux” or “Stop the game before managing saves.”
- Use clear error dialogs with the relevant log path and a copyable technical detail section.

---

## 7. Architecture

Keep UI code separate from process execution, repository integration, and settings.

Suggested structure:

```text
 tools/
   mandy_gui.py                 # GUI entry point; may replace/refactor leonida_launcher.py
   mandy_gui/
     __init__.py
     app.py                     # Window lifecycle and page routing
     theme.py                   # Colors, fonts, reusable styles
     widgets.py                 # Cards, path rows, status labels, log panel
     pages/
       dashboard.py
       setup.py
       build.py
       play.py
       graphics.py
       mods.py
       saves.py
       tools.py
       diagnostics.py
       settings.py
       about.py
     services/
       settings_store.py        # Load/save/migrate launcher.json
       repository.py            # Paths and repository feature detection
       build_service.py         # make_game.py command construction and process
       runtime_service.py       # Existing launch-script integration
       process_manager.py       # One active task, output queue, cancellation
       disk_service.py          # Free space and path validation
       diagnostics_service.py   # Local, best-effort system checks
       mods_service.py          # Adapter around leonida_mods.py
       saves_service.py         # Adapter around leonida_saves.py
 tests/
   test_mandy_gui_settings.py
   test_mandy_gui_commands.py
   test_mandy_gui_process.py
   test_mandy_gui_paths.py
 docs/
   gui.md
```

This is a suggested organization. If the current launcher is refactored in place, retain its existing tested helper functions where practical instead of moving everything at once.

### Process-management design

- Build commands as Python argument lists; do not concatenate user paths into shell command strings.
- Use `subprocess.Popen` with `shell=False`.
- Set `cwd` to the repository root for project scripts.
- Capture stdout/stderr and forward decoded log lines through a thread-safe queue.
- Update Tkinter widgets only from the main GUI thread.
- Keep a handle to the active process and its process group/tree.
- On Stop/Cancel, request confirmation and terminate the process tree using platform-appropriate logic already established by the current launcher.
- Record the exit code and preserve the complete log.
- Never silently swallow exceptions or report success before the child process exits successfully.

### Repository integration rules

- Build: call `make_game.py` with its documented CLI options.
- Windows Play: use `scripts/play-d3d12.ps1` through the existing launch path.
- Linux Play: use `scripts/play-linux.sh` through the existing launch path.
- Mods: use `tools/leonida_mods.py` and its current metadata format.
- Saves: use `tools/leonida_saves.py` and its supported safety checks.
- Existing Tools: reuse the current scripts and platform checks.

Avoid importing `make_game.py` as a library unless it exposes a stable public function for that purpose. Running it as a subprocess is the safer initial integration because it preserves the script's own environment, logging, and CLI validation.

---

## 8. Build command mapping

The GUI should construct the following command shape, adjusting optional arguments based on the user's selections:

```text
python -I make_game.py
  --game <game-folder>
  --out <output-folder>
  --work <work-folder>
  --stages <comma-separated-supported-stages>
  --jobs <worker-count>
```

Optional flags must map exactly to the current parser:

```text
--check
--full-check
--wav
--runtime-build <directory>
--wave64-emulate <supported-value>
--descriptor-hash <supported-value>
--loop-limit <integer>
--compact-spirv
--no-float-controls
--bary-fallback
--texel-offset-fallback
--f16-interp-fallback
--mesh-compute
--native-mesh
--pc-tail-ubo
--no-tui
```

The GUI should normally include `--no-tui` because it provides its own progress and log view. Confirm supported choices and defaults from the current `make_game.py` before exposing an option. Do not hard-code a CLI option that no longer exists.

The command preview should show the executable and arguments in a copyable, properly quoted display. Execute the argument list directly rather than passing the preview string to a shell.

---

## 9. Configuration and migration

Keep the existing settings file and add a version field, for example `schema_version`.

Suggested new settings:

```json
{
  "schema_version": 2,
  "game_dir": "",
  "data_dir": "",
  "mods_dir": "",
  "out_dir": "",
  "work_dir": "",
  "build_stages": ["check", "shaders", "runtime", "pipelines"],
  "jobs": 4,
  "dlss": true,
  "ray_tracing": false,
  "smooth_motion": false,
  "precompile": true,
  "series_x": false,
  "quality": false,
  "resolution": "",
  "capture": false,
  "log": false,
  "window_width": 1240,
  "window_height": 800,
  "last_page": "Dashboard"
}
```

The values above are examples. Use the current launcher's actual defaults when migrating an existing configuration. Loading an older configuration must fill in new defaults without losing the user's current values. Saving must preserve keys that a newer or third-party version may have added.

---

## 10. Error handling and safety

- Validate folder paths before invoking a repository operation.
- Check that the expected game executable exists before allowing Play.
- Do not run a command if its script or executable is missing.
- Show the original exception and log location in an expandable error detail area.
- Keep the game folder read-only from the GUI's perspective.
- Do not auto-delete generated work files, shader caches, logs, mods, saves, or user data.
- Never upload game paths, save files, system diagnostics, or logs.
- Do not add arbitrary PowerShell, CMD, terminal, or Python command execution to the interface.
- Respect the repository license and preserve required notices when redistributing modified files.
- Disable save operations during gameplay and retain the existing backup-before-restore safeguards.

---

## 11. Testing plan

### Unit tests

- Load, normalize, migrate, and save old/new settings.
- Preserve unknown configuration keys.
- Validate supported build stages and arguments.
- Verify command construction for Windows and Linux.
- Verify paths containing spaces and non-ASCII characters.
- Verify invalid directories and missing executables are rejected.
- Verify the process manager forwards output and exit codes correctly.
- Verify cancellation does not delete partial build output.
- Verify unsupported platform actions are hidden or disabled.

### Integration tests

- Use a fake Python script to simulate a long build, progress output, failure, and cancellation.
- Use temporary game/output/work directories; never use copyrighted game files in automated tests.
- Test mods and saves against the repository's existing fixtures and test helpers.
- Verify the current `Launcher.bat` still starts the GUI.

### Manual acceptance checks

- Windows 11 at 100%, 125%, and 150% display scaling.
- Window resizing down to the minimum supported size.
- Folder paths containing spaces and non-English characters.
- A build that takes several minutes while the UI remains responsive.
- Failed build with a readable log and actionable error.
- Stop runtime and stop build actions.
- Settings migration from an existing `%APPDATA%/Leonida/launcher.json`.
- Linux behavior for supported pages and Windows-only tool visibility.

---

## 12. Implementation phases

### Phase 1 — Foundation

- Review the current launcher and its tests.
- Refactor shared settings, path, and process functions into reusable services without changing behavior.
- Add the new navigation shell, theme, status strip, and responsive layout.
- Keep `Launcher.bat` working.

### Phase 2 — Setup and Build Center

- Add output/work folder selection and disk-space checks.
- Integrate `make_game.py` stages and documented options.
- Add asynchronous output, stage status, cancellation, command preview, and full logs.
- Add tests for build command construction and process lifecycle.

### Phase 3 — Runtime and existing features

- Move the existing Play and Graphics settings into the new layout.
- Integrate the current Mods and Saves services.
- Add Tools and platform-aware availability states.

### Phase 4 — Dashboard and Diagnostics

- Add live setup/build/runtime status cards.
- Add local system and GPU diagnostics.
- Add recent activity and links to logs, screenshots, and output folders.

### Phase 5 — Polish and release

- Complete keyboard navigation, tooltips, resizing, display-scaling checks, and error handling.
- Run unit/integration tests on supported platforms.
- Update `docs/launcher.md` and add `docs/gui.md` with screenshots created from the GUI itself (no bundled game assets).
- Add a changelog entry and document any new dependencies (ideally none).

---

## 13. Definition of done

The GUI is ready for an initial release when:

- `Launcher.bat` opens the upgraded GUI on a supported Windows setup.
- Existing users retain their saved launcher settings.
- Users can select and validate their own game folder without modifying it.
- Users can select separate output and work folders.
- Users can start each supported build stage through the existing `make_game.py` interface.
- Long-running tasks do not freeze the window and show truthful progress/log information.
- Build/runtime processes can be stopped safely.
- Play settings generate the same supported commands as the existing launcher.
- Mods and Saves use the repository's existing tools and file formats.
- Unsupported platform/runtime functionality is clearly labeled.
- No game files, keys, copyrighted artwork, or protected-content bypass functionality are bundled or downloaded.
- Automated tests pass and the GUI documentation matches the implementation.

---

## 14. Repository references to inspect during implementation

- [Repository README](https://github.com/ProjectLeonida/mandy/blob/main/README.md)
- [Current launcher documentation](https://github.com/ProjectLeonida/mandy/blob/main/docs/launcher.md)
- [Current launcher entry point](https://github.com/ProjectLeonida/mandy/blob/main/Launcher.bat)
- [Current Tkinter launcher](https://github.com/ProjectLeonida/mandy/blob/main/tools/leonida_launcher.py)
- [Build driver and CLI](https://github.com/ProjectLeonida/mandy/blob/main/make_game.py)
- [Mod manager](https://github.com/ProjectLeonida/mandy/blob/main/tools/leonida_mods.py)
- [Save manager](https://github.com/ProjectLeonida/mandy/blob/main/tools/leonida_saves.py)
- [Play script for Windows](https://github.com/ProjectLeonida/mandy/blob/main/scripts/play-d3d12.ps1)
- [Play script for Linux](https://github.com/ProjectLeonida/mandy/blob/main/scripts/play-linux.sh)

**Implementation note:** Re-check these files on the exact branch/commit being modified. The repository is actively developed, so CLI flags, runtime status, and supported options can change.
