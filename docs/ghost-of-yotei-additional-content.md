# Ghost of Yōtei: infinite loading spinner and additional-content registrations

This note covers the **initial infinite loading spinner** observed with Ghost of Yōtei
(PPSA26344, v02.002.000) on the AMD BC-250 native GPU backend. It is separate from
the **black screen after the opening cutscene** discussed in
[issue #17](https://github.com/superkeka/KytyPS5-Native/issues/17).

## Observed behavior

The game remained at the loading wheel without progressing. An entitlement trace of
the affected launch showed that queries for two labels returned
`0x817D0007` (`NO_ENTITLEMENT`):

- `GHOST2APP0000000`
- `GHOST2BASE000000`

With those content entries registered, both queries returned `0x00000000`.
The game then progressed past the original loading wheel.

Importantly, this also worked using the **unmodified upstream emulator** at commit
`47857b4`: the missing content registrations, rather than a native GPU change,
accounted for the observed difference at this loading stage.

## Configuring existing additional-content support

KytyPS5-Native already supports per-game additional-content entries; a new
game-specific entitlement override in the emulator is **not** required.

In `kyty_launcher`, select the game, open its context menu, and choose
**Additional Content...**. Add the corresponding labels for content available
to your installation. For content with extracted data, specify the associated
data directory; the dialog also supports license-only registrations.

For command-line troubleshooting, the existing `--addcont` option accepts a
16-character label. The tested launch included:

```sh
./kyty_emulator --game /path/to/game/eboot.bin \
  --addcont GHOST2APP0000000 \
  --addcont GHOST2BASE000000
```

A Steam shortcut made by the Decky plugin starts the game through `kytyps5-run.sh`,
which reads the game's add-ons from `~/.config/KytyPS5-Native/addcont/<title id>`,
a line each (`<label>` or `<label>=<folder with its data>`, `#` starts a comment):

```sh
mkdir -p ~/.config/KytyPS5-Native/addcont
printf '%s\n' GHOST2APP0000000 GHOST2BASE000000 > ~/.config/KytyPS5-Native/addcont/PPSA26344
```

Without the two entries the game shows a black screen at a steady 30 FPS and never
leaves it; the session's log then has no `add-ons from ...` line under its `==` header.

These flags register labels with the emulator; they **do not install missing
game data or verify ownership**. Only register content you are authorized to
use. Content names and requirements may differ across releases/editions.

## Evidence and scope

Three controlled observations were recorded on the BC-250:

1. **No additional-content entries:** both queried labels returned
   `NO_ENTITLEMENT`; the loading wheel persisted.
2. **Same diagnostic binary, entries registered:** both queried labels returned
   success; the game progressed past the spinner.
3. **Original upstream release `47857b4`, entries registered:** the game
   progressed as well, without the experimental emulator modifications.

The investigation also encountered other Ghost of Yōtei-related labels in
community discussions. In particular, `GHOSTYOTEIDUPLEX` was mentioned,
but **it was not required by the controlled two-label comparison** and is not
presented here as a verified requirement.

This is a **configuration-dependent compatibility finding**, not proof that
all endless loading screens are caused by missing add-ons. It does not fix
the separate post-cutscene video transition or any later gameplay crash.
When investigating a similar stall, compare the missing-label results and
verify the selected game's actual additional-content registrations before
changing GPU or kernel settings.

See [issue #17](https://github.com/superkeka/KytyPS5-Native/issues/17)
for the original report and subsequent video-decoder investigation.
