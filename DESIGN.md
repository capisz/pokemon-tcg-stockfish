# Replay workspace design

The approved direction is a compact chess-analysis workspace: board and decision timeline on the left, analysis on the right. The game is the focal surface. There is no hero, navigation rail, or unrelated dashboard.

## Layout and hierarchy

- A quiet header identifies the local, experimental tool.
- A bounded run form contains deck choices and reproducible game settings. Deck support notes expand inline.
- Saved replay and perspective controls sit directly above the board.
- Each player board shows their active Pokémon, bench, public counts, discard, and the selected player's visible hand. The opponent sits above the selected player.
- The timeline stays directly below the board. The analysis pane places resource advantage before its breakdown, candidate actions, and opponent beliefs.
- Below 900px the layout becomes one column: board, timeline, then analysis. At phone width the active slot and bench stack and the run form uses two columns with full-width deck selectors.

## Visual system

Warm offwhite canvas, almost-white working surfaces, dark green-black text, and restrained forest green for primary actions and selected states. The system sans stack supports a dense working interface without downloading fonts. Numbers use tabular figures where comparison matters. Color reinforces labels rather than replacing them.

Replay Pokémon use compact text tiles with actual replay data. The approved Play and teaching surfaces use familiar 2D card zones and catalog-provided card images, with readable text faces when art is missing or offline. Public attachments, damage and conditions remain visible; the opponent hand shows card backs and its count. An unavailable evaluation bar is visibly neutral and explicitly labeled unavailable.

## Interaction and accessibility

Native labeled inputs, buttons, selects, range controls, and disclosure elements provide familiar keyboard behavior. All controls have visible focus, disabled, hover, and active states. Errors name recovery actions; asynchronous completion uses a polite live region. Timeline shortcuts never consume keys inside inputs or selects. Reduced motion disables loading pulses and transitions. No automatic playback or decorative entry choreography is used.

The implementation follows the settled function-first product brief and the Impeccable Operate and craft-floor guidance. The approved system font and restrained product styling take precedence over unrelated brand/display defaults.
