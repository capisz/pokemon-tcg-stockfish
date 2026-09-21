# Claude update: Strategy Contract v1.1 draft

The approved v1 release remains intact at commit `b176362`. The user approved the separate v1.1 overlay on September 21, 2026. Treat v1.1 as the active strategy contract, but do not start the baseline milestone without separate authorization.

## Implemented

- Shared blind-opening policy: go first until opponent archetype is public.
- Matchup-specific opening preferences restricted to public or explicitly known-matchup contexts.
- `critical-error` keeps its internal ID and maps to display label `Blunder`.
- Structured `cardRefs` replace prose card-name scanning.
- Crustle overlay adds current Handheld Fan, Eri, Special Red Card, Lumiose City, Jumbo Ice Cream and Energy-role lines.
- Crustle's second-fortress rule is corrected: trap risk is a reason not to add one.
- Crustle mirror distinguishes Kangaskhan and Dwebble starts and corrects the Xerosic/Lillie explanation.
- Dragapult overlay adds committed-Energy Hammer timing, exceptions, Kangaskhan race/Judge timing, underprotected-Crustle pressure and Dudunsparce's role.
- Every new principle and matchup replacement has source page citations and structured card references.
- The validator cross-checks executable references against the frozen specialist and opponent manifests.

## Frozen-list decision

The authoritative Crustle list remains version 3 with two Eri and two Xerosic's Machinations. Older guide proposals for a third Eri are strategy evidence only and do not change the count.

## Still unresolved

- Hand Trimmer and Bianca's Devotion are absent from the frozen Crustle list and remain non-executable.
- Special Red Card has no approved default mirror role.
- Dragapult's mirror go-second preference is low confidence because it predates the frozen list's second Budew and third Night Stretcher.

## Next gate

The strategy approval gate is complete. A new, separately authorized bounded task may build the honest strategy baseline. No engine optimization, training or long run is authorized by this approval.
