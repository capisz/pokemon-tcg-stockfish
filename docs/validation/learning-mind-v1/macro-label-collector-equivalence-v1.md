# v17 Collector Source Equivalence Audit

Status: audited compatibility for the two completed v17 source runs only.
This is not a policy-label eligibility decision, ranker acceptance, or
promotion evidence.

## Finding

The Python and TypeScript v17 manifests record different whole-module hashes
for `src/ptcg_lab/learning_mind/experiment.py`:

| Family | Source commit | Whole-module SHA-256 | Rollout identity |
| --- | --- | --- | --- |
| Python heuristic | `cc3802e38c9e8b41e49196309bf03976eac7251b` | `1b7e8df4a69cba6dcbb74d23ba310a9ed18c8ca5859ae5b9c6efbf2f8e4e646a` | `b10771fdeec178a3c226348d840911467d3de7bb9e1ba6df8f3e49032b013729` |
| TypeScript heuristic | `45620268a1f085184b58d834c0a547164c507649` | `822500b4113aea6c69abeb7fe069ce2647d561d742977f5453097c0bae6882a4` | `0fa1007ae2a04ca16ecda591d39fe9fb9c19b13544a127a311ba2ee0b8a53ba8` |

The collection surface—definitions from `transition_generator_identity`
through the complete `collect_macro_labels` function—has the same exact-byte
SHA-256 in both source revisions:

```text
a3f0225ffb11a69b87b6ee786dcaaab4e5ebe484b333424fc88bdee61e0811f9
```

The diff outside that surface is ranker input/model-fit code and the `tempfile`
import used by that code. The collector version, candidate generator identity,
rollout seed version, adaptive allocation version, and shared rollout settings
match. The candidate-generator component hashes are identical in both source
manifests. The frozen selection assigns disjoint positions to the two policy
families, and both run manifests independently verify against that selection.

## Combination safeguards

The compatibility registry permits only these two exact whole-module hashes
for this collector version and binds them to the common source-surface hash.
It rejects unknown mixed hashes and still requires identical candidate
generator identities, collector versions, and shared rollout settings.

The combined manifest retains both original source hashes, commits, manifest
hashes, dataset identities, and rollout identities. It does not rewrite either
run's identity or pool rollout counts. Confidence intervals are computed within
each position from that record's own completed outcomes; the ranker validates
the preserved rollout identity by policy family. No high-confidence policy
labels are created by this audit, and PPO, continuous operation, and trusted
promotion remain disabled.

## Reproduction

The whole-file hashes are the `labelCollectorSha256` fields in each immutable
v17 run manifest. The following source extraction is the exact common surface
used by the registry:

```bash
for commit in 45620268a1f085184b58d834c0a547164c507649 cc3802e38c9e8b41e49196309bf03976eac7251b; do
  git show "${commit}:src/ptcg_lab/learning_mind/experiment.py" \
    | awk '/^def transition_generator_identity/{inside=1} /^def collect_macro_labels/{collecting=1} inside && collecting && /^def / && !/^def collect_macro_labels/{exit} inside{print}' \
    | shasum -a 256
done
```

Both commands produce the common hash above. This allowlist is intentionally
narrow; future source revisions require a separate code review and evidence
entry rather than inheriting compatibility automatically.
