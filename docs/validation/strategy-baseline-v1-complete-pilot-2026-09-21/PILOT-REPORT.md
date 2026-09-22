# Strategy Baseline v1 complete-policy pilot

Status: blocked after the requested P3-only pilot.

Engine fingerprint: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd`  
Engine build SHA-256: `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`

An existing environment was found at `/Users/admin/Documents/ChatGPT/Pokemon Ai project/.venv`. It uses Python 3.10.20 and PyTorch 2.14.0. The frozen guide checkpoint loaded successfully as the expected 38,532-parameter policy-only model with feature version `visible-energy-coverage-v3`; no installation or download was performed.

Only P3's eight requested pilot cells were newly run. All eight reached the 1,000-decision cap and were correctly recorded as truncated with no outcome. They are not draws. Each cell took approximately 0.56–0.65 seconds, but these are capped-loop times rather than completed-game runtimes and cannot be extrapolated to 96 terminal games.

The full 384-game runtime projection is therefore unavailable, and the 60-minute reduction rule cannot yet be decided. The previously valid P1, P2 and P4 projection remains 38.25 serial-equivalent minutes for 288 games. No runtime was imputed for P3.

No production policy, search, gameplay behavior, feature, model, training data, teaching label, or file under `data/competitive` was changed. Diagnosing or correcting P3's repeated nonterminal decisions is a separate milestone and was not performed here.
