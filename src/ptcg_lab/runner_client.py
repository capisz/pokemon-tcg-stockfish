"""Control the existing local supervisor; never starts a second worker pool."""
from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def call(base, path, body=None):
    if urlparse(base).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('The learning supervisor must be a local loopback server')
    data = None if body is None else json.dumps(body).encode()
    request = Request(base.rstrip('/')+'/api/learning/'+path, data=data,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start')
    start.add_argument('--seed', type=int, default=42)
    start.add_argument('--allow-sleep', action='store_true')
    start.add_argument('--config', type=Path, help='Optional bounded configuration JSON')
    commands.add_parser('list')
    for name in ('status', 'pause', 'resume', 'stop'):
        commands.add_parser(name).add_argument('run_id')
    soak = commands.add_parser('soak', help='Record actual progress; does not start or resume learning')
    soak.add_argument('run_id')
    soak.add_argument('--seconds', type=int, default=86400)
    soak.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'start':
            config = json.loads(args.config.read_text()) if args.config else {}
            config.update(seed=args.seed, keepAwake=not args.allow_sleep)
            result = call(args.url, 'runs', config)
        elif args.command == 'list':
            result = call(args.url, 'runs')
        elif args.command == 'soak':
            if not 1 <= args.seconds <= 7*86400:
                raise ValueError('Soak duration must be 1 second to 7 days')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            errors = 0
            # Exclusive creation protects previous evidence from replacement.
            with args.output.open('x') as output:
                while True:
                    elapsed = time.monotonic()-started
                    try:
                        state = call(args.url, 'runs/'+args.run_id)
                        entry = {'elapsedSeconds': elapsed, 'observedAt': time.time(), 'run': state}
                    except (URLError, HTTPError) as exc:
                        errors += 1
                        entry = {'elapsedSeconds': elapsed, 'observedAt': time.time(), 'error': str(exc)}
                    output.write(json.dumps(entry, allow_nan=False)+'\n'); output.flush()
                    if elapsed >= args.seconds:
                        break
                    time.sleep(min(30, args.seconds-elapsed))
            result = {'status': 'observation-completed', 'seconds': time.monotonic()-started,
                      'connectionErrors': errors, 'output': str(args.output.resolve()),
                      'note': 'Inspect recorded progress and pauses; elapsed observation alone is not a successful learning soak.'}
        else:
            result = call(args.url, 'runs/'+args.run_id)
            if args.command != 'status':
                result = call(args.url, f'runs/{args.run_id}/control/{args.command}',
                              {'revision': result['revision'], 'requestId': uuid.uuid4().hex})
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
