/** SplitMix32 stream; rejection sampling avoids modulo bias in permutations. */
export class SeededRandom {
  constructor(public state: number, private onAmbientRandom?: (value:number)=>void) { this.state >>>= 0; }
  uint32(): number {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    return (z ^ (z >>> 15)) >>> 0;
  }
  float(): number { return this.uint32() / 0x100000000; }
  int(n: number): number {
    if (!Number.isSafeInteger(n) || n < 1 || n > 0x100000000) throw new Error('Invalid random bound');
    const limit = Math.floor(0x100000000 / n) * n;
    let x: number; do { x = this.uint32(); } while (x >= limit);
    return x % n;
  }
  shuffle(n: number): number[] {
    const a = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = this.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  /** Upstream has rare synchronous RNG paths outside chance prompts. Scope them to this worker. */
  scoped<T>(fn: () => T): T {
    const previous = Math.random; Math.random = () => {const value=this.float();this.onAmbientRandom?.(value);return value;};
    try { return fn(); } finally { Math.random = previous; }
  }
}
