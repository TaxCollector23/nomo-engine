/** Standalone RNG parity probe: run with `node scripts/pyrandom-probe.mjs` after esbuild-bundling pyrandom.ts. */
import { PyRandom, samplePoisson } from "../src/simulation/pyrandom";

const r = new PyRandom(20261027);
console.log("random:", [r.random(), r.random(), r.random()].map((v) => v.toFixed(14)).join(" "));

const g = new PyRandom(20261027);
console.log("gauss:", [g.gauss(0, 1), g.gauss(0, 1), g.gauss(0, 1)].map((v) => v.toFixed(12)).join(" "));

const nv = new PyRandom(20261027);
console.log("normalvariate:", [nv.normalvariate(0, 1), nv.normalvariate(0, 1)].map((v) => v.toFixed(12)).join(" "));

const e = new PyRandom(20261027);
console.log("expovariate:", [e.expovariate(5), e.expovariate(5), e.expovariate(5)].map((v) => v.toFixed(12)).join(" "));

const ln = new PyRandom(20261027);
console.log("lognormvariate:", [ln.lognormvariate(1.0, 0.5), ln.lognormvariate(1.0, 0.5)].map((v) => v.toFixed(12)).join(" "));

const u = new PyRandom(20261027);
console.log("uniform:", [u.uniform(2, 5), u.uniform(2, 5)].map((v) => v.toFixed(12)).join(" "));

const c = new PyRandom(20261027);
console.log("choice:", [c.choice([7, 11, 13]), c.choice([7, 11, 13]), c.choice([7, 11, 13])].join(" "));

const z = new PyRandom(0);
console.log("seed0:", [z.random(), z.random()].map((v) => v.toFixed(16)).join(" "));

const ga = new PyRandom(20261027);
console.log("gamma(2,3):", [ga.gammavariate(2.0, 3.0), ga.gammavariate(2.0, 3.0)].map((v) => v.toFixed(12)).join(" "));

const gb = new PyRandom(20261027);
console.log("gamma(0.5,3):", [gb.gammavariate(0.5, 3.0), gb.gammavariate(0.5, 3.0)].map((v) => v.toFixed(12)).join(" "));

const bp = new PyRandom(2 ** 35 + 7);
console.log("bigseed:", bp.random().toFixed(16));

const p = new PyRandom(7);
console.log("poisson(3):", [samplePoisson(p, 3), samplePoisson(p, 3), samplePoisson(p, 3)].join(" "));
