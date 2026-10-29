// Nomo Planner browser engine: hardware, cost models, model presets.
// Line-for-line port of nomo_planner (Python); verified against golden outputs (scripts/verify-engine.mjs).

export type Provenance = "spec" | "placeholder" | "calibrated" | "user";

export interface Device {
  name: string;
  peakFlops: Record<string, number>; // dtype -> dense FLOP/s
  memBytes: number;
  memBw: number; // bytes/s
  powerW: number;
  efficiency: { matmul: number; memory: number };
}

export interface Link { bw: number; latencyS: number }

export interface Cluster {
  key: string;
  label: string;
  device: Device;
  devicesPerNode: number;
  intra: Link;
  inter: Link;
  costPerDeviceHour: number;
  provenance: Record<string, Provenance | string>;
}

const eff = { matmul: 0.55, memory: 0.8 };

export const H100: Device = { name: "NVIDIA H100 SXM", peakFlops: { bf16: 989e12, fp8: 1979e12, fp32: 67e12 }, memBytes: 80e9, memBw: 3.35e12, powerW: 700, efficiency: eff };
export const A100: Device = { name: "NVIDIA A100 SXM 80GB", peakFlops: { bf16: 312e12, fp32: 19.5e12 }, memBytes: 80e9, memBw: 2.039e12, powerW: 400, efficiency: eff };
export const L4: Device = { name: "NVIDIA L4", peakFlops: { bf16: 121e12, fp8: 242e12 }, memBytes: 24e9, memBw: 0.3e12, powerW: 72, efficiency: eff };

// AMD Instinct (CDNA3/CDNA4), datasheet dense (non-sparse) rates.
// Sources: amd.com MI300X / MI325X / MI355X product pages (peak matrix performance, dense).
//   MI300X: BF16 1307.4 TF, FP8 2614.9 TF, FP32 163.4 TF; 192 GB HBM3 @ 5.3 TB/s; 750 W TBP.
//   MI325X: same compute as MI300X; 256 GB HBM3E @ 6.0 TB/s; 1000 W TBP.
//   MI355X: BF16 2500 TF, FP8 5000 TF (OCP FP8), FP32 157.3 TF; 288 GB HBM3E @ 8 TB/s; 1400 W TBP.
// MXFP6/MXFP4 (MI355X, 10.1 PF) are outside the planner's dtype set (bf16/fp8 only).
export const MI300X: Device = { name: "AMD Instinct MI300X", peakFlops: { bf16: 1307.4e12, fp8: 2614.9e12, fp32: 163.4e12 }, memBytes: 192e9, memBw: 5.3e12, powerW: 750, efficiency: eff };
export const MI325X: Device = { name: "AMD Instinct MI325X", peakFlops: { bf16: 1307.4e12, fp8: 2614.9e12, fp32: 163.4e12 }, memBytes: 256e9, memBw: 6.0e12, powerW: 1000, efficiency: eff };
export const MI355X: Device = { name: "AMD Instinct MI355X", peakFlops: { bf16: 2500e12, fp8: 5000e12, fp32: 157.3e12 }, memBytes: 288e9, memBw: 8.0e12, powerW: 1400, efficiency: eff };

export const CLUSTERS: Record<string, Cluster> = {
  h100_nvlink_ib: { key: "h100_nvlink_ib", label: "H100 SXM · NVLink + InfiniBand", device: H100, devicesPerNode: 8,
    intra: { bw: 450e9, latencyS: 3e-6 }, inter: { bw: 50e9, latencyS: 10e-6 }, costPerDeviceHour: 2.5,
    provenance: { peak: "spec", "intra.bw": "spec", "inter.bw": "spec", latency: "placeholder", price: "placeholder", efficiency: "placeholder" } },
  a100_nvlink_ib: { key: "a100_nvlink_ib", label: "A100 80GB · NVLink + InfiniBand", device: A100, devicesPerNode: 8,
    intra: { bw: 300e9, latencyS: 3e-6 }, inter: { bw: 25e9, latencyS: 10e-6 }, costPerDeviceHour: 1.5,
    provenance: { peak: "spec", "intra.bw": "spec", "inter.bw": "spec", latency: "placeholder", price: "placeholder", efficiency: "calibrated (training)" } },
  l4_pcie: { key: "l4_pcie", label: "L4 · PCIe", device: L4, devicesPerNode: 8,
    intra: { bw: 32e9, latencyS: 5e-6 }, inter: { bw: 12.5e9, latencyS: 15e-6 }, costPerDeviceHour: 0.8,
    provenance: { peak: "spec", "intra.bw": "spec (PCIe Gen4 x16)", "inter.bw": "placeholder", latency: "placeholder", price: "placeholder", efficiency: "placeholder" } },
  // AMD Instinct platforms: 8 OAM modules per UBB node, fully connected over AMD Infinity Fabric (full mesh);
  // scale-out over standard 400GbE Ethernet (RoCEv2), the common Instinct platform networking.
  // intra bw = uni-directional aggregate per GPU: MI300X/MI325X 7 peer links x 128 GB/s / 2 = 448 GB/s;
  // MI355X 7 links x 153 GB/s / 2 ~= 535 GB/s (CDNA4 whitepaper: >1 TB/s bi-directional per GPU).
  // Fabric hop latency and cloud price are engineering placeholders until measured.
  mi300x_if_roce: { key: "mi300x_if_roce", label: "MI300X · Infinity Fabric + 400GbE", device: MI300X, devicesPerNode: 8,
    intra: { bw: 448e9, latencyS: 5e-6 }, inter: { bw: 50e9, latencyS: 10e-6 }, costPerDeviceHour: 2.0,
    provenance: { peak: "spec (AMD datasheet, dense)", "intra.bw": "spec (7x128 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", latency: "placeholder", price: "placeholder (market est.)", efficiency: "placeholder" } },
  mi325x_if_roce: { key: "mi325x_if_roce", label: "MI325X · Infinity Fabric + 400GbE", device: MI325X, devicesPerNode: 8,
    intra: { bw: 448e9, latencyS: 5e-6 }, inter: { bw: 50e9, latencyS: 10e-6 }, costPerDeviceHour: 2.5,
    provenance: { peak: "spec (AMD datasheet, dense)", "intra.bw": "spec (7x128 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", latency: "placeholder", price: "placeholder (market est.)", efficiency: "placeholder" } },
  mi355x_if_roce: { key: "mi355x_if_roce", label: "MI355X · Infinity Fabric + 400GbE", device: MI355X, devicesPerNode: 8,
    intra: { bw: 535e9, latencyS: 5e-6 }, inter: { bw: 50e9, latencyS: 10e-6 }, costPerDeviceHour: 3.0,
    provenance: { peak: "spec (AMD datasheet, dense)", "intra.bw": "spec (7x153 GB/s IF full mesh, uni-dir)", "inter.bw": "placeholder (400GbE RoCE est.)", latency: "placeholder", price: "placeholder (market est.)", efficiency: "placeholder" } },
};

export function flops(d: Device, dtype: string): number {
  const v = d.peakFlops[dtype];
  if (v === undefined) throw new Error(`${d.name} has no ${dtype} rate`);
  return v;
}

export function linkForGroup(c: Cluster, size: number, stride = 1): Link {
  return size * stride <= c.devicesPerNode ? c.intra : c.inter;
}

// ---- cost models (ring collectives, alpha-beta; roofline) -------------------------------------
export function rooflineTime(fl: number, bytes: number, d: Device, dtype = "bf16"): number {
  const tc = fl ? fl / (flops(d, dtype) * d.efficiency.matmul) : 0;
  const tm = bytes ? bytes / (d.memBw * d.efficiency.memory) : 0;
  return Math.max(tc, tm);
}
export function allReduce(S: number, n: number, l: Link): number {
  if (n <= 1 || S <= 0) return 0;
  return 2 * (n - 1) * l.latencyS + ((2 * (n - 1)) / n) * S / l.bw;
}
export function allGather(S: number, n: number, l: Link): number {
  if (n <= 1 || S <= 0) return 0;
  return (n - 1) * l.latencyS + ((n - 1) / n) * S / l.bw;
}
export function p2p(S: number, l: Link): number {
  return S > 0 ? l.latencyS + S / l.bw : 0;
}

// ---- transformer description -------------------------------------------------------------------
export interface Transformer {
  name: string; layers: number; hidden: number; heads: number; kvHeads: number; ffn: number; vocab: number;
  gatedMlp: boolean; tiedEmbeddings: boolean;
}
export const headDim = (m: Transformer) => Math.floor(m.hidden / m.heads);
export function paramsPerLayer(m: Transformer): number {
  const h = m.hidden, kv = m.kvHeads * headDim(m);
  return h * h + 2 * h * kv + h * h + (m.gatedMlp ? 3 : 2) * h * m.ffn + 2 * h;
}
export const embeddingParams = (m: Transformer) => m.vocab * m.hidden * (m.tiedEmbeddings ? 1 : 2);
export const params = (m: Transformer) => m.layers * paramsPerLayer(m) + embeddingParams(m) + m.hidden;

const T = (name: string, layers: number, hidden: number, heads: number, kvHeads: number, ffn: number, vocab: number,
  gatedMlp = true, tiedEmbeddings = false): Transformer => ({ name, layers, hidden, heads, kvHeads, ffn, vocab, gatedMlp, tiedEmbeddings });

export const MODELS: Record<string, Transformer> = {
  "llama2-7b": T("Llama 2 7B", 32, 4096, 32, 32, 11008, 32000),
  "llama3-8b": T("Llama 3 8B", 32, 4096, 32, 8, 14336, 128256),
  "llama2-70b": T("Llama 2 70B", 80, 8192, 64, 8, 28672, 32000),
  "gpt3-175b": T("GPT-3 175B", 96, 12288, 96, 96, 49152, 50257, false, true),
};
for (const [k, heads, hidden, layers] of [["1.7b", 24, 2304, 24], ["3.6b", 32, 3072, 30], ["7.5b", 32, 4096, 36],
  ["18.4b", 48, 6144, 40], ["39.1b", 64, 8192, 48], ["76.1b", 80, 10240, 60], ["145.6b", 96, 12288, 80],
  ["310.1b", 128, 16384, 96], ["529.6b", 128, 20480, 105], ["1008b", 160, 25600, 128], ["174.6b", 96, 12288, 96]] as const) {
  MODELS[`megatron-gpt-${k}`] = T(`Megatron GPT ${k.toUpperCase()}`, layers, hidden, heads, heads, 4 * hidden, 51200, false, true);
}
