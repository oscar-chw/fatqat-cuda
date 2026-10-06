// Check software binary64 on the GPU against the CPU, bit for bit, then time
// a one-qubit dense gate built from it.
// Build: swiftc -O fp64check.swift -o fp64check ; run: ./fp64check fp64.metal out.json
import Foundation
import Metal

let args = CommandLine.arguments
let source = try! String(contentsOfFile: args[1], encoding: .utf8)
let device = MTLCreateSystemDefaultDevice()!
let library = try! device.makeLibrary(source: source, options: nil)
let queue = device.makeCommandQueue()!

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

func run(_ name: String, _ buffers: [MTLBuffer], threads: Int, bit: UInt32? = nil) -> Double {
    let p = pipeline(name)
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(p)
    for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
    if var value = bit { encoder.setBytes(&value, length: 4, index: 2) }
    encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    let start = Date(); command.commit(); command.waitUntilCompleted()
    return Date().timeIntervalSince(start)
}

// --- random operands covering every class ---------------------------------
var rng = SystemRandomNumberGenerator()
func randomDouble() -> Double {
    switch Int.random(in: 0..<10, using: &rng) {
    case 0: return Double(bitPattern: UInt64.random(in: 0...UInt64.max, using: &rng))  // any pattern
    case 1: return Double(bitPattern: UInt64.random(in: 0...0x000F_FFFF_FFFF_FFFF, using: &rng)) * (Bool.random() ? 1 : -1)  // subnormal
    case 2: return [0.0, -0.0, .infinity, -.infinity, .nan, Double.leastNonzeroMagnitude, Double.greatestFiniteMagnitude, 1.0, -1.0][Int.random(in: 0..<9, using: &rng)]
    case 3: return Double.random(in: -1e-300...1e-300, using: &rng)
    case 4: return Double.random(in: -1e300...1e300, using: &rng)
    case 5: return Double(Int.random(in: -1000...1000, using: &rng)) / 64.0  // exact, tie-prone sums
    default: return Double.random(in: -1...1, using: &rng)  // amplitudes and gate entries
    }
}

func scaled(_ exponent: Int) -> Double {
    Double(sign: Bool.random(using: &rng) ? .minus : .plus, exponent: exponent,
           significand: Double.random(in: 1..<2, using: &rng))
}

// Pairs aimed at the hard cases, mixed with the random ones above.
func edgePair(_ kind: Int) -> (Double, Double) {
    switch kind {
    case 0:  // products at the subnormal/normal boundary (and rounding onto it)
        let e = Int.random(in: -540...(-480), using: &rng)
        return (scaled(e), scaled(-1022 - e + Int.random(in: -3...2, using: &rng)))
    case 1:  // products at the overflow edge
        let e = Int.random(in: 480...540, using: &rng)
        return (scaled(e), scaled(1023 - e + Int.random(in: -1...1, using: &rng)))
    case 2:  // sums after an alignment shift of 0-70 bits, with ties likely
        let x = scaled(Int.random(in: -20...20, using: &rng))
        let k = Int.random(in: 0...70, using: &rng)
        var y = Double(sign: Bool.random(using: &rng) ? .minus : .plus, exponent: x.exponent - k, significand: 1.0)
        if Bool.random(using: &rng) { y = y.nextUp }
        return (x, y)
    default:  // subnormal plus or minus subnormal (cancellation into subnormals)
        let x = Double(bitPattern: UInt64.random(in: 1...0x000F_FFFF_FFFF_FFFF, using: &rng))
        return (x, -x.nextUp)
    }
}

let n = 4_000_000
var a = [UInt64](repeating: 0, count: n), b = a
for i in 0..<n {
    var x = randomDouble()
    var y = randomDouble()
    if i % 7 == 0 { y = -x * (1 + Double.random(in: -1e-15...1e-15, using: &rng)) }  // cancellation
    if i % 11 == 0 { y = x.nextUp }
    if i % 3 == 0 { (x, y) = edgePair(i % 4) }
    a[i] = x.bitPattern; b[i] = y.bitPattern
}
let bufA = device.makeBuffer(bytes: a, length: n * 8, options: .storageModeShared)!
let bufB = device.makeBuffer(bytes: b, length: n * 8, options: .storageModeShared)!
let out = device.makeBuffer(length: n * 8, options: .storageModeShared)!

var report: [String: Any] = [
    "cases": 0, "mismatches": 0,
    "description": "Software binary64 multiply and add on the Apple GPU against the CPU: random operands of every class plus pairs aimed at the subnormal boundary, the overflow edge, ties after alignment shifts and subnormal cancellation. Non-NaN results are compared bit for bit; NaN results only as NaN.",
    "code_revision": args.count > 3 ? args[3] : "",
    "measurement_date": ISO8601DateFormatter().string(from: Date()).prefix(10).description,
]
var examples: [String] = []
var total = 0, bad = 0, nanResults = 0, nanPatternsDiffer = 0
for (name, op) in [("check_mul", { (x: Double, y: Double) in x * y }), ("check_add", { (x: Double, y: Double) in x + y })] {
    _ = run(name, [bufA, bufB, out], threads: n)
    let got = out.contents().bindMemory(to: UInt64.self, capacity: n)
    var mismatches = 0
    for i in 0..<n {
        let want = op(Double(bitPattern: a[i]), Double(bitPattern: b[i]))
        let g = got[i]
        if want.isNaN {
            nanResults += 1
            if g != want.bitPattern { nanPatternsDiffer += 1 }
        }
        let ok = want.isNaN ? Double(bitPattern: g).isNaN : g == want.bitPattern
        if !ok {
            mismatches += 1
            if examples.count < 12 {
                examples.append(String(format: "%@ %016llx %016llx -> gpu %016llx cpu %016llx", name, a[i], b[i], g, want.bitPattern))
            }
        }
    }
    total += n; bad += mismatches
    print("\(name): \(mismatches) mismatches in \(n)")
}
report["cases"] = total
report["mismatches"] = bad
report["nan_results"] = nanResults
report["nan_results_with_another_payload"] = nanPatternsDiffer
report["examples"] = examples

// --- a one-qubit dense gate (H), bit-identical to the CPU loop -----------
let r = 1.0 / 2.0.squareRoot()
let matrix: [UInt64] = [r, 0, r, 0, r, 0, -r, 0].map { $0.bitPattern }
let bufM = device.makeBuffer(bytes: matrix, length: 64, options: .storageModeShared)!
func cmul(_ mr: Double, _ mi: Double, _ ar: Double, _ ai: Double) -> (Double, Double) {
    let p1 = mr * ar, p2 = mi * ai, p3 = mr * ai, p4 = mi * ar
    return (p1 - p2, p3 + p4)
}
let small = 1 << 20
var state = [UInt64](repeating: 0, count: 2 * small)
for i in 0..<(2 * small) { state[i] = Double.random(in: -1...1, using: &rng).bitPattern }
let bufS = device.makeBuffer(bytes: state, length: small * 16, options: .storageModeShared)!
_ = run("dense_1q", [bufS, bufM], threads: small / 2, bit: 3)
let gpuState = bufS.contents().bindMemory(to: UInt64.self, capacity: 2 * small)
var gateMismatches = 0
for g in 0..<(small / 2) {
    let low = g & ((1 << 3) - 1), i0 = ((g >> 3) << 4) | low, i1 = i0 | 8
    let a0 = (Double(bitPattern: state[2 * i0]), Double(bitPattern: state[2 * i0 + 1]))
    let a1 = (Double(bitPattern: state[2 * i1]), Double(bitPattern: state[2 * i1 + 1]))
    for (row, index) in [(0, i0), (1, i1)] {
        let m0 = Double(bitPattern: matrix[4 * row]), m0i = Double(bitPattern: matrix[4 * row + 1])
        let m1 = Double(bitPattern: matrix[4 * row + 2]), m1i = Double(bitPattern: matrix[4 * row + 3])
        var (sr, si) = cmul(m0, m0i, a0.0, a0.1)
        sr = 0.0 + sr; si = 0.0 + si
        let (tr, ti) = cmul(m1, m1i, a1.0, a1.1)
        sr = sr + tr; si = si + ti
        if gpuState[2 * index] != sr.bitPattern || gpuState[2 * index + 1] != si.bitPattern { gateMismatches += 1 }
    }
}
print("dense_1q vs CPU loop: \(gateMismatches) mismatches in \(small)")
report["gate_mismatches"] = gateMismatches

let big = 1 << 26
let bufBig = device.makeBuffer(length: big * 16, options: .storageModeShared)!
var best = Double.infinity
for _ in 0..<3 {
    var elapsed = 0.0
    for q in 0..<20 { elapsed += run("dense_1q", [bufBig, bufM], threads: big / 2, bit: UInt32(q)) }
    best = min(best, elapsed / 20)
}
print(String(format: "dense_1q at 2^26 amplitudes: %.2f ms per gate (software binary64)", best * 1e3))
report["dense_1q_ms_per_gate_2_26"] = best * 1e3

let json = try! JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try! json.write(to: URL(fileURLWithPath: args[2]))
