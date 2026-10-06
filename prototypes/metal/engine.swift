// Prototype bridge: a shared-memory complex128 state and the Metal tile kernel.
// Build: swiftc -O -emit-library engine.swift -o libfqmetal.dylib
import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var tilePipeline: MTLComputePipelineState? = nil
var buffers: [UInt: MTLBuffer] = [:]

@_cdecl("fq_metal_init")
public func fqMetalInit(_ path: UnsafePointer<CChar>) -> Int32 {
    guard let source = try? String(contentsOfFile: String(cString: path), encoding: .utf8) else { return 1 }
    do {
        let library = try device.makeLibrary(source: source, options: nil)
        tilePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "gate_tile")!)
    } catch {
        FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
        return 2
    }
    return 0
}

@_cdecl("fq_metal_alloc")
public func fqMetalAlloc(_ bytes: Int) -> UnsafeMutableRawPointer? {
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else { return nil }
    buffers[UInt(bitPattern: buffer.contents())] = buffer
    return buffer.contents()
}

@_cdecl("fq_metal_free")
public func fqMetalFree(_ pointer: UnsafeMutableRawPointer) {
    buffers.removeValue(forKey: UInt(bitPattern: pointer))
}

var pending: MTLCommandBuffer? = nil

@_cdecl("fq_metal_wait")
public func fqMetalWait() {
    pending?.waitUntilCompleted()
    pending = nil
}

// Tiles [first, first + count) of one batch, started without waiting (call
// fq_metal_wait): each threadgroup loads one tile of 2^11 amplitudes into
// threadgroup memory, applies every gate, writes it back.
@_cdecl("fq_metal_tiles")
public func fqMetalTiles(_ pointer: UnsafeMutableRawPointer,
                         _ matrices: UnsafeRawPointer, _ matrixBytes: Int,
                         _ gates: UnsafeRawPointer, _ nGates: Int32,
                         _ masks: UnsafeRawPointer,
                         _ tileBits: UnsafeRawPointer, _ restBits: UnsafeRawPointer, _ nRest: Int32,
                         _ first: UInt32, _ count: UInt32) {
    let buffer = buffers[UInt(bitPattern: pointer)]!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(tilePipeline!)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    let matrixBuffer = device.makeBuffer(bytes: matrices, length: max(matrixBytes, 16), options: .storageModeShared)!
    let gateBuffer = device.makeBuffer(bytes: gates, length: max(Int(nGates) * 15 * 4, 4), options: .storageModeShared)!
    let maskBuffer = device.makeBuffer(bytes: masks, length: max(Int(nGates) * 16, 8), options: .storageModeShared)!
    let tileBuffer = device.makeBuffer(bytes: tileBits, length: 11 * 4, options: .storageModeShared)!
    let restBuffer = device.makeBuffer(bytes: restBits, length: max(Int(nRest) * 4, 4), options: .storageModeShared)!
    encoder.setBuffer(matrixBuffer, offset: 0, index: 1)
    encoder.setBuffer(gateBuffer, offset: 0, index: 2)
    encoder.setBuffer(maskBuffer, offset: 0, index: 3)
    var n = nGates, r = nRest
    encoder.setBytes(&n, length: 4, index: 4)
    encoder.setBuffer(tileBuffer, offset: 0, index: 5)
    encoder.setBuffer(restBuffer, offset: 0, index: 6)
    encoder.setBytes(&r, length: 4, index: 7)
    var start = first
    encoder.setBytes(&start, length: 4, index: 8)
    encoder.dispatchThreadgroups(MTLSize(width: Int(count), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    pending = command
}
