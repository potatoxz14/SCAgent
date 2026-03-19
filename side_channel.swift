import CoreML
import Darwin
import Combine
import Foundation
import SwiftUI

class Prober: ObservableObject {
    static let shared = Prober()
    //    var latestData: (aneLatency: Double, diskLatency: Double, cacheLatency: Double, gpuLatency: Double , cpuJitter: Double , fsFreeBytes: UInt64 ) = (0,0,0,0,0,0)
    @Published var aneLatency: Double = 0.0
    @Published var diskLatency: Double = 0.0
    @Published var cacheLatency: Double = 0.0
    @Published var gpuLatency: Double = 0.0 //
    @Published var cpuJitter: Double = 0.0
//    @Published var fsChurnDelta: Int64 = 0
    @Published var fsFreeBytes: UInt64 = 0
    
    private var isRunning = false
    private let jitterQueue = DispatchQueue(label: "com.probe.jitter", qos: .userInteractive)
    private let monitorQueue = DispatchQueue(label: "com.probe.monitor", qos: .userInitiated)
    

    private var cacheProbeArray: [Int]?
    

    private var aneModel: voice_remove?
    private let aneInputData = try! MLMultiArray(shape: [1024], dataType: .double)
    

    private let F_FULLFSYNC: Int32 = 51 // macOS/iOS kernel constant
    
    
    private var metalDevice: MTLDevice?
    private var metalQueue: MTLCommandQueue?
    private var dummyBuffer: MTLBuffer?
    private var metalPipeline: MTLComputePipelineState?
    private let metalLibrarySource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void dummy_kernel(device float *result [[buffer(0)]],
                                 uint id [[thread_position_in_grid]]) {

            result[id] = 1.0;
        }
        """
    init() {
        // 初始化缓存数组
        self.cacheProbeArray = [Int](repeating: 1, count: 2 * 1024 * 1024)
        setup_M()
        setup_A()
    }
    private func setup_M() {
            guard let device = MTLCreateSystemDefaultDevice() else {
//                print("Metal not supported")
                return
            }
            self.metalDevice = device
            self.metalQueue = device.makeCommandQueue()
            
            self.dummyBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)
            
            do {
                let library = try device.makeLibrary(source: metalLibrarySource, options: nil)
                guard let function = library.makeFunction(name: "dummy_kernel") else {
//                    print("Failed to find kernel function")
                    return
                }
                self.metalPipeline = try device.makeComputePipelineState(function: function)
            } catch {
//                print("Metal setup failed: \(error)")
            }
        }
        

        private func test_G() -> Double {

            guard let queue = metalQueue,
                  let buffer = dummyBuffer,
                  let pipeline = metalPipeline else { return 0 }
            
            let start = mach_absolute_time()
            
            guard let cmdBuffer = queue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else { return 0 }
            

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            
 
            encoder.dispatchThreadgroups(MTLSizeMake(1, 1, 1),
                                         threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
            encoder.endEncoding()
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            
            let end = mach_absolute_time()
            return timeDiff(start, end)
        }
    private func setup_A() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.aneModel = try? voice_remove(configuration: config)
    }
    
    private func test_A() -> Double {
        guard let model = self.aneModel else { return 0 }
        
        let start = mach_absolute_time()
        let input = voice_removeInput(input: aneInputData)
        let _ = try? model.prediction(input: input)
        
        let end = mach_absolute_time()
        return timeDiff(start, end)
    }


    private var lastFSFree: UInt64 = 0
    
    private func test_F() -> (UInt64) {
        var stats = statfs()
        let path = "/private/var"
        
        if statfs(path, &stats) == 0 {
            let freeBytes = UInt64(stats.f_bfree) * UInt64(stats.f_bsize)
            lastFSFree = freeBytes
            return (freeBytes)
        } else {

            statfs(NSTemporaryDirectory(), &stats)
            return (0)
        }
    }

    private var currentJitterMax: Double = 0.0
    
    private func setupJitterProbe() {
        jitterQueue.async {
            while self.isRunning {
                let t1 = mach_absolute_time()
                for _ in 0..<100 { _ = 1 + 1 }
                let t2 = mach_absolute_time()
                
                let delta = self.timeDiff(t1, t2)
                
                if delta > self.currentJitterMax {
                    self.currentJitterMax = delta
                }
                usleep(100)
            }
        }
    }

    private func test_D() -> Double {
        let tempFile = NSTemporaryDirectory() + UUID().uuidString
        let data = Data(repeating: 0, count: 4096) // 4KB
        
        let start = mach_absolute_time()
        let fd = open(tempFile, O_WRONLY | O_CREAT, 0o644)
        if fd != -1 {
            data.withUnsafeBytes { buffer in
                write(fd, buffer.baseAddress, 4096)
            }
            fcntl(fd, F_FULLFSYNC)
            close(fd)
            unlink(tempFile)
        }
        let end = mach_absolute_time()
        return timeDiff(start, end)
    }

    private func test_C() -> Double {
        return measureBlock {
            guard let arr = self.cacheProbeArray else { return }
            let stride = 64
            var sum = 0
            let limit = 1000
            let count = arr.count
            for i in 0..<limit {
                let idx = (i * stride) % count
                sum &+= arr[idx]
            }
        }
    }
    
    private func timeDiff(_ start: UInt64, _ end: UInt64) -> Double {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let elapsed = end - start
        let nanos = elapsed * UInt64(timebase.numer) / UInt64(timebase.denom)
        return Double(nanos) / 1_000_000.0
    }
    
    private func measureBlock(block: () -> Void) -> Double {
        let start = mach_absolute_time()
        block()
        let end = mach_absolute_time()
        return timeDiff(start, end)
    }
    
    
    func start() {
        guard !isRunning else { return }
        isRunning = true
        
        setupJitterProbe()
        

        monitorQueue.async {
            while self.isRunning {

                let dLatency = self.test_D()
                let cLatency = self.test_C()
                let (fsFree) = self.test_F()
                let aLatency = self.test_A()
                let gpuLatency = self.test_G()
                
                let jitter = self.currentJitterMax
                self.currentJitterMax = 0
                
                
                usleep(200_000)
            }
        }
    }
    
    func stop() {
        isRunning = false
    }
}

