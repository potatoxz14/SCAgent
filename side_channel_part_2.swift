import Foundation
import UIKit
import QuartzCore
import Metal
import Vision
import CoreMotion
import AVFoundation

class SideChannelCollector: NSObject {
    
    private var collectorQueue = DispatchQueue(label: "com.sidechannel.collector", qos: .userInteractive)
    private var isRunning = false
    private var tickCounter: UInt64 = 0
    
    var onNewData: ((String) -> Void)?
    
    // 1. Metal (GPU)
    private let device = MTLCreateSystemDefaultDevice()
    private var commandQueue: MTLCommandQueue?
    // 2. Vision (ANE)
    private var visionRequest: VNDetectFaceRectanglesRequest?
    private let dummyImageHandler: VNImageRequestHandler?
    // 3. Motion (Sensor)
    private let motionManager = CMMotionManager()
    // 4. Disk (IO)
    private let tempFileURL: URL
    
    private var memoryBlock: UnsafeMutablePointer<Int>
    private let memorySize = 1024 * 512 
    
    override init() {
        self.commandQueue = device?.makeCommandQueue()
        
        let data = NSMutableData(length: 100)!
        self.dummyImageHandler = VNImageRequestHandler(data: data as Data, options: [:])
        self.visionRequest = VNDetectFaceRectanglesRequest()
        
        self.tempFileURL = FileManager.default.temporaryDirectory.appendingPathComponent("probe_io.dat")
        FileManager.default.createFile(atPath: tempFileURL.path, contents: Data(count: 4096), attributes: nil)
        
        self.memoryBlock = UnsafeMutablePointer<Int>.allocate(capacity: memorySize)
        
        super.init()
    }
    
    deinit {
        memoryBlock.deallocate()
    }

    public var csvHeader: String {
        var h = "timestamp,idx,"
        
        h += "vm_comp,vm_decomp,vm_swapin,vm_swapout,vm_ext,vm_int,vm_cow,"
        h += "net_lo_ib,net_lo_ob,net_en_ib,net_en_ob,net_pdp_op,"
        
        h += "act_l2_mem,"      
        h += "act_main_pre,"    
        if Config.enableMetalProbe { h += "act_gpu_lat," } // Metal
        if Config.enableANEProbe   { h += "act_ane_lat," } // ANE
        if Config.enableAudioProbe { h += "act_aud_blk," } // Audio Daemon
        if Config.enableDiskProbe  { h += "act_dsk_sat," } // Disk Saturation
        
        h += "phy_therm,"       // Thermal State
        h += "phy_bright,"      // Screen Brightness
        if Config.enableSensorProbe { h += "phy_acc_z," }  // Accelerometer Z
        
        if Config.enableMetadataProbe {
            h += "meta_arkit,"   // ARKit Framework Access
            h += "meta_swiftui," // SwiftUI Framework Access
            h += "meta_sys_flt"  // Shared Cache Fault
        }
        
        return h + "\n"
    }
    
    
    func startCollecting() {
        guard !isRunning else { return }
        isRunning = true
        tickCounter = 0
        
        if Config.enableSensorProbe && motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = 0.02
            motionManager.startAccelerometerUpdates()
        }
        
        onNewData?(csvHeader)
        loop()
    }
    
    func stopCollecting() {
        isRunning = false
        motionManager.stopAccelerometerUpdates()
    }
    
    
    private func loop() {
        collectorQueue.asyncAfter(deadline: .now() + .microseconds(Int(Config.baseIntervalMicroseconds))) { [weak self] in
            guard let self = self, self.isRunning else { return }
            self.collectSingleSample()
            self.loop()
        }
    }
    
    private func collectSingleSample() {
        let timestamp = Date().timeIntervalSince1970
        tickCounter += 1
        
        let vmStats = MachBridge.getDeepVMStatistics()
        let netStats = MachBridge.getDetailedNetworkStatistics()
        
        
        let latMem = measureNano {
            var sum = 0
            // Stride access to force cache misses
            for i in stride(from: 0, to: self.memorySize, by: 64) {
                sum &+= self.memoryBlock[i]
            }
        }
        
        let latMain = measureNano {
            DispatchQueue.main.sync { var _ = 1 + 1 }
        }
        
        var latGPU = 0.0
        if Config.enableMetalProbe && (tickCounter % UInt64(Config.gpuProbeFrequency) == 0) {
            latGPU = measureNano {
                guard let buffer = self.commandQueue?.makeCommandBuffer(),
                      let encoder = buffer.makeBlitCommandEncoder() else { return }
                encoder.endEncoding()
                buffer.commit()
                buffer.waitUntilCompleted() 
            }
        }
        
        var latANE = 0.0
        if Config.enableANEProbe && (tickCounter % UInt64(Config.aneProbeFrequency) == 0) {
            latANE = measureNano {
                try? self.dummyImageHandler?.perform([self.visionRequest!])
            }
        }
        
        var latAudio = 0.0        
        if Config.enableAudioProbe {
            latAudio = measureNano {
                _ = AVAudioSession.sharedInstance().outputVolume
            }
        }
        
        var latDisk = 0.0
        if Config.enableDiskProbe && (tickCounter % UInt64(Config.diskProbeFrequency) == 0) {
            latDisk = measureNano {
                let fd = open(self.tempFileURL.path, O_RDWR)
                if fd > 0 {
                    fcntl(fd, F_FULLFSYNC) 
                    close(fd)
                }
            }
        }
        
        
        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        
        let brightness = UIScreen.main.brightness
        
        var accelZ = 0.0
        if Config.enableSensorProbe, let data = motionManager.accelerometerData {
            accelZ = data.acceleration.z
        }
        
        
        var latARKit = 0.0
        var latSwiftUI = 0.0
        var latSysFlt = 0.0
        
        if Config.enableMetadataProbe {
            latARKit = measureNano {
                var statBuf = stat()
                _ = stat("/System/Library/Frameworks/ARKit.framework", &statBuf)
            }
            
            latSwiftUI = measureNano {
                var statBuf = stat()
                _ = stat("/System/Library/Frameworks/SwiftUI.framework", &statBuf)
            }
            
            latSysFlt = measureNano {
                let _ = try? Data(contentsOf: URL(fileURLWithPath: "/System/Library/CoreServices/SystemVersion.plist"))
            }
        }
        
        var line = String(format: "%.3f,%d,", timestamp, tickCounter)
        
        line += String(format: "%@,%@,%@,%@,%@,%@,%@,",
                       vmStats["compressions"] ?? 0, vmStats["decompressions"] ?? 0,
                       vmStats["swapins"] ?? 0, vmStats["swapouts"] ?? 0,
                       vmStats["external_pages"] ?? 0, vmStats["internal_pages"] ?? 0,
                       vmStats["cow_faults"] ?? 0)
        
        line += String(format: "%@,%@,%@,%@,%@,",
                       netStats["lo0_ib"] ?? 0, netStats["lo0_ob"] ?? 0,
                       netStats["en0_im"] ?? 0, netStats["en0_om"] ?? 0,
                       netStats["pdp_op"] ?? 0)
        
        // Active
        line += String(format: "%.6f,%.6f,", latMem, latMain)
        if Config.enableMetalProbe { line += String(format: "%.6f,", latGPU) }
        if Config.enableANEProbe   { line += String(format: "%.6f,", latANE) }
        if Config.enableAudioProbe { line += String(format: "%.6f,", latAudio) }
        if Config.enableDiskProbe  { line += String(format: "%.6f,", latDisk) }
        
        // Physical
        line += String(format: "%d,%.3f,", thermal, brightness)
        if Config.enableSensorProbe { line += String(format: "%.4f,", accelZ) }
        
        // Metadata
        if Config.enableMetadataProbe {
            line += String(format: "%.6f,%.6f,%.6f", latARKit, latSwiftUI, latSysFlt)
        }
        
        line += "\n"
        onNewData?(line)
    }
    
    private func measureNano(_ block: () -> Void) -> Double {
        let start = mach_absolute_time()
        block()
        let end = mach_absolute_time()
        var info = mach_timebase_info()
        mach_timebase_info(&info)
        return Double((end - start) * UInt64(info.numer) / UInt64(info.denom)) / 1e9
    }
}

