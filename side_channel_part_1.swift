import Foundation
import UIKit
import QuartzCore

class SideChannelCollector: NSObject {
    
    private var collectorQueue = DispatchQueue(label: "com.sidechannel.collector", qos: .userInteractive)
    private var isRunning = false
    
    var onNewData: ((String) -> Void)?
    
    private var collectionCount: UInt64 = 0
    
    private let jitterLock = NSLock()
    private var displayLink: CADisplayLink?
    private var _mainThreadJitter: Double = 0.0
    
    
    public var csvHeader: String {
        return """
        timestamp,idx,\
        vm_compress,vm_decompress,vm_swapin,vm_swapout,vm_comp_pages,vm_uncomp_pages,\
        vm_ext_pages,vm_int_pages,vm_spec,vm_throt,vm_zero,vm_purges,vm_purgeable,vm_react,vm_cow,\
        lo0_ib,lo0_ob,lo0_ip,awdl_ib,awdl_ob,awdl_im,awdl_om,\
        en0_im,en0_om,en0_iqdrop,en0_v6ib,utun_ib,pdp_op,\
        probe_jit_sim,probe_dns,probe_font,probe_tls_cpu,probe_ui_jitter,probe_img_io,probe_fs_meta,probe_socket_lat\n
        """
    }
    
    
    func startCollecting() { 
        guard !isRunning else { return }
        isRunning = true
        collectionCount = 0
        
        
        onNewData?(csvHeader)
        
        DispatchQueue.main.async {
            self.displayLink = CADisplayLink(target: self, selector: #selector(self.displayLinkTick))
            self.displayLink?.add(to: .main, forMode: .common)
        }
        
        loop()
    }
    
    func stopCollecting() {
        isRunning = false
        displayLink?.invalidate()
        displayLink = nil
    }
    
    
    private func loop() {
        collectorQueue.asyncAfter(deadline: .now() + .microseconds(Int(Config.sleepInterval))) { [weak self] in
            guard let self = self, self.isRunning else { return }
            self.collectSingleSample()
            self.loop()
        }
    }
    
    private func collectSingleSample() {
        let timestamp = Date().timeIntervalSince1970
        collectionCount += 1
        
        // 1. VM
        let vmStats = MachBridge.getDeepVMStatistics()
        // 2. Network
        let netStats = MachBridge.getDetailedNetworkStatistics()
        
        // 3. Probes
        let latJIT = measure {
             let ptr = UnsafeMutablePointer<Int>.allocate(capacity: 1024)
             for i in 0..<1024 { ptr[i] = i }
             ptr.deallocate()
        }
        
        let latDNS = measure { _ = gethostbyname("localhost") }
        let latFont = measure { _ = UIFont.systemFont(ofSize: 12).fontName }
        let latCPU = measure { var v=0; for _ in 0..<1000 { v = v &+ 1 } }
        
        var latJitter = 0.0
        jitterLock.lock()
        latJitter = _mainThreadJitter
        _mainThreadJitter = 0
        jitterLock.unlock()
        
        let latImg = measure { _ = CGColorSpaceCreateDeviceRGB() }
        let latFS = measure { _ = try? FileManager.default.attributesOfItem(atPath: NSTemporaryDirectory()) }
        let latSock = measure { let fd = socket(AF_INET, SOCK_STREAM, 0); if fd > 0 { close(fd) } }
        
        // 4. Stringify
        let strVM = String(format: "%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@",
                           vmStats["compressions"] ?? 0, vmStats["decompressions"] ?? 0,
                           vmStats["swapins"] ?? 0, vmStats["swapouts"] ?? 0,
                           vmStats["compressor_pages"] ?? 0, vmStats["uncompressed_pages"] ?? 0,
                           vmStats["external_pages"] ?? 0, vmStats["internal_pages"] ?? 0,
                           vmStats["speculative"] ?? 0, vmStats["throttled"] ?? 0,
                           vmStats["zero_fill"] ?? 0, vmStats["purges"] ?? 0,
                           vmStats["purgeable"] ?? 0, vmStats["reactivations"] ?? 0,
                           vmStats["cow_faults"] ?? 0)
        
        let strNet = String(format: "%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@,%@",
                            netStats["lo0_ib"] ?? 0, netStats["lo0_ob"] ?? 0, netStats["lo0_ip"] ?? 0,
                            netStats["awdl0_ib"] ?? 0, netStats["awdl0_ob"] ?? 0,
                            netStats["awdl0_im"] ?? 0, netStats["awdl0_om"] ?? 0,
                            netStats["en0_im"] ?? 0, netStats["en0_om"] ?? 0,
                            netStats["en0_iqdrop"] ?? 0, netStats["en0_v6_ib"] ?? 0,
                            netStats["utun_ib"] ?? 0, netStats["pdp_op"] ?? 0)
        
        let strProbe = String(format: "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                              latJIT, latDNS, latFont, latCPU, latJitter, latImg, latFS, latSock)
        
        let line = "\(String(format: "%.3f", timestamp)),\(collectionCount),\(strVM),\(strNet),\(strProbe)\n"
        
        onNewData?(line)
    }
    
    private func measure(_ block: () -> Void) -> Double {
        let start = mach_absolute_time()
        block()
        let end = mach_absolute_time()
        var info = mach_timebase_info()
        mach_timebase_info(&info)
        return Double((end - start) * UInt64(info.numer) / UInt64(info.denom)) / 1e9
    }
    
    @objc private func displayLinkTick(link: CADisplayLink) {
        let duration = link.targetTimestamp - link.timestamp
        jitterLock.lock()
        let delta = abs(duration - (1.0/60.0))
        if delta > _mainThreadJitter { _mainThreadJitter = delta }
        jitterLock.unlock()
    }
}

