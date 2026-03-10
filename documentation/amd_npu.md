Title: AMD NPU — The Linux Kernel documentation

URL Source: https://docs.kernel.org/accel/amdxdna/amdnpu.html

Published Time: Mon, 09 Mar 2026 00:09:01 GMT

Markdown Content:
AMD NPU — The Linux Kernel documentation
===============

[![Image 1: Logo](https://docs.kernel.org/_static/logo.svg)](https://docs.kernel.org/index.html)

[The Linux Kernel](https://docs.kernel.org/index.html)
======================================================

7.0.0-rc3

### Quick search

### Contents

- [x] 

*   [Development process](https://docs.kernel.org/process/development-process.html)
*   [Submitting patches](https://docs.kernel.org/process/submitting-patches.html)
*   [Code of conduct](https://docs.kernel.org/process/code-of-conduct.html)
*   [Maintainer handbook](https://docs.kernel.org/maintainer/index.html)
*   [All development-process docs](https://docs.kernel.org/process/index.html)

*   [Core API](https://docs.kernel.org/core-api/index.html)
*   [Driver APIs](https://docs.kernel.org/driver-api/index.html)
*   [Subsystems](https://docs.kernel.org/subsystem-apis.html)
    *   [Core subsystems](https://docs.kernel.org/subsystem-apis.html#core-subsystems)
    *   [Human interfaces](https://docs.kernel.org/subsystem-apis.html#human-interfaces)
    *   [Networking interfaces](https://docs.kernel.org/subsystem-apis.html#networking-interfaces)
    *   [Storage interfaces](https://docs.kernel.org/subsystem-apis.html#storage-interfaces)
    *   [Other subsystems](https://docs.kernel.org/subsystem-apis.html#other-subsystems)
        *   [Accounting](https://docs.kernel.org/accounting/index.html)
        *   [CPUFreq - CPU frequency and voltage scaling code in the Linux(TM) kernel](https://docs.kernel.org/cpu-freq/index.html)
        *   [EDAC Subsystem](https://docs.kernel.org/edac/index.html)
        *   [FPGA](https://docs.kernel.org/fpga/index.html)
        *   [I2C/SMBus Subsystem](https://docs.kernel.org/i2c/index.html)
        *   [Industrial I/O](https://docs.kernel.org/iio/index.html)
        *   [PCMCIA](https://docs.kernel.org/pcmcia/index.html)
        *   [Serial Peripheral Interface (SPI)](https://docs.kernel.org/spi/index.html)
        *   [1-Wire Subsystem](https://docs.kernel.org/w1/index.html)
        *   [Watchdog Support](https://docs.kernel.org/watchdog/index.html)
        *   [Virtualization Support](https://docs.kernel.org/virt/index.html)
        *   [Hardware Monitoring](https://docs.kernel.org/hwmon/index.html)
        *   [Compute Accelerators](https://docs.kernel.org/accel/index.html)
        *   [Security Documentation](https://docs.kernel.org/security/index.html)
        *   [Crypto API](https://docs.kernel.org/crypto/index.html)
        *   [BPF Documentation](https://docs.kernel.org/bpf/index.html)
        *   [USB support](https://docs.kernel.org/usb/index.html)
        *   [PCI Bus Subsystem](https://docs.kernel.org/PCI/index.html)
        *   [Assorted Miscellaneous Devices Documentation](https://docs.kernel.org/misc-devices/index.html)
        *   [PECI Subsystem](https://docs.kernel.org/peci/index.html)
        *   [WMI Subsystem](https://docs.kernel.org/wmi/index.html)
        *   [TEE Subsystem](https://docs.kernel.org/tee/index.html)

*   [Locking](https://docs.kernel.org/locking/index.html)

*   [Licensing rules](https://docs.kernel.org/process/license-rules.html)
*   [Writing documentation](https://docs.kernel.org/doc-guide/index.html)
*   [Development tools](https://docs.kernel.org/dev-tools/index.html)
*   [Testing guide](https://docs.kernel.org/dev-tools/testing-overview.html)
*   [Hacking guide](https://docs.kernel.org/kernel-hacking/index.html)
*   [Tracing](https://docs.kernel.org/trace/index.html)
*   [Fault injection](https://docs.kernel.org/fault-injection/index.html)
*   [Livepatching](https://docs.kernel.org/livepatch/index.html)
*   [Rust](https://docs.kernel.org/rust/index.html)

*   [Administration](https://docs.kernel.org/admin-guide/index.html)
*   [Build system](https://docs.kernel.org/kbuild/index.html)
*   [Reporting issues](https://docs.kernel.org/admin-guide/reporting-issues.html)
*   [Userspace tools](https://docs.kernel.org/tools/index.html)
*   [Userspace API](https://docs.kernel.org/userspace-api/index.html)

*   [Firmware](https://docs.kernel.org/firmware-guide/index.html)
*   [Firmware and Devicetree](https://docs.kernel.org/devicetree/index.html)

*   [CPU architectures](https://docs.kernel.org/arch/index.html)

*   [Unsorted documentation](https://docs.kernel.org/staging/index.html)

*   [Translations](https://docs.kernel.org/translations/index.html)

### This Page

*   [Show Source](https://docs.kernel.org/_sources/accel/amdxdna/amdnpu.rst.txt)

AMD NPU[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#amd-npu "Permalink to this heading")
=================================================================================================

Copyright:
© 2024 Advanced Micro Devices, Inc.

Author:
Sonal Santan <[sonal.santan@amd.com](mailto:sonal.santan%40amd.com)>

Overview[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#overview "Permalink to this heading")
---------------------------------------------------------------------------------------------------

AMD NPU (Neural Processing Unit) is a multi-user AI inference accelerator integrated into AMD client APU. NPU enables efficient execution of Machine Learning applications like CNN, LLM, etc. NPU is based on [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna.html). NPU is managed by **amdxdna** driver.

Hardware Description[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#hardware-description "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------

AMD NPU consists of the following hardware components:

### AMD XDNA Array[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#amd-xdna-array "Permalink to this heading")

AMD XDNA Array comprises of 2D array of compute and memory tiles built with [AMD AI Engine Technology](https://www.xilinx.com/products/technology/ai-engine.html). Each column has 4 rows of compute tiles and 1 row of memory tile. Each compute tile contains a VLIW processor with its own dedicated program and data memory. The memory tile acts as L2 memory. The 2D array can be partitioned at a column boundary creating a spatially isolated partition which can be bound to a workload context.

Each column also has dedicated DMA engines to move data between host DDR and memory tile.

AMD Phoenix and AMD Hawk Point client NPU have a 4x5 topology, i.e., 4 rows of compute tiles arranged into 5 columns. AMD Strix Point client APU have 4x8 topology, i.e., 4 rows of compute tiles arranged into 8 columns.

### Shared L2 Memory[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#shared-l2-memory "Permalink to this heading")

The single row of memory tiles create a pool of software managed on chip L2 memory. DMA engines are used to move data between host DDR and memory tiles. AMD Phoenix and AMD Hawk Point NPUs have a total of 2560 KB of L2 memory. AMD Strix Point NPU has a total of 4096 KB of L2 memory.

### Microcontroller[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#microcontroller "Permalink to this heading")

A microcontroller runs NPU Firmware which is responsible for command processing, XDNA Array partition setup, XDNA Array configuration, workload context management and workload orchestration.

NPU Firmware uses a dedicated instance of an isolated non-privileged context called ERT to service each workload context. ERT is also used to execute user provided `ctrlcode` associated with the workload context.

NPU Firmware uses a single isolated privileged context called MERT to service management commands from the amdxdna driver.

### Mailboxes[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#mailboxes "Permalink to this heading")

The microcontroller and amdxdna driver use a privileged channel for management tasks like setting up of contexts, telemetry, query, error handling, setting up user channel, etc. As mentioned before, privileged channel requests are serviced by MERT. The privileged channel is bound to a single mailbox.

The microcontroller and amdxdna driver use a dedicated user channel per workload context. The user channel is primarily used for submitting work to the NPU. As mentioned before, a user channel requests are serviced by an instance of ERT. Each user channel is bound to its own dedicated mailbox.

### PCIe EP[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#pcie-ep "Permalink to this heading")

NPU is visible to the x86 host CPU as a PCIe device with multiple BARs and some MSI-X interrupt vectors. NPU uses a dedicated high bandwidth SoC level fabric for reading or writing into host memory. Each instance of ERT gets its own dedicated MSI-X interrupt. MERT gets a single instance of MSI-X interrupt.

The number of PCIe BARs varies depending on the specific device. Based on their functions, PCIe BARs can generally be categorized into the following types.

*   PSP BAR: Expose the AMD PSP (Platform Security Processor) function

*   SMU BAR: Expose the AMD SMU (System Management Unit) function

*   SRAM BAR: Expose ring buffers for the mailbox

*   Mailbox BAR: Expose the mailbox control registers (head, tail and ISR registers etc.)

*   Public Register BAR: Expose public registers

On specific devices, the above-mentioned BAR type might be combined into a single physical PCIe BAR. Or a module might require two physical PCIe BARs to be fully functional. For example,

*   On AMD Phoenix device, PSP, SMU, Public Register BARs are on PCIe BAR index 0.

*   On AMD Strix Point device, Mailbox and Public Register BARs are on PCIe BAR index 0. The PSP has some registers in PCIe BAR index 0 (Public Register BAR) and PCIe BAR index 4 (PSP BAR).

### Process Isolation Hardware[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#process-isolation-hardware "Permalink to this heading")

As explained before, XDNA Array can be dynamically divided into isolated spatial partitions, each of which may have one or more columns. The spatial partition is setup by programming the column isolation registers by the microcontroller. Each spatial partition is associated with a PASID which is also programmed by the microcontroller. Hence multiple spatial partitions in the NPU can make concurrent host access protected by PASID.

The NPU FW itself uses microcontroller MMU enforced isolated contexts for servicing user and privileged channel requests.

Mixed Spatial and Temporal Scheduling[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#mixed-spatial-and-temporal-scheduling "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------------------------------------------------

AMD XDNA architecture supports mixed spatial and temporal (time sharing) scheduling of 2D array. This means that spatial partitions may be setup and torn down dynamically to accommodate various workloads. A _spatial_ partition may be _exclusively_ bound to one workload context while another partition may be _temporarily_ bound to more than one workload contexts. The microcontroller updates the PASID for a temporarily shared partition to match the context that has been bound to the partition at any moment.

### Resource Solver[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#resource-solver "Permalink to this heading")

The Resource Solver component of the amdxdna driver manages the allocation of 2D array among various workloads. Every workload describes the number of columns required to run the NPU binary in its metadata. The Resource Solver component uses hints passed by the workload and its own heuristics to decide 2D array (re)partition strategy and mapping of workloads for spatial and temporal sharing of columns. The FW enforces the context-to-column(s) resource binding decisions made by the Resource Solver.

AMD Phoenix and AMD Hawk Point client NPU can support 6 concurrent workload contexts. AMD Strix Point can support 16 concurrent workload contexts.

Application Binaries[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#application-binaries "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------

A NPU application workload is comprised of two separate binaries which are generated by the NPU compiler.

1.   AMD XDNA Array overlay, which is used to configure a NPU spatial partition. The overlay contains instructions for setting up the stream switch configuration and ELF for the compute tiles. The overlay is loaded on the spatial partition bound to the workload by the associated ERT instance. Refer to the [Versal Adaptive SoC AIE-ML Architecture Manual (AM020)](https://docs.amd.com/r/en-US/am020-versal-aie-ml) for more details.

2.   `ctrlcode`, used for orchestrating the overlay loaded on the spatial partition. `ctrlcode` is executed by the ERT running in protected mode on the microcontroller in the context of the workload. `ctrlcode` is made up of a sequence of opcodes named `XAie_TxnOpcode`. Refer to the [AI Engine Run Time](https://github.com/Xilinx/aie-rt/tree/release/main_aig) for more details.

Special Host Buffers[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#special-host-buffers "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------

### Per-context Instruction Buffer[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#per-context-instruction-buffer "Permalink to this heading")

Every workload context uses a host resident 64 MB buffer which is memory mapped into the ERT instance created to service the workload. The `ctrlcode` used by the workload is copied into this special memory. This buffer is protected by PASID like all other input/output buffers used by that workload. Instruction buffer is also mapped into the user space of the workload.

### Global Privileged Buffer[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#global-privileged-buffer "Permalink to this heading")

In addition, the driver also allocates a single buffer for maintenance tasks like recording errors from MERT. This global buffer uses the global IOMMU domain and is only accessible by MERT.

High-level Use Flow[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#high-level-use-flow "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------------

Here are the steps to run a workload on AMD NPU:

1.   Compile the workload into an overlay and a `ctrlcode` binary.

2.   Userspace opens a context in the driver and provides the overlay.

3.   The driver checks with the Resource Solver for provisioning a set of columns for the workload.

4.   The driver then asks MERT to create a context on the device with the desired columns.

5.   MERT then creates an instance of ERT. MERT also maps the Instruction Buffer into ERT memory.

6.   The userspace then copies the `ctrlcode` to the Instruction Buffer.

7.   Userspace then creates a command buffer with pointers to input, output, and instruction buffer; it then submits command buffer with the driver and goes to sleep waiting for completion.

8.   The driver sends the command over the Mailbox to ERT.

9.   ERT _executes_ the `ctrlcode` in the instruction buffer.

10.   Execution of the `ctrlcode` kicks off DMAs to and from the host DDR while AMD XDNA Array is running.

11.   When ERT reaches end of `ctrlcode`, it raises an MSI-X to send completion signal to the driver which then wakes up the waiting workload.

Boot Flow[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#boot-flow "Permalink to this heading")
-----------------------------------------------------------------------------------------------------

amdxdna driver uses PSP to securely load signed NPU FW and kick off the boot of the NPU microcontroller. amdxdna driver then waits for the alive signal in a special location on BAR 0. The NPU is switched off during SoC suspend and turned on after resume where the NPU FW is reloaded, and the handshake is performed again.

Userspace components[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#userspace-components "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------

### Compiler[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#compiler "Permalink to this heading")

Peano is an LLVM based open-source single core compiler for AMD XDNA Array compute tile. Peano is available at: [https://github.com/Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie)

IRON is an open-source array compiler for AMD XDNA Array based NPU which uses Peano underneath. IRON is available at: [https://github.com/Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)

### Usermode Driver (UMD)[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#usermode-driver-umd "Permalink to this heading")

The open-source XRT runtime stack interfaces with amdxdna kernel driver. XRT can be found at: [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT)

The open-source XRT shim for NPU is can be found at: [https://github.com/amd/xdna-driver](https://github.com/amd/xdna-driver)

DMA Operation[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#dma-operation "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------

DMA operation instructions are encoded in the `ctrlcode` as `XAIE_IO_BLOCKWRITE` opcode. When ERT executes `XAIE_IO_BLOCKWRITE`, DMA operations between host DDR and L2 memory are effected.

Error Handling[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#error-handling "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------

When MERT detects an error in AMD XDNA Array, it pauses execution for that workload context and sends an asynchronous message to the driver over the privileged channel. The driver then sends a buffer pointer to MERT to capture the register states for the partition bound to faulting workload context. The driver then decodes the error by reading the contents of the buffer pointer.

Telemetry[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#telemetry "Permalink to this heading")
-----------------------------------------------------------------------------------------------------

MERT can report various kinds of telemetry information like the following:

*   L1 interrupt counter

*   DMA counter

*   Deep Sleep counter

*   etc.

References[¶](https://docs.kernel.org/accel/amdxdna/amdnpu.html#references "Permalink to this heading")
-------------------------------------------------------------------------------------------------------

*   [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna.html)

*   [AMD AI Engine Technology](https://www.xilinx.com/products/technology/ai-engine.html)

*   [Peano](https://github.com/Xilinx/llvm-aie)

*   [Versal Adaptive SoC AIE-ML Architecture Manual (AM020)](https://docs.amd.com/r/en-US/am020-versal-aie-ml)

*   [AI Engine Run Time](https://github.com/Xilinx/aie-rt/tree/release/main_aig)

 ©The kernel development community. | Powered by [Sphinx 5.3.0](https://www.sphinx-doc.org/)&[Alabaster 0.7.16](https://alabaster.readthedocs.io/) | [Page source](https://docs.kernel.org/_sources/accel/amdxdna/amdnpu.rst.txt)