# Gazebo pacing and WSL diagnostics

The live driver now uses one wall-clock pacer, targeting 0.99 by default.
The stepped physics server runs without a second real-time cap. The 5 ms
physics step, 100 ms control interval and policy weights remain unchanged.
Use --real-time-factor 0.99 explicitly, or --fast for unpaced evaluation.
Summary JSON records target and measured real-time factors separately.

The 30 s headless actuator probe measured 0.989925. Graphical policy samples
measured 0.861196 (30 s), 0.977776 (30 s), and 0.932891 (45 s) while VS Code
indexing competed for CPU. These are short performance tests, not full mission
acceptance. A 99% target cannot guarantee sustained 99% on a loaded machine.

Workspace-only VS Code settings disable unnecessary C++ indexing and exclude
virtual environments, experiment outputs, upstream sources and generated ROS
files from file watching. Actual source files remain intact. Current C++ index
workers can finish at low scheduling priority; this does not stop compilation.

WSL intermittently returned E_UNEXPECTED and 0x8007274c while ordinary commands
also sometimes succeeded. Memory and disk were not exhausted in the samples.
Terminating Ubuntu alone was insufficient; wsl --shutdown temporarily recovered
connections. Four new connection probes passed, and VS Code established new
management/extension connections. The timeout later recurred, so a permanent
WSL repair is NOT established. No distribution was unregistered or reinstalled.
No project data or server installation was deleted. If this continues, restart
Windows after saving work and collect Microsoft WSL diagnostics before changing
OS/network/virtualization settings. Do not repeatedly reinstall this project as
a remedy for a host WSL service failure.

See experiments/runs/rtf_before_001, rtf_after_001 and rtf_gui_policy_001..003
for raw measurements. This performance revision does not retrain the policies
or replace the previous checksummed shadow-runtime archive.

Final loaded GUI recheck (20 simulated seconds, run rtf_gui_policy_005):
measured factor 0.908983. The indexer nice level
was reduced to 19. This did not establish sustained 98-99% GUI performance.
The 0.99 configured target remains distinct from the measured achieved rate.

## WSLg renderer verification

The 2026-11-19 check found llvmpipe software rendering by default.
With GALLIUM_DRIVER=d3d12 and MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA,
glxinfo reported D3D12 (NVIDIA GeForce GTX 1650) and Accelerated: yes.
The mixed realtime launcher now probes this combination and selects it
when available, preserving an explicitly set GALLIUM_DRIVER.

Restart the Gazebo GUI through the launcher to load the new renderer.
The existing GUI keeps its previously loaded graphics backend. This check
establishes OpenGL acceleration availability. A complete GPU-rendered
mission benchmark is still needed to quantify the change in measured RTF.

## Physics transport batching test

A headless 11 s actuator probe compared 20-step and 100-step transport batches.
Both kept the 1 ms physics step and 100 ms policy interval.
Measured RTF increased from 0.254 to 0.281. The final platform displacements
were nearly equal, with the Go2 differing by approximately 2 mm.
The modest gain does not establish full mission equivalence or a 0.98 RTF.

SWARM_PHYSICS_CHUNK_STEPS accepts 20 (default), 50 or 100 for diagnostics.
The runtime summary records the value. Keep the default for accepted runs
until a complete mission benchmark validates a change. The remaining
physics/ROS execution cost needs profiling before further tuning.
