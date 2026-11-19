# Current requirements and delivery audit

| Requirement | Current implementation | Remaining boundary |
| --- | --- | --- |
| Official MAPPO | Pinned unchanged marlbenchmark/on-policy; project encoder/runners | Runtime supervisors and expert routing are project extensions |
| Three local actors, no leader | Full local cores and three ROS processes | Physical distributed-host trials pending |
| Formation, 5 m, stop | Local phases, fixed own targets, terminal stop supervisor | Finite test evidence; not arbitrary-start guarantee |
| Degraded communication | Synthetic loss/delay/bursts/queues and explicit history | Real network/clock measurements pending |
| Fixed offline policy | Checksummed release; no training updates in runtime | Calibrated heterogeneous-policy acceptance pending |
| ROS/Gazebo | Typed peer message package, full ROS shadow node, live Gazebo physics | Vendor actuator adapters not implemented without specs |
| Localization | Common-frame contract and frame/timestamp rejection | Registration and drift correction on real sensors pending |
| Installation | CPU package and tested fresh-directory installer | Confirmed hardware OS/architecture and SDK installation pending |
| Stops | Terminal latch, predictive veto and software watchdogs | No certified collision avoidance or independent physical stop claim |
| Variable fleet size | Set encoder; explicit rejection outside IDs 0..2 in release | Retrain/redesign and evaluate before adding robots |
| English/math/reproducibility | Documentation, manifests, hashes and failure records | Complete physical experiment report after lab access |

See common_release_validation.md for measured evidence and deployment/README.md
for operational instructions. Passing software checks is not hardware approval.
