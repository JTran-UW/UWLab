<!-- <p align="center">
  <img src="docs/source/_static/uwlab.png" alt="Isaac Lab">
</p>

---
-->
# UW Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-yellow.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

### Dextrous Manipulation Tasks
[`source/uwlab_tasks/uwlab_tasks/manager_based/manipulation`](https://github.com/UW-Lab/UWLab/tree/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation)

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/ur5/track_goal_ur5_env_cfg.py"><img src="https://uw-lab.github.io/UWLab/main/_images/ur5_track_goal.jpg"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/tycho/tycho_track_goal.py"><img src="https://uw-lab.github.io/UWLab/main/_images/tycho_track_goal.jpg"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/xarm_leap/track_goal_xarm_leap.py"><img src="https://uw-lab.github.io/UWLab/main/_images/xarm_leap_track_goal.jpg"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/factory_extension/gearmesh_env_cfg.py"><img src="https://uw-lab.github.io/UWLab/main/_images/gear_mesh_ext.jpg"></a></td>
        </tr>
        <tr>
            <td align="center">Ur5 w/ Robotiq Hand</td>
            <td align="center">Tycho Dextrous Chopsticks Manipulator</td>
            <td align="center">X-arm w/ Leap Hand</td>
            <td align="center">Franka with NIST gear mesh task</td>
        </tr>
</table>

### Advanced Locomotion Skills
[`source/uwlab_tasks/uwlab_tasks/manager_based/locomotion`](https://github.com/UW-Lab/UWLab/tree/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion)

> _Note:_ We focus solely on Boston Dynamics Spot as this is the hardware we currently have set up for sim2real testing.

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py"><img src="https://github.com/UW-Lab/UWLab/blob/main/docs/source/_static/tasks/locomotion/spot_gap.gif"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py"><img src="https://github.com/UW-Lab/UWLab/blob/main/docs/source/_static/tasks/locomotion/spot_pit.gif"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/risky_terrains/config/spot/spot_env_cfg.py"><img src="https://github.com/UW-Lab/UWLab/blob/main/docs/source/_static/tasks/locomotion/spot_stepping_stone.gif"></a></td>
            <td><a href="https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py"><img src="https://github.com/UW-Lab/UWLab/blob/main/docs/source/_static/tasks/locomotion/spot_inv_slope.gif"></a></td>
        </tr>
        <tr>
            <td align="center">Jumping Gaps</td>
            <td align="center">Climbing Cliffs</td>
            <td align="center">Precision Stepping Stones</td>
            <td align="center">Escaping Sloped Pits</td>
        </tr>
</table>

## Overview

**UW Lab**, is our open source project that builds upon the robust foundation established by Isaac Lab. This effort is organized by the UW (University of Washington) Robotic Students, across labs, and aims to create an centralized repository for high-quality, tested research ready for publication that resides in the IsaacLab Ecosystem. This repo is designed and structured to reuse the toolkit of ongoing IsaacLab development, track a strictly maintained IsaacLab version, and the relevant extensions required for our research on hardware here at UW.

## Key Features

In addition to what IsaacLab provides, UW Lab brings:

- **Environments**: Clean Implementation of tested environments in the ManagerBased format. We both implement our own novel settings and reproduce results from our favorite papers.
- **Sim to Real**: Providing robots and configurations that have been tested in the UW Robotics Labs, demonstrating effective sim2real transfer.


## Getting Started

Our [documentation page](https://uw-lab.github.io/UWLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://uw-lab.github.io/UWLab/main/source/setup/installation/local_installation.html)
- [Available environments](https://uw-lab.github.io/UWLab/main/source/overview/uw_environments.html)


## Contributing to UW Lab

Please refer to Isaac Lab
[contribution guideline](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).


## Troubleshooting

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

or bugs and troubleshooting specific to the UWLab environments, please [submit an issue](https://github.com/UW-Lab/UWLab/issues).

## Support

* Please use GitHub [Discussions](https://github.com/UW-Lab/UWLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/UW-Lab/UWLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

UW Lab is released under [BSD-3 License](LICENSE). The Isaac Lab framework is released under [BSD-3 License](LICENSE).

## Acknowledgement

If you found UWLab useful, we appreciate if you cite it in academic publications:
```
@software{zhang2025uwlab,
  author={Zhang, Zhengyu and Yu, Feng and Castro, Mateo and Yin, Patrick and Peng, Quanquan and Scalise, Rosario},
  title={{UWLab}: Environments for robotics research at the edge of reinforcement learning, imitation learning, and sim2real,
  year={2025},
  url={https://github.com/UW-Lab/UWLab}
}
```
UW Lab originated thanks to the ongoing community effort of Isaac Lab. Our repository tracks it closely. As a gratitude we also appreciate if you cite Isaac Lab in academic publications:
```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
