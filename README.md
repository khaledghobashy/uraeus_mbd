# URAEUS MBD

An open-source framework for the modeling, simulation and visualization of constrained multi-body systems.

---

## Description

The **uraeus.mbd** is an under-development open-source framework for modeling, simulation and visualization of constrained multi-body systems.

A multi-body system is hereby defined as *a finite number of material bodies connected in an arbitrary fashion by mechanical joints that limit the relative motion between pairs of bodies*. Practitioners of multi-body dynamics study the generation and solution of the equations governing the motion of such systems [1].

This repository aims to provide a continuously updated database of multi-body systems' models, that are developed using the uraeus open-source framework.
The goal is to provide various modeling examples that walks through the typical modeling process that the tool adopts in order to make things familiar.

---

### Audience and Fields of Application

Initially, the main targeted audience was the **Formula Student** community. The motive was *encouraging a deeper understanding of the modeling processes and the underlying theories used in other commercial software packages*, which is a way of giving back to the community, and supporting the concept of *"knowledge share"* adopted there by exposing it to the open-source community as well.

Currently, the tool aims to serve a wider domain of users with different usage goals and different backgrounds, such as students, academic researchers and industry professionals.

Fields of application include any domain that deals with the study of interconnected bodies, such as:

- Ground Vehicles' Systems.
- Construction Equipment.
- Industrial Mechanisms.
- Robotics.
- Biomechanics.
- .. etc.

----

### Features 

#### Symbolic Model Creation

The [**uraeus.smbd**](https://github.com/khaledghobashy/uraeus-smbd) is a python package developed for the symbolic creation and analysis of constrained multi-body systems. Mainly, it provides the user with the ability to:

- Create symbolic template-based and stand-alone multi-body systems using minimal API via python scripting..
- Create complex multi-body assemblies using template-based models.
- Visualize the systems' connectivity/topology as convenient network graphs.
- View the systems' symbolic equations in a natural mathematical format using Latex printing.



#### Code-Generation and Numerical Simulation

**Python Numerical Environment**

[**uraeus.nmbd.python**](https://github.com/khaledghobashy/uraeus-nmbd-python) is a numerical simulation environment developed in python, that generates numerical object-oriented code from symbolic models developed in **[uraeus.smbd](https://github.com/khaledghobashy/uraeus-smbd)**, and provides various numerical solvers to solve for the dynamics and kinematics of multi-body systems.

**C++ Numerical Environment**

*under development ...*



#### 3D Visualization

**Babylon.JS**

[**uraeus.visenv.babylon**](https://github.com/khaledghobashy/uraeus_visenv_babylon) is a browser-based, WebGL visualization environment, that makes use of the **[babylon.js](https://www.babylonjs.com/)** javascript library to easily visualize and animate the models developed in the uraeus framework.

**Blender**

*under development ...*



## References

[1] : McPhee, J.J. Nonlinear Dyn (1996) 9: 73. https://doi.org/10.1007/BF01833294



