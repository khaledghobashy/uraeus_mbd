# URAEUS MBD

An open-source framework for the modeling, simulation and visualization of constrained multi-body systems.

---

## Description

The **uraeus.mbd** is an open-source framework for modeling, simulation and visualization of constrained multi-body systems.

A multi-body system is hereby defined as *a finite number of material bodies connected in an arbitrary fashion by mechanical joints that limit the relative motion between pairs of bodies*. Practitioners of multi-body dynamics study the generation and solution of the equations governing the motion of such systems [1].

This repository aims to provide a continuously updated database of multi-body systems' models, that are developed using the **uraeus.mbd** open-source framework.
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

[**uraeus.nmbd.python**](https://github.com/khaledghobashy/uraeus-nmbd-python) is a numerical simulation environment developed in python, that generates numerical object-oriented code from symbolic models developed using **[uraeus.smbd](https://github.com/khaledghobashy/uraeus-smbd)**, and provides various numerical solvers to solve for the dynamics and kinematics of multi-body systems.

**C++ Numerical Environment**

*under development ...*



#### 3D Visualization

**Babylon.JS**

[**uraeus.visenv.babylon**](https://github.com/khaledghobashy/uraeus_visenv_babylon) is a browser-based, WebGL visualization environment, that makes use of the **[babylon.js](https://www.babylonjs.com/)** javascript library to easily visualize and animate the models developed in the **uraeus.mbd** framework.

**Blender**

*under development ...*

----

## Background

#### What is the problem to be solved?

The primary interest in multi-body dynamics is to analyze the system behavior for given inputs. In analogy with control systems; a multi-body system can be thought as a **_system_** subjected to some **_inputs_** producing some **_outputs_**. These three parts of the problem are dependent on the analyst end goal of the analysis and simulation. 

#### How is the system physics abstracted mathematically?

An unconstrained body in space is normally defined using 6 generalized coordinates defining its location and orientation. For example, a system of 10 bodies requires 60 generalized coordinates to be fully defined, which in turn requires 60 *independent equations* to be solved for these  _unknown_ generalized coordinates.

The way we achieve a solution for the system is dependent on the type of study we are performing. Mainly we have **four types** of analysis that are of interest for a given multi-body system. These are:

- **Kinematic Analysis**</br>
  *"How does the whole system move if we moved this particular body ?"*
- **Inverse Dynamic Analysis**</br>
  *"What are the forces needed to achieve this motion we just did ?"*
- **Equilibrium Analysis**</br>
  *"How does the system look if we did nothing ?"*
- **Dynamic Analysis**</br>
  *"Now we gave it a force, how does it behave ?"*

Each analysis type -or question- can be modelled by a set of algebraic and/or differential equations that can be solved for the system generalized states (positions, velocities and accelerations). A more detailed discussion of each analysis type will be provided in another documentation.

### Approach

The philosophy of the tool is to isolate the model creation process form the actual numerical and computational representation of the system that will be used in the simulation process. This is done through the ideas of **symbolic computing** and **code-generation** as well be shown below.



#### Model Creation

Using [**uraeus.smbd**](https://github.com/khaledghobashy/uraeus-smbd), the topology of a given multi-body system is represented as a multi-directed graph, where each node represents a body and each edge represents a connection between the end nodes, where this connection may represents a joint, actuator or a force element. This serves mainly two aspects:

1. A natural way to create and represent the topology of a given multi-body system.
2. A convenient way to abstract the system programmatically, where all the topological data of the system are stored in a graph.

This is achieved by making heavy use the [NetworkX](https://networkx.github.io/documentation/stable/index.html) python package to create topology graphs and to construct the governing equations of the system. The equations themselves are represented symbolically using [SymPy](https://www.sympy.org/en/index.html), which is a Python library for symbolic mathematics.

The combination of both, NetworkX and SymPy, provides a very simple, easy-to-use and convenient interface for the process of model creation and topology design, where the user only focuses on the validity of the system topology in hand, as he thinks only in terms of the topological components - bodies, joints, actuators and forces-, without the burden of frequent numerical inputs for each component, or how the actual system is configured in space. In short, the tool divide the typical model creation process in halves, the system topology design and the system configuration assignment.

#### Code Generation and Model Simulation

The process of performing actual simulations on the created model requires the generation of a valid numerical and computational code of the developed model. This is done by taking in the symbolic model and create a valid code base written in the desired programming language with the desired programming paradigm and structure.

Each numerical environment is responsible for the translation of the developed symbolic models into valid numerical code, and for the features it aims to provide for the users.

The development of such environments in different languages requires a good grasp of several aspects such as :

- Good knowledge of the **uraeus.smbd** symbolic models' interfaces and structure.
- Good knowledge of the target language.
- Appropriate environment architecture/structure that serves the intended usage requirements.
- Good knowledge of the available linear algebra and math libraries for that language.
- Design for minimal dependencies on 3rd parties libraries.
- Simple API for usage and simple build process for compiled languages.

_**Note**: The development of such environments will be discussed in a separate documentation for those interested in developing their own._

#### Conclusion

Several benefits of the adopted approach can be stated here, but the major theme here is the flexibility and modularity, in both software usage and software development. These can be summarized as follows:

- The distinction between the topology design phase and the configuration assignment phase, which gives proper focus for each at its' own.
- Natural adoption of the template-based modeling theme that emerges from the use of network-graphs to represent the system, which allows convenient assemblage of several graphs to form a new system. 
- Uncoupled simulation environment, where the symbolic equations generated form the designed topology is free to be written in any programming language with any desired numerical libraries.



---

## The Modeling and Simulation Process



## References

[1] : McPhee, J.J. Nonlinear Dyn (1996) 9: 73. https://doi.org/10.1007/BF01833294



