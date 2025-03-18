# System Architecture Diagrams

This document provides visual representations of the Portfolio Optimization Testbed architecture to help developers understand the system structure and component interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Portfolio Optimization Testbed                      │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                 ┌──────────────────┬┴─────────────────────┐
                 │                  │                      │
    ┌────────────▼─────────┐ ┌─────▼────────┐ ┌───────────▼────────────┐
    │     Core Library     │ │   Frontend   │ │       Utilities         │
    │  (Python Package)    │ │  (Dashboard) │ │ (Data & Visualization)  │
    └────────────┬─────────┘ └─────┬────────┘ └───────────┬────────────┘
                 │                 │                      │
                 └─────────────────┼──────────────────────┘
                                   │
                        ┌──────────▼─────────┐
                        │     Test Suite     │
                        └────────────────────┘
```

## Core Library Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Core Library                                   │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────────────┬───────────────┐
    │           │           │                   │               │
┌───▼───┐ ┌─────▼────┐ ┌────▼─────┐ ┌──────────▼────────┐ ┌────▼─────┐
│Problem│ │Objectives│ │Constraints│ │     Solvers      │ │ Metrics  │
└───┬───┘ └─────┬────┘ └────┬─────┘ └──────────┬────────┘ └────┬─────┘
    │           │           │                  │                │
    │           │           │                  │                │
    │      ┌────▼───────────▼──────────────────▼────────────┐  │
    └──────►                                                │  │
           │              Optimization Engine               │◄─┘
           │                                                │
           └────────────────────┬───────────────────────────┘
                                │
                         ┌──────▼─────────┐
                         │ Solution Result │
                         └────────────────┘
```

## Data Flow Diagram

```
┌───────────┐         ┌───────────────┐         ┌───────────────┐
│           │         │               │         │               │
│  Input    │────────►│  Problem      │────────►│  Solver       │
│  Data     │         │  Definition   │         │  Engine       │
│           │         │               │         │               │
└───────────┘         └───────┬───────┘         └───────┬───────┘
                              │                         │
                              │                         │
                              │                         ▼
┌───────────┐         ┌──────▼────────┐         ┌───────────────┐
│           │         │               │         │               │
│  Results  │◄────────│  Solution     │◄────────│  Optimization │
│  Analysis │         │  Processing   │         │  Process      │
│           │         │               │         │               │
└───────────┘         └───────────────┘         └───────────────┘
```

## Component Interactions

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  PortfolioOpt   │────►│  Constraints    │────►│  Objective      │
│  Problem        │     │                 │     │  Function       │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                         Solver                                  │
│                                                                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      Optimization Result                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ HTTP/WebSocket
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                        Dashboard Server                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ Python API
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                     Portfolio Optimization Core                 │
└─────────────────────────────────────────────────────────────────┘
```

## Class Diagram (Core Components)

```
┌───────────────────────┐      ┌───────────────────────┐
│ PortfolioOptProblem   │      │ Constraint            │
├───────────────────────┤      ├───────────────────────┤
│ - returns             │      │ + validate()          │
│ - volumes             │      │ + evaluate()          │
│ - market_impact_model │      └─────────┬─────────────┘
├───────────────────────┤                │
│ + get_covariance()    │                │
│ + validate()          │      ┌─────────▼─────────────┐
└──────────┬────────────┘      │ FullInvestmentConstraint
           │                   ├───────────────────────┤
           │                   └───────────────────────┘
           │
           │                   ┌───────────────────────┐
           │                   │ Objective             │
           │                   ├───────────────────────┤
           │                   │ + evaluate()          │
           │                   └─────────┬─────────────┘
           │                             │
           │                   ┌─────────▼─────────────┐
           │                   │ MinimumVarianceObjective
           │                   ├───────────────────────┤
           │                   └───────────────────────┘
           │
           │                   ┌───────────────────────┐
           │                   │ Solver                │
           └───────────────────► + solve()             │
                               └─────────┬─────────────┘
                                         │
                               ┌─────────▼─────────────┐
                               │ ClassicalSolver       │
                               ├───────────────────────┤
                               └───────────────────────┘
```

## Sequence Diagram (Optimization Process)

```
┌─────────┐          ┌──────────┐          ┌─────────┐          ┌───────────┐
│ Client  │          │ Problem  │          │ Solver  │          │ Constraint │
└────┬────┘          └────┬─────┘          └────┬────┘          └─────┬─────┘
     │                    │                     │                     │
     │ create             │                     │                     │
     │ ──────────────────>│                     │                     │
     │                    │                     │                     │
     │ solve              │                     │                     │
     │ ────────────────────────────────────────>│                     │
     │                    │                     │                     │
     │                    │    get_covariance   │                     │
     │                    │ <───────────────────│                     │
     │                    │                     │                     │
     │                    │ covariance_matrix   │                     │
     │                    │ ──────────────────> │                     │
     │                    │                     │                     │
     │                    │                     │ validate            │
     │                    │                     │ ───────────────────>│
     │                    │                     │                     │
     │                    │                     │ valid/invalid       │
     │                    │                     │ <───────────────────│
     │                    │                     │                     │
     │                    │                     │ optimize            │
     │                    │                     │ ────┐               │
     │                    │                     │     │               │
     │                    │                     │ <───┘               │
     │                    │                     │                     │
     │ result             │                     │                     │
     │ <────────────────────────────────────────│                     │
     │                    │                     │                     │
┌────┴────┐          ┌────┴─────┐          ┌────┴────┐          ┌─────┴─────┐
│ Client  │          │ Problem  │          │ Solver  │          │ Constraint │
└─────────┘          └──────────┘          └─────────┘          └───────────┘
```

## Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Environment                         │
│                                                                 │
│   ┌─────────────┐           ┌─────────────┐                     │
│   │             │           │             │                     │
│   │  Web        │           │  Jupyter    │                     │
│   │  Browser    │           │  Notebook   │                     │
│   │             │           │             │                     │
│   └──────┬──────┘           └──────┬──────┘                     │
│          │                         │                            │
└──────────┼─────────────────────────┼────────────────────────────┘
           │                         │
           │ HTTP/                   │ Python
           │ WebSocket               │ API
           │                         │
┌──────────▼─────────────────────────▼────────────────────────────┐
│                                                                 │
│                    Portfolio Optimization Server                │
│                                                                 │
│   ┌─────────────────────────────────────────────────────┐       │
│   │                                                     │       │
│   │              Dashboard Web Server                   │       │
│   │                                                     │       │
│   └───────────────────────┬─────────────────────────────┘       │
│                           │                                     │
│   ┌───────────────────────▼─────────────────────────────┐       │
│   │                                                     │       │
│   │           Portfolio Optimization Core               │       │
│   │                                                     │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│  portopt.core │────►│ portopt.solvers│────►│portopt.metrics│
│               │     │               │     │               │
└───────┬───────┘     └───────────────┘     └───────────────┘
        │
        │
        │             ┌───────────────┐     ┌───────────────┐
        │             │               │     │               │
        └────────────►│portopt.utils  │────►│portopt.viz    │
                      │               │     │               │
                      └───────┬───────┘     └───────────────┘
                              │
                              │
                              │             ┌───────────────┐
                              │             │               │
                              └────────────►│portopt.data   │
                                            │               │
                                            └───────────────┘
```

## Interactive Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dashboard UI                             │
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│   │             │   │             │   │             │           │
│   │  Input      │   │  Portfolio  │   │  Results    │           │
│   │  Panel      │   │  Viewer     │   │  Panel      │           │
│   │             │   │             │   │             │           │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│          │                 │                 │                  │
└──────────┼─────────────────┼─────────────────┼──────────────────┘
           │                 │                 │
           │                 │                 │
┌──────────▼─────────────────▼─────────────────▼──────────────────┐
│                                                                 │
│                       Dashboard Backend                         │
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│   │             │   │             │   │             │           │
│   │  Data       │   │  Optimizer  │   │  Results    │           │
│   │  Manager    │   │  Interface  │   │  Generator  │           │
│   │             │   │             │   │             │           │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│          │                 │                 │                  │
└──────────┼─────────────────┼─────────────────┼──────────────────┘
           │                 │                 │
           │                 │                 │
┌──────────▼─────────────────▼─────────────────▼──────────────────┐
│                                                                 │
│                  Portfolio Optimization Core                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Notes on Architecture

1. **Modular Design**: The system is designed with a modular architecture to allow for easy extension and maintenance. Each component has a well-defined responsibility and interface.

2. **Core Library**: The central component is the core library, which implements the portfolio optimization algorithms, problem definitions, and constraints.

3. **Frontend**: The dashboard provides a user-friendly interface for configuring and running optimizations, as well as visualizing results.

4. **Extension Points**: The system is designed to be extensible, with clear interfaces for adding new solvers, constraints, objectives, and metrics.

5. **Testing**: A comprehensive test suite ensures the correctness and performance of the system.

For more detailed information about the system architecture, see the [Architecture Overview](./architecture.md) document.
