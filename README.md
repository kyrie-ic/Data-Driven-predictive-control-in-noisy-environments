# Data-Driven-predictive-control-in-noisy-environments
## Description
In the field of systems control, achieving optimal trajectory tracking with theoretical guarantees on constraint satisfaction remains a significant challenge. This thesis introduces a novel convex combination constraints framework within a data-driven predictive control scheme for unknown \ac{LTI} systems in noisy environments. By incorporating convex combination constraints into the classical \ac{DeePC} algorithm, the proposed algorithm reduces deviations from nominal system behavior, resulting in enhanced tracking performance. The thesis also presents a method for specifying the prediction error bound within this framework, alongside a two-step constraint tightening algorithm that ensures constraint satisfaction without introducing unnecessary conservatism. As a result, the proposed DeePC algorithm effectively achieves reference trajectory tracking under inexact data conditions, with theoretical guarantees on closed-loop constraint satisfaction. The impact of data length on performance is thoroughly analyzed, and a regularization-based method is proposed to improve tracking performance while minimizing data and computational requirements. Additionally, a tuning method is introduced to reduce closed-loop constraint violations by softening the Hankel matrix equality constraints, successfully achieving constraint satisfaction in a numerical sense with proper tuning.

## Project structure
- DeePC Algorithm
- Convex Combination Constraint Framework
- Tuning Method
- README.md

## Program run instruction
1. Clone the repository
2. Open the matlab and navigate to the code directory
3. Open each folder to run the corresponding part
