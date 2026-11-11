# MIBO Optimization Experiment Analysis Report

---

## 1. Executive Summary

This report summarizes the Bayesian-optimization exploration of the process-parameter space for REBCO superconducting thin films. Y1–Y5 denote the 1st–5th Optimisation conditions in the SI. Tc values are the mean M–T values from three independent films per condition; Jc values are reported at the temperature and magnetic field specified below. The results show application-dependent performance differences and suggest nonlinear sensitivity to the process parameters.

- **Y2 (preferred for 77 K operation):** This sample gives the best liquid-nitrogen-temperature performance, with Jc approximately 2.79 MA/cm2 and mean M–T Tc of 88.63 K. It represents an optimization direction dominated by high crystallinity, low defect density and high film density.
- **Y3 (strong 4.2 K high-field performance):** Among Y1–Y4, this sample gives the highest Jc at 4.2 K and 5 T, 12.78 MA/cm2. It represents an optimization direction dominated by strong flux pinning.
- **Y5 (highest final 4.2 K, 5 T performance):** The 5th Optimisation condition gives Jc of 14.11 MA/cm2 at 4.2 K and 5 T, Jc of 0.65 MA/cm2 at 77 K and 0 T, and mean M–T Tc of 87.26 K.
- **Y1 (lower current-carrying performance):** Although its growth parameters are close to those of Y3, its performance decreases sharply. The analysis suggests that a small increase in growth oxygen pressure crossed a thermodynamic threshold for c-axis/a-axis growth competition.

---

## 2. Key Parameters and Performance

| Sample | Growth temperature (deg C) | Growth oxygen pressure (Pa) | Annealing pressure (Pa) | Tc (M–T, K) | Jc (4.2 K, 5 T) | Jc (20 K, 5 T) | Jc (77 K, 0 T) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SupraMAS-150 nm | 675 | 26.7 | 80,000 | 84.62 | 3.78 | 2.32 | 0.10 |
| SupraMAS-280 nm | 675 | 26.7 | 80,000 | 86.95 | 11.25 | 6.84 | 0.89 |
| Y1 (1st Optimisation) | 705 | 26.3 | 22,000 | 86.40 | 3.60 | 1.14 | 0.17 |
| Y2 (2nd Optimisation) | 675 | 29.6 | 82,000 | 88.63 | 7.99 | 3.79 | 2.79 |
| Y3 (3rd Optimisation) | 706 | 24.8 | 19,000 | 85.73 | 12.78 | 6.44 | 0.42 |
| Y4 (4th Optimisation) | 709 | 25.1 | 62,000 | 87.20 | 11.21 | 5.66 | 0.31 |
| Y5 (5th Optimisation) | 689 | 26.5 | 55,000 | 87.26 | 14.11 | 7.67 | 0.65 |

All Jc values are in MA/cm2.

---

## 3. Mechanistic Analysis

### 3.1 Why do Y1 and Y3 behave so differently despite similar growth temperatures?

Y1 (705 deg C, 26.3 Pa) and Y3 (706 deg C, 24.8 Pa) have nearly identical growth temperatures, but their transport performance differs strongly.

#### Mechanism A: Orientation competition induced by growth oxygen pressure

- **Mechanistic basis:** Higher growth oxygen pressure can reduce the nucleation barrier and make a-axis nucleation competitive with c-axis nucleation, leading to mixed-orientation growth.
- **Interpretation:** Around 706 deg C, approximately 25 Pa may act as a critical boundary for maintaining pure c-axis growth.
  - **Y3 (24.8 Pa):** Located below the inferred boundary, preserving lower supersaturation and favoring thermodynamically stable c-axis growth.
  - **Y1 (26.3 Pa):** Slightly above the inferred boundary, potentially triggering rapid a-axis grain nucleation and interrupting in-plane superconducting current paths.

#### Mechanism B: Intermediate annealing-pressure window

- **Y3 (19 kPa):** Low-pressure annealing may promote rapid removal of volatile or trapped gas while retaining a moderate number of oxygen vacancies that can serve as pinning centers.
- **Y1 (22 kPa):** This intermediate-pressure condition may neither remove gas as efficiently as the lower-pressure condition nor suppress pore formation as effectively as the high-pressure condition used for Y2, leading to a loose microstructure and secondary-phase formation.

---

### 3.2 Successful Regimes

#### Y2: Best 77 K performance through low-temperature growth and high-pressure annealing

- **High-pressure annealing (82 kPa):** A near-atmospheric annealing environment can suppress large pores, reduce porosity and improve film density.
- **Lower growth temperature (675 deg C):** Lower-temperature growth may slow interfacial reactions and reduce unwanted diffusion into the superconducting layer, supporting a higher Tc.

#### Y3: Strong 4.2 K high-field performance through high-temperature growth and low-pressure annealing

- **Point-defect pinning:** Strong Jc at 4.2 K is consistent with enhanced flux pinning. Lower annealing pressure may preserve oxygen vacancies or other small lattice defects. These defects can act as weak links at 77 K, but as effective isotropic pinning centers at 4.2 K.
- **Nanophase precipitation:** Higher growth temperature may promote self-assembly or precipitation of nanoscale secondary phases, providing additional lattice-distortion fields and enhancing high-field performance.

---

## 4. Parameter-Space Logic

The experiment reveals a non-convex and boundary-sensitive process space.

1. **Growth-temperature modulation of oxygen-pressure sensitivity:**
   - At lower growth temperature (<680 deg C), the system appears more robust against a-axis impurity growth and is better suited for high-Tc and 77 K operation.
   - At higher growth temperature (>700 deg C), the system becomes highly sensitive to growth oxygen pressure. The model should strongly penalize conditions above approximately 25 Pa in this region.

2. **Application-dependent trade-off:**
   - For **77 K applications**, the priority is film density and phase purity, favoring lower growth temperature and high-pressure annealing.
   - For **4.2 K high-field applications**, the priority is a strong pinning landscape, favoring higher growth temperature and lower-pressure annealing with controlled defect preservation.

---

## 5. Recommended Next Experiments

- **Strategy A: Boundary validation for 4.2 K operation**
  - Starting from Y3, keep the growth temperature near 706 deg C and reduce the growth oxygen pressure to 20-22 Pa.
  - Goal: test whether pinning can be improved while maintaining c-axis growth.

- **Strategy B: Balanced improvement for 77 K operation**
  - Starting from Y2, keep the annealing pressure near 80 kPa and increase the growth temperature to 685-690 deg C.
  - Goal: introduce a limited number of nanoscale precipitates while preserving high density and Tc.

- **Strategy C: Model calibration**
  - Treat Y1 as an important negative example in the Bayesian-optimization model.
  - Suggested constraint: when growth temperature is above 700 deg C, increase the penalty on growth oxygen pressure and avoid exploration above approximately 25 Pa.
