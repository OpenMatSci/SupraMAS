# Architecture

SupraMAS is organized as a mechanism-informed multi-agent workflow:

1. Strategy-tree construction mines literature-grounded tuning directions.
2. Idea innovation proposes candidate scientific hypotheses.
3. Protocol optimization decomposes ideas into executable experimental plans.
4. Explainability tracing links protocol decisions back to strategy and mechanism evidence.
5. Mechanism-informed Bayesian optimization supports closed-loop parameter selection.

The public package layout follows these boundaries:

- `supramas.strategy_tree`
- `supramas.protocol_optimizer`
- `supramas.parameter_optimization`
- `supramas.explainability`

