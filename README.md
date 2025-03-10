Overview
In this project, we analyzed two production lines at a German car manufacturing plant. These lines correspond to:

Station 120: Equipped with three robot groups, responsible for placing electric batteries onto a trolley (a cart used to transport batteries through the production line).
Station 160: Equipped with two robot groups, responsible for installing the connectors that link the batteries to the trolley.
Problem Statement
We observed high failure rates at both stations:

Station 120: 80% “Not OK” rate

Station 160: 70% “Not OK” rate

Failures at Station 120 often carried over to Station 160, causing further problems with battery placement and connector installation.

Approach
We addressed these issues using:

Data Cleaning and Analysis: We collected and interleaved data from both stations to get a comprehensive view of the process.
Machine Learning: We applied XGBoost and SHAP (SHapley Additive exPlanations) to identify the most influential factors behind the failures.
Root Cause Investigation: Our analysis revealed two main contributing factors:
Screw Process: The parameters for “screw limits” needed adjustment to improve accuracy.
Trolley Variability: Different trolleys, each programmed for a specific route, had varying failure rates.
Results
By optimizing the screw parameters and standardizing the trolley configurations, we significantly reduced the “Not OK” rates at both Station 120 and Station 160. This, in turn, improved overall efficiency and reduced downtime caused by faulty battery placements and connector installations.

Conclusion
Our data-driven approach successfully identified and resolved key process failures in battery placement and connector installation. The insights gained from this project can be applied to similar assembly lines to enhance quality and minimize production errors.

