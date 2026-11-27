"""Scenario views (opt-in; docs/tasks/VISUALIZATION.md).

`trace.build_trace` reduces a scenario to a small "scene trace" (a new file kind, not the record format): the
geometry, the eigenrays the simulator uses with their bounce points, the arrivals at each hydrophone and the
models in force. `page2d` draws it with matplotlib; `page3d` writes it into one HTML page drawn with three.js in
the browser (no Python dependency). Nothing here is imported by the simulator, the processing chain or the
live chain, and nothing here changes what they compute.
"""
