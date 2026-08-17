"""ATEX - Accessible Travel Expert.

A supervisor multi-agent system that plans accessible trips and reports
accessibility as verified, flagged, or explicitly unknown -- never guessed.
"""

__version__ = "0.1.0"

# The canonical module names. These strings are the single source of truth and
# must appear identically in the architecture PNG, in every /api/execute step,
# and in /api/agent_info. tests/test_module_names.py enforces this.
SUPERVISOR = "Supervisor"
USER_PROFILE_AGENT = "UserProfileAgent"
ACTIVITY_LOGISTICS_FINDER = "ActivityLogisticsFinder"
ACCESSIBILITY_VALIDATOR = "AccessibilityValidator"
SCHEDULE_PLANNER = "SchedulePlanner"

MODULE_NAMES = (
    SUPERVISOR,
    USER_PROFILE_AGENT,
    ACTIVITY_LOGISTICS_FINDER,
    ACCESSIBILITY_VALIDATOR,
    SCHEDULE_PLANNER,
)
