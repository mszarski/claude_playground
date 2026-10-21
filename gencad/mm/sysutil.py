"""Process-level helpers."""
import os


def usable_cpus():
    """CPUs this process may actually use: the cgroup CPU quota if one is set, else the affinity mask.
    Inside HF Jobs containers os.cpu_count() reports the host (e.g. 64 on an 8-vCPU job); sizing worker pools from it
    oversubscribes the CPUs and makes CAD builds time out."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))
