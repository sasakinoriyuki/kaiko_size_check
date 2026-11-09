"""Frame-local visible-instance estimates, independent of measurable lengths."""


def count_summary(candidates):
    counted=[c for c in candidates if c.counted]
    measured=sum(c.length_px is not None for c in counted)
    return {'estimated_count':len(counted),'measured_count':measured,
            'unmeasured_count':len(counted)-measured,
            'uncertain_regions':len(candidates)-len(counted),
            'count_status':'visible_estimate'}


def assign_instance_ids(candidates):
    """Numbers identify this frame only; never imply persistent animal tracking."""
    number=0
    for c in candidates:
        c.instance_id=None
        if c.counted:
            number+=1;c.instance_id=number
    return count_summary(candidates)
