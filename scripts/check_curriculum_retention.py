"""Rehearsal resets cannot inflate frontier mastery and must be reproducible."""
from types import SimpleNamespace
from swarm_core.curriculum import HeadingCurriculum

def main():
    config=dict(half_widths_rad=[.15,.6,1.2,1.8,2.4,3.141592653589793],
        window_episodes=4,promotion_success_rate=.75,previous_stage_probability=.75)
    a=HeadingCurriculum(config,seed=12);b=HeadingCurriculum(config,seed=12)
    sa=SimpleNamespace(task={'initialization':{}})
    sb=SimpleNamespace(task={'initialization':{}})
    for _ in range(4):
        a.apply(sa);b.apply(sb)
        a.record(True);b.record(True)
    assert a.stage==b.stage==1
    samples=[];frontier=0
    for _ in range(100):
        a.apply(sa);b.apply(sb)
        assert a.sampled_stage==b.sampled_stage
        samples.append(a.sampled_stage)
        before=len(a.history)
        rehearsal=a.sampled_stage<1
        a.record(rehearsal);b.record(rehearsal)
        if rehearsal:assert len(a.history)==before
        else:frontier+=1
    assert set(samples)=={0,1} and frontier>0
    assert a.stage==b.stage==1
    assert sum(a.history)==0
    print('Seeded previous-stage rehearsal and frontier-only promotion: PASSED')

if __name__=='__main__':main()
