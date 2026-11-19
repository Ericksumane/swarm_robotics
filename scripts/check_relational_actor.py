"""Check relational features without privileged state or imposed actions."""
import torch
from swarm_core.learning.mappo_adapter import SetBase

def main():
    torch.manual_seed(67)
    base=SetBase(64,neighbor_dim=9,geometry_summary=True)
    local=torch.zeros((2,9))
    rows=torch.randn(2,2,9,requires_grad=True)
    mask=torch.tensor([[1.,0.],[1.,1.]])
    def run(r,m):return base(torch.cat([local,r.reshape(2,-1),m],dim=-1))
    output=run(rows,mask)
    torch.testing.assert_close(output,run(rows.flip(1),mask.flip(1)))
    changed=rows.detach().clone();changed[0,1]=10000.
    torch.testing.assert_close(output[0],run(changed,mask)[0])
    output.square().sum().backward()
    assert rows.grad[0,1].abs().sum()==0
    assert rows.grad[1].abs().sum()>0
    zero=base(torch.zeros(2,9))
    assert zero.shape==(2,64) and torch.isfinite(zero).all()
    print('Relational actor permutation invariance and mask isolation: PASSED')
    print('Relational encoder gradients and empty-neighbor handling: PASSED')

if __name__=='__main__':main()
