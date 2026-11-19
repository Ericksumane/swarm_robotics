"""One local robot per process; reset routing state at each new mission."""
import json,sys,torch
from swarm_core.learning.routed_policy import RoutedPolicy
torch.set_num_threads(1)
policy=RoutedPolicy(torch.load(sys.argv[1],map_location='cpu',weights_only=True))
for line in sys.stdin:
 try:
  request=json.loads(line)
  if request.get('reset'):policy.reset()
  if 'observation' not in request and request.get('reset'):response={'reset':True}
  else:
   action,mode=policy.act(request['observation'],goal_distance_m=request.get('goal_distance_m'));response={'action':action.tolist(),'mode':mode}
 except Exception as error:response={'error':str(error)}
 print(json.dumps(response),flush=True)
