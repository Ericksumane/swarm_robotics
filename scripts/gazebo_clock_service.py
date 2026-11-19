"""Isolate synchronous Gazebo requests from Python subscriber callbacks."""
import json,sys
from gz.transport13 import Node
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
node=Node()
for line in sys.stdin:
 request=json.loads(line);message=WorldControl();message.pause=True;message.multi_step=request['steps']
 success,response=node.request('/world/swarm_live/control',message,WorldControl,Boolean,request['timeout'])
 print(json.dumps(dict(accepted=bool(success and response.data),transport_success=bool(success),response=bool(response.data))),flush=True)
