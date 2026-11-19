"""Strict versioned peer envelope shared by ROS and independent core tests."""
from dataclasses import dataclass,asdict
import json,math,re
from swarm_core.communication.message import StateMessage

VERSION=1
TOKEN=re.compile(r"[A-Za-z0-9_.:/-]{1,96}\Z")

def token(value):
    if not isinstance(value,str) or not TOKEN.fullmatch(value):raise ValueError('Invalid mission, session or frame identifier.')
    return value

@dataclass(frozen=True)
class PeerEnvelope:
    mission_id:str
    session_id:str
    frame_id:str
    state:StateMessage
    faulted:bool=False
    protocol_version:int=VERSION

    def __post_init__(self):
        for value in (self.mission_id,self.session_id,self.frame_id):token(value)
        if type(self.protocol_version) is not int or self.protocol_version!=VERSION:raise ValueError('Unsupported peer protocol version.')
        if type(self.faulted) is not bool:raise ValueError('Fault flag must be boolean.')
        if type(self.state.formation_ready) is not bool:raise ValueError('Ready flag must be boolean.')
        if self.state.sender_id not in (0,1,2):raise ValueError('This release supports robot IDs 0,1,2 only.')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)!={'mission_id','session_id','frame_id','state','faulted','protocol_version'}:raise ValueError('Peer envelope fields differ from protocol v1.')
        state=value['state']
        if not isinstance(state,dict) or set(state)!=set(StateMessage.__dataclass_fields__):raise ValueError('Peer state fields differ from protocol v1.')
        return cls(**{**value,'state':StateMessage(**state)})
