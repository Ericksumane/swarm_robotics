# Protocol v1. Header frame is the agreed shared planar pose frame.
# Header stamp is the original local odometry measurement time.
std_msgs/Header header
uint16 protocol_version
string mission_id
string session_id
uint16 sender_id
uint64 sequence
float64 x
float64 y
float64 heading
float64 speed
float64 yaw_rate
bool formation_ready
bool has_start_proposal
float64 start_at_s
bool start_committed
bool faulted
