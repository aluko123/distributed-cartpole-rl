package buffer

type Step struct {
	Obs     []float64 `json:"obs"`
	Action  int       `json:"action"`
	Reward  float64   `json:"reward"`
	Done    bool      `json:"done"`
	LogProb float64   `json:"log_prob"`
	Value   float64   `json:"value"`
}

type Trajectory struct {
	WorkerID      string  `json:"worker_id"`
	EpisodeID     int     `json:"episode_id"`
	Steps         []Step  `json:"steps"`
	EpisodeReward float64 `json:"episode_reward"`
	CreatedAtMs   int64   `json:"created_at_ms"`
}

type EnqueueRequest struct {
	BatchSentAtMs int64        `json:"batch_sent_at_ms"`
	Trajectories  []Trajectory `json:"trajectories"`
}

type DequeueResponse struct {
	Trajectories []Trajectory `json:"trajectories"`
}
