package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"math/rand"
	"net/http"
	"time"

	"distributed-cartpole-rl/internal/buffer"
	"distributed-cartpole-rl/internal/cartpole"
)

type Runner struct {
	WorkerID      string
	BufferURL     string
	TrainerURL    string
	BatchEpisodes int
	PolicyRefresh time.Duration
	Seed          int64
	Backoff       time.Duration
	Client        *http.Client
}

func (r *Runner) Run(ctx context.Context) error {
	if r.BatchEpisodes <= 0 {
		return errors.New("batch episodes must be > 0")
	}
	if r.Backoff <= 0 {
		r.Backoff = 500 * time.Millisecond
	}
	client := r.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}

	rng := rand.New(rand.NewSource(r.Seed))
	env := cartpole.NewEnv(rng)
	policy := NewPolicy(DefaultWeights())
	lastPolicyPull := time.Time{}
	var episodeID int

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if r.TrainerURL != "" && (r.PolicyRefresh == 0 || time.Since(lastPolicyPull) >= r.PolicyRefresh) {
			if weights, err := fetchPolicy(client, r.TrainerURL); err == nil {
				policy = NewPolicy(weights)
				lastPolicyPull = time.Now()
			} else {
				log.Printf("policy fetch failed: %v", err)
			}
		}

		trajectories := make([]buffer.Trajectory, 0, r.BatchEpisodes)

		for i := 0; i < r.BatchEpisodes; i++ {
			episodeID++
			state := env.Reset()
			steps := make([]buffer.Step, 0, cartpole.MaxSteps())
			episodeReward := 0.0

			for {
				obs := []float64{state.X, state.XDot, state.Theta, state.ThetaDot}
				action, logProb, value := policy.Action(obs, rng)
				nextState, reward, done := env.Step(action)

				steps = append(steps, buffer.Step{
					Obs:     obs,
					Action:  action,
					Reward:  reward,
					Done:    done,
					LogProb: logProb,
					Value:   value,
				})

				episodeReward += reward
				state = nextState
				if done {
					break
				}
			}

			trajectories = append(trajectories, buffer.Trajectory{
				WorkerID:      r.WorkerID,
				EpisodeID:     episodeID,
				Steps:         steps,
				EpisodeReward: episodeReward,
				CreatedAtMs:   time.Now().UnixMilli(),
			})
		}

		req := buffer.EnqueueRequest{
			BatchSentAtMs: time.Now().UnixMilli(),
			Trajectories:  trajectories,
		}

		status, err := postJSON(client, r.BufferURL+"/enqueue", req)
		if err != nil {
			log.Printf("enqueue failed: %v", err)
			time.Sleep(r.Backoff)
			continue
		}
		if status == http.StatusTooManyRequests {
			time.Sleep(r.Backoff)
		}
	}
}

type policyResponse struct {
	Weights PolicyWeights `json:"weights"`
}

func fetchPolicy(client *http.Client, trainerURL string) (PolicyWeights, error) {
	resp, err := client.Get(trainerURL + "/policy")
	if err != nil {
		return PolicyWeights{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return PolicyWeights{}, errors.New("trainer returned non-200")
	}
	var payload policyResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return PolicyWeights{}, err
	}
	return payload.Weights, nil
}

func postJSON(client *http.Client, url string, payload any) (int, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}
