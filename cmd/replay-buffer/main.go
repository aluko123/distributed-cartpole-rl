package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"distributed-cartpole-rl/internal/buffer"
)

const (
	defaultCapacity = 2048
	defaultPort     = "9001"
)

func main() {
	flag.Parse()

	capacity := getenvInt("BUFFER_CAPACITY", defaultCapacity)
	policy := getenv("BUFFER_POLICY", "fifo")
	port := getenv("PORT", defaultPort)

	replay, err := buffer.NewReplayBuffer(capacity, policy)
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		payload := map[string]any{
			"queue_length": replay.Size(),
			"capacity":     replay.Capacity(),
			"policy":       replay.Policy(),
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(payload); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	})
	mux.HandleFunc("/config", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			payload := map[string]any{
				"policy":   replay.Policy(),
				"capacity": replay.Capacity(),
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(payload); err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
		case http.MethodPost:
			var payload map[string]any
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if value, ok := payload["policy"]; ok {
				policyValue, ok := value.(string)
				if !ok {
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if err := replay.SetPolicy(policyValue); err != nil {
					w.WriteHeader(http.StatusBadRequest)
					return
				}
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/enqueue", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req buffer.EnqueueRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		now := time.Now()

		var dropped bool
		for _, traj := range req.Trajectories {
			item := buffer.Item{Trajectory: traj, EnqueuedAt: now}
			if err := replay.Enqueue(item); err != nil {
				dropped = true
				continue
			}
		}

		if dropped {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusAccepted)
	})
	mux.HandleFunc("/dequeue", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		batchSize := 0
		if value := r.URL.Query().Get("batch_size"); value != "" {
			if parsed, err := strconv.Atoi(value); err == nil {
				batchSize = parsed
			}
		}
		if batchSize <= 0 {
			batchSize = 1
		}

		items := make([]buffer.Item, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			item, err := replay.Dequeue()
			if err != nil {
				break
			}
			items = append(items, item)
		}
		if len(items) == 0 {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		response := buffer.DequeueResponse{Trajectories: make([]buffer.Trajectory, 0, len(items))}
		for _, item := range items {
			response.Trajectories = append(response.Trajectories, item.Trajectory)
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	})

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("replay buffer listening on :%s (capacity=%d policy=%s)", port, capacity, policy)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func getenv(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

func getenvInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}
