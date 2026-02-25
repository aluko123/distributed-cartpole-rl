package main

import (
	"context"
	"log"
	"os"
	"strconv"
	"time"

	"distributed-cartpole-rl/internal/worker"
)

const (
	defaultBufferURL  = "http://localhost:9001"
	defaultTrainerURL = "http://localhost:9002"
)

func main() {
	workerID := getenv("WORKER_ID", "worker-"+strconv.FormatInt(time.Now().UnixNano(), 10))
	bufferURL := getenv("BUFFER_URL", defaultBufferURL)
	trainerURL := getenv("TRAINER_URL", defaultTrainerURL)
	batchEpisodes := getenvInt("BATCH_EPISODES", 8)
	policyRefresh := time.Duration(getenvInt("POLICY_REFRESH_SEC", 5)) * time.Second
	seed := getenvInt64("SEED", time.Now().UnixNano())
	backoff := time.Duration(getenvInt("BACKOFF_MS", 500)) * time.Millisecond

	runner := &worker.Runner{
		WorkerID:      workerID,
		BufferURL:     bufferURL,
		TrainerURL:    trainerURL,
		BatchEpisodes: batchEpisodes,
		PolicyRefresh: policyRefresh,
		Seed:          seed,
		Backoff:       backoff,
	}

	if err := runner.Run(context.Background()); err != nil {
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

func getenvInt64(key string, fallback int64) int64 {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fallback
	}
	return parsed
}
